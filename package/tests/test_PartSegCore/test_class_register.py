from typing import Any, Dict

import pytest
from local_migrator import REGISTER, class_to_str, register_class, rename_key, update_argument


@register_class
class SampleClass1:
    pass


def rename_a_to_c(dkt: Dict[str, Any]) -> Dict[str, Any]:
    dkt = dict(dkt)
    dkt["c"] = dkt["a"]
    del dkt["a"]
    return dkt


@register_class(version="0.0.1", migrations=[("0.0.1", rename_a_to_c)])
class SampleClass2:
    pass


class SampleClass3:
    pass


@register_class(old_paths=["test.test.BBase"], version="0.0.2")
class SampleClass4:
    pass


class SampleClass6:
    pass


def test_migrate():
    assert REGISTER.migrate_data(class_to_str(SampleClass1), {}, {"a": 1, "b": 2}) == {"a": 1, "b": 2}
    assert REGISTER.migrate_data(
        class_to_str(SampleClass2), {class_to_str(SampleClass2): "0.0.1"}, {"a": 1, "b": 2}
    ) == {"a": 1, "b": 2}
    assert REGISTER.migrate_data(class_to_str(SampleClass2), {}, {"a": 1, "b": 2}) == {"c": 1, "b": 2}


def test_unregistered_class():
    assert REGISTER.get_class(class_to_str(SampleClass3)) is SampleClass3


def test_old_paths():
    assert REGISTER.get_class("test.test.BBase") is SampleClass4


def test_old_class_error():
    with pytest.raises(RuntimeError):
        register_class(SampleClass6, old_paths=[class_to_str(SampleClass4)])


def test_get_version():
    assert str(REGISTER.get_version(SampleClass1)) == "0.0.0"
    assert str(REGISTER.get_version(SampleClass2)) == "0.0.1"
    assert str(REGISTER.get_version(SampleClass4)) == "0.0.2"


def test_rename_key():
    dkt = {"aaa": 1, "bbb": 2}
    assert rename_key(from_key="aaa", to_key="ccc")(dkt) == {"bbb": 2, "ccc": 1}
    with pytest.raises(KeyError):
        rename_key("ccc", "ddd")(dkt)

    assert rename_key("ccc", "ddd", optional=True)(dkt) == dkt


def test_update_argument(clean_register):
    @REGISTER.register(version="0.0.1")
    class MigrateClass:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    class ClassToCall:
        __argument_class__ = MigrateClass

        @classmethod
        @update_argument("arg")
        def call_func(cls, aa, arg: MigrateClass):
            assert arg.a == 1
            assert arg.b == 2
            assert aa == 1

    ClassToCall.call_func(aa=1, arg={"a": 1, "b": 2})
    ClassToCall.call_func(1, {"a": 1, "b": 2})


def test_migrate_parent_class(clean_register):
    @register_class(version="0.0.1", migrations=[("0.0.1", rename_key("field", "field1"))])
    class BaseMigrateClass:
        def __init__(self, field1):
            self.field1 = field1

    @register_class
    class MigrateClass(BaseMigrateClass):
        def __init__(self, field2, **kwargs):
            super().__init__(**kwargs)
            self.field2 = field2

    migrated = REGISTER.migrate_data(class_to_str(MigrateClass), {}, {"field2": 1, "field": 5})
    assert "field1" in migrated
