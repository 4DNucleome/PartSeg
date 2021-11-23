from typing import Any, Dict

import pytest

from PartSegCore.class_register import REGISTER, class_to_str, register_class


@register_class
class SampleClass1:
    pass


def rename_a_to_c(dkt: Dict[str, Any]) -> Dict[str, Any]:
    dkt = dict(dkt)
    dkt["c"] = dkt["a"]
    del dkt["a"]
    return dkt


@register_class(migrations=[("0.0.1", rename_a_to_c)])
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
    assert REGISTER.migrate_data(class_to_str(SampleClass1), "0.0.0", {"a": 1, "b": 2}) == {"a": 1, "b": 2}
    assert REGISTER.migrate_data(class_to_str(SampleClass2), "0.0.1", {"a": 1, "b": 2}) == {"a": 1, "b": 2}
    assert REGISTER.migrate_data(class_to_str(SampleClass2), "0.0.0", {"a": 1, "b": 2}) == {"c": 1, "b": 2}


def test_unregistered_class():
    assert REGISTER.get_class(class_to_str(SampleClass3)) is SampleClass3


def test_old_paths():
    assert REGISTER.get_class("test.test.BBase") is SampleClass4


def test_import_part():
    obj = REGISTER.get_class("test_PartSegCore.class_register_util.SampleClass5")
    from .class_register_util import SampleClass5

    assert SampleClass5 is obj


def test_old_class_error():
    with pytest.raises(RuntimeError):
        register_class(SampleClass6, old_paths=[class_to_str(SampleClass4)])
    register_class(SampleClass4, old_paths=["test.test.BBase"], version="0.0.2")


def test_get_version():
    assert str(REGISTER.get_version(SampleClass1)) == "0.0.0"
    assert str(REGISTER.get_version(SampleClass2)) == "0.0.1"
    assert str(REGISTER.get_version(SampleClass4)) == "0.0.2"
