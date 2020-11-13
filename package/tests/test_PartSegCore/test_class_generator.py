import typing
from collections import OrderedDict

import pytest

from PartSegCore.algorithm_describe_base import Register
from PartSegCore.class_generator import BaseSerializableClass, base_serialize_register

copy_register = Register()


def setup_module():
    """ setup any state specific to the execution of the given module."""
    from copy import deepcopy

    from PartSegCore import class_generator

    global copy_register  # pylint: disable=W0603
    copy_register = deepcopy(class_generator.base_serialize_register)


def teardown_module():
    """teardown any state that was previously setup with a setup_module
    method.
    """
    from PartSegCore import class_generator

    class_generator.base_serialize_register = copy_register


def empty(*_):
    pass


def test_readonly():
    class Test1(BaseSerializableClass):
        field1: str
        field2: int
        field3: float

    val = Test1("a", 1, 0.7)
    with pytest.raises(AttributeError):
        val.field1 = "a"
    with pytest.raises(AttributeError):
        val.field2 = 1
    with pytest.raises(AttributeError):
        val.field3 = 1.1
    assert val.field1 == "a"
    assert val.field2 == 1
    assert val.field3 == 0.7
    base_serialize_register.clear()


def test_default_values():
    class Test2(BaseSerializableClass):
        field1: str
        field2: int = 3
        field3: float = 11.1

    val1 = Test2("a")
    assert val1.field1 == "a"
    assert val1.field2 == 3
    assert val1.field3 == 11.1

    val2 = Test2("b", 7)
    assert val2.field1 == "b"
    assert val2.field2 == 7
    assert val2.field3 == 11.1

    val3 = Test2("c", field3=7.7)
    assert val3.field1 == "c"
    assert val3.field2 == 3
    assert val3.field3 == 7.7

    val4 = Test2(field3=5.5, field1="d")
    assert val4.field1 == "d"
    assert val4.field2 == 3
    assert val4.field3 == 5.5
    base_serialize_register.clear()


def test_functions_save():
    class Test3(BaseSerializableClass):
        field1: str
        field2: int

        def __str__(self):
            return f"Test3({self.field1}, {self.field2})"

        def test1(self):
            return self.field2 * self.field1

    val = Test3("aa", 4)
    assert str(val) == "Test3(aa, 4)"
    assert val.test1() == "aaaaaaaa"
    base_serialize_register.clear()


def test_name_collision():
    class Test4(BaseSerializableClass):
        field1: str

    empty(Test4)
    with pytest.raises(ValueError):

        class Test4(BaseSerializableClass):  # pylint: disable=E0102
            field1: str
            field2: str

    base_serialize_register.clear()
    empty(Test4)


def test_subclasses():
    class Test4(BaseSerializableClass):
        field1: str

    class Test5(Test4):
        field2: int

    assert issubclass(Test5, BaseSerializableClass)
    val0 = Test4("c")
    empty(val0)
    val = Test5("a", "b")
    assert val.field1 == "a"
    assert val.field2 == "b"
    assert isinstance(val, Test4)
    assert issubclass(Test5, Test4)
    base_serialize_register.clear()


def test_typing():
    class Test(BaseSerializableClass):
        field1: typing.Union[str, float]
        field2: typing.Any
        field3: typing.Optional[int]
        field4: typing.List[str]

    val = Test("a", [1, 2, 3], 5, ["a", "b"])
    assert val.field1 == "a"
    assert val.field2 == [1, 2, 3]
    assert val.field3 == 5
    assert val.field4 == ["a", "b"]

    class MyClass:
        def __init__(self, f=1, k=2):
            self.f = f
            self.k = k

    class Test2(BaseSerializableClass):
        field1: typing.Union[str, float]
        field2: typing.Any
        field3: typing.Optional[int]
        field4: typing.List[str]
        field5: typing.Optional[MyClass]

    Test2("aa", 1, None, ["b", "c"], MyClass())
    base_serialize_register.clear()


def test_forward_ref():
    class Test(BaseSerializableClass):  # pylint: disable=W0612
        val: int
        child: typing.Optional["Test"] = None  # noqa F821

    base_serialize_register.clear()


def test_generic_types():
    class Test1(BaseSerializableClass):
        list1: typing.Optional[int]
        list2: typing.Union[str, int]

    class Test2(BaseSerializableClass):
        list1: typing.List[int]
        list2: typing.List
        dict1: typing.Dict[str, int]
        dict2: typing.Dict

    empty(Test1, Test2)
    base_serialize_register.clear()


def test_post_init():
    class Test1(BaseSerializableClass):
        field1: str
        field2: int
        field3: str = None

        def __post_init__(self):
            self.field3 = self.field1 * self.field2

        def __str__(self):
            return f"{self.field1}, {self.field2}, {self.field3}"

    with pytest.raises(AttributeError):
        Test1("a", 3)

    class Test2(BaseSerializableClass):
        __readonly__ = False
        field1: str
        field2: int
        field3: str = None

        def __post_init__(self):
            self.field3 = self.field1 * self.field2

        def __str__(self):
            return f"{self.field1}, {self.field2}, {self.field3}"

    val = Test2("a", 3)
    assert val.field3 == "aaa"
    base_serialize_register.clear()


def test_functions():
    class Test1(BaseSerializableClass):
        field1: str
        field2: int
        field3: float

    class Test2(BaseSerializableClass):
        __readonly__ = False
        field1: str
        field2: int
        field3: float

    val1 = Test1("a", 1, 0.7)
    val2 = Test2("b", 2, 0.9)
    assert val1.as_tuple() == ("a", 1, 0.7)
    assert val2.as_tuple() == ("b", 2, 0.9)
    val2.field1 = "c"
    assert val2.as_tuple() == ("c", 2, 0.9)
    val3 = val1.replace_(field1="d")
    assert val1.as_tuple() == ("a", 1, 0.7)
    assert val3.as_tuple() == ("d", 1, 0.7)
    val4 = Test1("d", 1, 0.7)
    val5 = Test2("d", 1, 0.7)
    # __eq__ test
    assert val4 != val5
    assert val3 != val5
    assert val3 == val4

    assert OrderedDict([("field1", "d"), ("field2", 1), ("field3", 0.7)]) == val5.asdict()
    assert val4.asdict() == val5.asdict()

    val6 = Test1.make_(("a", 1, 0.7))
    assert isinstance(val6, Test1)
    assert val6 == val1

    val6 = Test1.make_({"field1": "a", "field2": 1, "field3": 0.7})
    assert isinstance(val6, Test1)
    assert val6 == val1
    base_serialize_register.clear()


def test_statistic_type():
    from PartSegCore.analysis.measurement_base import Leaf, Node

    empty(Leaf, Node)
    base_serialize_register.clear()
