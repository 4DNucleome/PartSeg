import pytest
import typing
from enum import Enum
from PartSeg.utils.class_generator import BaseSerializableClass, base_serialize_register


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

    with pytest.raises(ValueError):
        class Test4(BaseSerializableClass):
            field1: str
            field2: str
    base_serialize_register.clear()


def test_subclasses():

    class Test4(BaseSerializableClass):
        field1: str

    class Test5(Test4):
        field2: int

    assert issubclass(Test5, BaseSerializableClass)
    val0 = Test4("c")
    val = Test5("a", "b")
    assert val.field1 == "a"
    assert val.field2 == "b"
    assert isinstance(val, Test4)
    assert issubclass(Test5, Test4)
    base_serialize_register.clear()

def test_typing():
    a = typing.Optional[int]
    b = typing.List[str]
    class Test(BaseSerializableClass):
        field1: typing.Union[str, float]
        field2: typing.Any
        field3: typing.Optional[int]
        field4: typing.List[str]

    val =  Test("a", [1, 2, 3], 5, ["a", "b"])
    assert val.field1 == "a"
    assert val.field2 == [1, 2, 3]
    assert val.field3 == 5
    assert val.field4 ==  ["a", "b"]

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


def test_statistic_type():
    from PartSeg.utils.analysis.statistics_calculation import Leaf, Node
