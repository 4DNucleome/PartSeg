import typing
from enum import Enum

import pytest
from pydantic import BaseModel, Field, ValidationError

from PartSegCore.algorithm_describe_base import (
    AlgorithmDescribeBase,
    AlgorithmProperty,
    AlgorithmSelection,
    _GetDescriptionClass,
    base_model_to_algorithm_property,
)
from PartSegCore.class_register import class_to_str


def test_get_description_class():
    class SampleClass:
        __test_class__ = _GetDescriptionClass()

        @classmethod
        def get_fields(self):
            return [AlgorithmProperty("test1", "Test 1", 1), AlgorithmProperty("test2", "Test 2", 2.0)]

    val = SampleClass.__test_class__
    assert val.__name__ == "__test_class__"
    assert val.__qualname__.endswith("SampleClass.__test_class__")
    assert issubclass(val, BaseModel)
    assert val.__fields__.keys() == {"test1", "test2"}


def test_algorithm_selection():
    class TestSelection(AlgorithmSelection):
        pass

    class TestSelection2(AlgorithmSelection):
        pass

    class Class1(AlgorithmDescribeBase):
        @classmethod
        def get_name(cls) -> str:
            return "test1"

        @classmethod
        def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
            return []

    class Class2(AlgorithmDescribeBase):
        @classmethod
        def get_name(cls) -> str:
            return "test2"

        @classmethod
        def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
            return []

    TestSelection.register(Class1)
    TestSelection.register(Class2)

    assert "test1" in TestSelection.__register__
    assert "test1" not in TestSelection2.__register__

    v = TestSelection(name="test1", values={})
    assert v.name == "test1"
    assert v.class_path == class_to_str(Class1)

    with pytest.raises(ValidationError):
        TestSelection(name="test3", values={})

    assert TestSelection["test1"] is Class1


def test_base_model_to_algorithm_property():
    class SampleEnum(Enum):
        a = 1
        b = 2

    class Sample(BaseModel):
        field1: int = Field(0, le=100, gt=0, title="Field 1")
        field2: SampleEnum = SampleEnum.a

    converted = base_model_to_algorithm_property(Sample)
    assert len(converted) == 2
