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
from PartSegCore.channel_class import Channel
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


def test_base_model_to_algorithm_property_base():
    class SampleEnum(Enum):
        a = 1
        b = 2

    class Sample(BaseModel):
        field1: int = Field(0, le=100, ge=0, title="Field 1")
        field2: SampleEnum = SampleEnum.a
        field3: float = Field(0, le=55, ge=-7, title="Field 3")
        channel: Channel = Field(0, title="Channel")

    s = Sample(field1=1, field3=1.5)
    assert s.field3 == 1.5

    converted = base_model_to_algorithm_property(Sample)
    assert len(converted) == 4
    assert converted[0].name == "field1"
    assert converted[0].user_name == "Field 1"
    assert issubclass(converted[0].value_type, int)
    assert converted[0].range == (0, 100)
    assert converted[1].name == "field2"
    assert converted[1].user_name == "field2"
    assert converted[1].value_type is SampleEnum
    assert converted[2].name == "field3"
    assert converted[2].user_name == "Field 3"
    assert issubclass(converted[2].value_type, float)
    assert converted[2].range == (-7, 55)

    assert converted[3].value_type is Channel
    assert converted[3].name == "channel"
    assert converted[3].user_name == "Channel"
