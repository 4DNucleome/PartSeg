# pylint: disable=R0201
import typing
from enum import Enum

import pytest
from nme import class_to_str, register_class
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, ValidationError

from PartSegCore.algorithm_describe_base import (
    AlgorithmDescribeBase,
    AlgorithmProperty,
    AlgorithmSelection,
    ROIExtractionProfile,
    _GetDescriptionClass,
    base_model_to_algorithm_property,
)
from PartSegCore.utils import BaseModel
from PartSegImage import Channel


def test_get_description_class():
    class SampleClass:
        __test_class__ = _GetDescriptionClass()

        @classmethod
        def get_fields(cls):
            return [AlgorithmProperty("test1", "Test 1", 1), AlgorithmProperty("test2", "Test 2", 2.0)]

    val = SampleClass.__test_class__
    assert val.__name__ == "__test_class__"
    assert val.__qualname__.endswith("SampleClass.__test_class__")
    assert issubclass(val, PydanticBaseModel)
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
    assert v.values == {}

    with pytest.raises(ValidationError):
        TestSelection(name="test3", values={})

    assert TestSelection["test1"] is Class1


def test_algorithm_selection_convert_subclass(clean_register):
    class TestSelection(AlgorithmSelection):
        pass

    @register_class
    class TestModel1(BaseModel):
        field1: int = 0

    @register_class(version="0.0.1", migrations=[("0.0.1", lambda x: {"field2": x["field"]})])
    class TestModel2(BaseModel):
        field2: int = 7

    class Class1(AlgorithmDescribeBase):
        __argument_class__ = TestModel1

        @classmethod
        def get_name(cls) -> str:
            return "test1"

    class Class2(AlgorithmDescribeBase):
        __argument_class__ = TestModel2

        @classmethod
        def get_name(cls) -> str:
            return "test2"

    TestSelection.register(Class1)
    TestSelection.register(Class2)

    ob = TestSelection(name="test1", values={"field1": 4})
    assert isinstance(ob.values, TestModel1)
    assert ob.values.field1 == 4

    ob = TestSelection(name="test2", values={"field": 5})
    assert isinstance(ob.values, TestModel2)
    assert ob.values.field2 == 5


def test_algorithm_selection_register_old(clean_register):
    class TestSelection(AlgorithmSelection):
        pass

    @register_class
    class TestModel1(BaseModel):
        field1: int = 0

    class Class1(AlgorithmDescribeBase):
        __argument_class__ = TestModel1

        @classmethod
        def get_name(cls) -> str:
            return "test3"

    TestSelection.register(Class1, old_names=["test"])

    ob = TestSelection(name="test3", values={"field1": 4})
    assert isinstance(ob.values, TestModel1)
    ob = TestSelection(name="test", values={"field1": 4})
    assert isinstance(ob.values, TestModel1)
    with pytest.raises(ValidationError):
        TestSelection(name="test4", values={"field1": 4})


def test_base_model_to_algorithm_property_base():
    class SampleEnum(Enum):
        a = 1
        b = 2

    class Sample(BaseModel):
        field1: int = Field(0, le=100, ge=0, title="Field 1")
        field2: SampleEnum = SampleEnum.a
        field_3: float = Field(0, le=55, ge=-7)
        channel: Channel = Field(0, title="Channel")

    s = Sample(field1=1, field_3=1.5)
    assert s.field_3 == 1.5

    converted = base_model_to_algorithm_property(Sample)
    assert len(converted) == 4
    assert converted[0].name == "field1"
    assert converted[0].user_name == "Field 1"
    assert issubclass(converted[0].value_type, int)
    assert converted[0].range == (0, 100)
    assert converted[1].name == "field2"
    assert converted[1].user_name == "Field2"
    assert converted[1].value_type is SampleEnum
    assert converted[2].name == "field_3"
    assert converted[2].user_name == "Field 3"
    assert issubclass(converted[2].value_type, float)
    assert converted[2].range == (-7, 55)

    assert converted[3].value_type is Channel
    assert converted[3].name == "channel"
    assert converted[3].user_name == "Channel"


def test_base_model_to_algorithm_property_algorithm_describe_base():
    class SampleSelection(AlgorithmSelection):
        pass

    class SampleClass1(AlgorithmDescribeBase):
        @classmethod
        def get_name(cls) -> str:
            return "1"

        @classmethod
        def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
            return []

    class SampleClass2(AlgorithmDescribeBase):
        @classmethod
        def get_name(cls) -> str:
            return "2"

        @classmethod
        def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
            return []

    SampleSelection.register(SampleClass1)
    SampleSelection.register(SampleClass2)

    d_text = "description text"

    class SampleModel(BaseModel):
        field1: int = Field(10, le=100, ge=0, title="Field 1", description=d_text)
        check_selection: SampleSelection = Field(SampleSelection(name="1", values={}), title="Class selection")

    converted = base_model_to_algorithm_property(SampleModel)
    assert len(converted) == 2
    assert issubclass(converted[0].value_type, int)
    assert converted[0].help_text == d_text
    assert issubclass(converted[1].value_type, AlgorithmDescribeBase)
    assert converted[1].default_value == "1"
    assert converted[1].possible_values is SampleSelection.__register__


def test_base_model_to_algorithm_property_algorithm_describe_empty():
    assert base_model_to_algorithm_property(BaseModel) == []


def test_text_addition_model_to_algorithm_property():
    class ModelWithText(BaseModel):
        field1: int = 1
        field2: int = Field(1, prefix="aaaa")
        field3: int = Field(1, suffix="bbbb")
        field4: int = 1

        @staticmethod
        def header():
            return "cccc"

    property_list = base_model_to_algorithm_property(ModelWithText)
    assert property_list[0] == "cccc"
    assert property_list[2] == "aaaa"
    assert property_list[5] == "bbbb"


def test_base_model_to_algorithm_property_position():
    class BBaseModel(BaseModel):
        field1: int = 1
        field2: int = 1

    class ModelWithPosition(BBaseModel):
        field3: int = Field(1, position=1)

    property_list = base_model_to_algorithm_property(ModelWithPosition)
    assert property_list[0].name == "field1"
    assert property_list[1].name == "field3"
    assert property_list[2].name == "field2"


def test_base_model_to_algorithm_property_magicgui_parameters():
    class BBaseModel(BaseModel):
        field1: int = Field(1, options={"a": 1, "b": 2})

    prop = base_model_to_algorithm_property(BBaseModel)[0]
    assert prop.mgi_options == {"a": 1, "b": 2}


class TestAlgorithmDescribeBase:
    def test_old_style_algorithm(self):
        class SampleAlgorithm(AlgorithmDescribeBase):
            @classmethod
            def get_name(cls) -> str:
                return "sample"

            @classmethod
            def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
                return ["aaaa", AlgorithmProperty("name", "Name", 1, options_range=(1, 10), help_text="ceeeec")]

        assert SampleAlgorithm.get_name() == "sample"
        assert len(SampleAlgorithm.get_fields()) == 2
        assert "ceeeec" in SampleAlgorithm.get_doc_from_fields()
        assert "(default values: 1)" in SampleAlgorithm.get_doc_from_fields()
        assert len(SampleAlgorithm.get_fields_dict()) == 1
        assert SampleAlgorithm.get_default_values() == {"name": 1}

    def test_new_style_algorithm(self):
        class DataModel(BaseModel):
            name: int = Field(1, ge=1, le=10, description="ceeeec", prefix="aaaa")

        class SampleAlgorithm(AlgorithmDescribeBase):
            __argument_class__ = DataModel

            @classmethod
            def get_name(cls) -> str:
                return "sample"

        assert SampleAlgorithm.get_name() == "sample"
        with pytest.warns(FutureWarning, match=r"Class has __argument_class__ defined"):
            assert len(SampleAlgorithm.get_fields()) == 2
        assert "ceeeec" in SampleAlgorithm.get_doc_from_fields()
        assert "(default values: 1)" in SampleAlgorithm.get_doc_from_fields()
        assert len(SampleAlgorithm.get_fields_dict()) == 1
        assert SampleAlgorithm.get_default_values() == {"name": 1}

    def test_new_style_algorithm_with_old_style_subclass(self):
        class DataModel(BaseModel):
            name: int = Field(1, ge=1, le=10, description="ceeeec", prefix="aaaa")

        class SampleAlgorithm(AlgorithmDescribeBase):
            __argument_class__ = DataModel

            @classmethod
            def get_name(cls) -> str:
                return "sample"

        class SampleSubAlgorithm(SampleAlgorithm):
            @classmethod
            def get_name(cls) -> str:
                return "sample2"

            @classmethod
            def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
                return super().get_fields() + [
                    AlgorithmProperty("name2", "Name 2", 3.0, options_range=(1, 10), help_text="deeeed")
                ]

        assert SampleSubAlgorithm.get_name() == "sample2"
        assert SampleAlgorithm.get_name() == "sample"
        with pytest.warns(FutureWarning, match=r"Class has __argument_class__ defined"):
            assert len(SampleSubAlgorithm.get_fields()) == 3
        with pytest.warns(FutureWarning, match=r"Class has __argument_class__ defined"):
            doc_text = SampleSubAlgorithm.get_doc_from_fields()
        assert "ceeeec" in doc_text
        assert "deeeed" in doc_text
        assert "(default values: 1)" in doc_text
        assert "(default values: 3.0)" in doc_text
        with pytest.warns(FutureWarning, match=r"Class has __argument_class__ defined"):
            assert len(SampleSubAlgorithm.get_fields_dict()) == 2
        with pytest.warns(FutureWarning, match=r"Class has __argument_class__ defined"):
            assert SampleSubAlgorithm.get_default_values() == {"name": 1, "name2": 3.0}


def test_roi_extraction_profile():
    ROIExtractionProfile(name="aaa", algorithm="aaa", values={})
    with pytest.warns(FutureWarning):
        ROIExtractionProfile("aaa", "aaa", {})
