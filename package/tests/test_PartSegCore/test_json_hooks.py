# pylint: disable=R0201

import json
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pytest
from napari.utils import Colormap
from napari.utils.notifications import NotificationSeverity

from PartSegCore._old_json_hooks import ProfileEncoder, profile_hook
from PartSegCore.class_register import class_to_str, register_class, rename_key
from PartSegCore.image_operations import RadiusType
from PartSegCore.json_hooks import PartSegEncoder, add_class_info, partseg_object_hook
from PartSegCore.utils import BaseModel, ProfileDict


@dataclass
class SampleDataclass:
    filed1: int
    field2: str


class SamplePydantic(BaseModel):
    sample_int: int
    sample_str: str
    sample_dataclass: SampleDataclass


class SampleAsDict:
    def __init__(self, value1, value2):
        self.value1 = value1
        self.value2 = value2

    def as_dict(self):
        return {"value1": self.value1, "value2": self.value2}


def test_profile_hook_colormap_load(bundle_test_dir):
    with open(bundle_test_dir / "view_settings_v0.12.6.json") as f_p:
        json.load(f_p, object_hook=profile_hook)


def test_colormap_dump(tmp_path):
    cmap_list = [Colormap([(0, 0, 0), (1, 1, 1)]), Colormap([(0, 0, 0), (1, 1, 1)], controls=[0, 1])]
    with open(tmp_path / "test.json", "w") as f_p:
        json.dump(cmap_list, f_p, cls=ProfileEncoder)

    with open(tmp_path / "test.json") as f_p:
        cmap_list2 = json.load(f_p, object_hook=profile_hook)

    assert np.array_equal(cmap_list[0].colors, cmap_list2[0].colors)
    assert np.array_equal(cmap_list[0].controls, cmap_list2[0].controls)
    assert np.array_equal(cmap_list[1].colors, cmap_list2[1].colors)
    assert np.array_equal(cmap_list[1].controls, cmap_list2[1].controls)

    cmap_list = [
        Colormap([(0, 0, 0), (1, 1, 1)]),
        Colormap([(0, 0, 0), (0, 0, 0), (1, 1, 1), (1, 1, 1)], controls=[0, 0.1, 0.8, 1]),
    ]
    with open(tmp_path / "test2.json", "w") as f_p:
        json.dump(cmap_list, f_p, cls=ProfileEncoder)

    with open(tmp_path / "test2.json") as f_p:
        cmap_list2 = json.load(f_p, object_hook=profile_hook)

    assert np.array_equal(cmap_list[0].colors, cmap_list2[0].colors)
    assert np.array_equal(cmap_list[0].controls, cmap_list2[0].controls)
    assert np.array_equal(cmap_list[1].colors, cmap_list2[1].colors)
    assert np.array_equal(cmap_list[1].controls, cmap_list2[1].controls)
    assert cmap_list2[1].controls[0] == 0
    assert cmap_list2[1].controls[-1] == 1
    assert np.array_equal(cmap_list[1].colors[0], cmap_list2[1].colors[0])
    assert np.array_equal(cmap_list[1].colors[-1], cmap_list2[1].colors[-1])


class TestProfileEncoder:
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.float32, np.float64])
    def test_dump_numpy_types(self, dtype):
        data = {"a": dtype(2)}
        text = json.dumps(data, cls=ProfileEncoder)
        loaded = json.loads(text)
        assert loaded["a"] == 2

    def test_dump_custom_types(self):
        prof_dict = ProfileDict()
        prof_dict.set("a.b.c", 1)
        data = {"a": RadiusType.R2D, "b": prof_dict}
        text = json.dumps(data, cls=ProfileEncoder)
        loaded = json.loads(text, object_hook=profile_hook)
        assert loaded["a"] == RadiusType.R2D
        assert loaded["b"].get("a.b.c") == 1


class TestPartSegEncoder:
    def test_enum_serialize(self, tmp_path):
        data = {"value1": RadiusType.R2D, "value2": RadiusType.NO, "value3": NotificationSeverity.ERROR}
        with (tmp_path / "test.json").open("w") as f_p:
            json.dump(data, f_p, cls=PartSegEncoder)
        with (tmp_path / "test.json").open("r") as f_p:
            data2 = json.load(f_p, object_hook=partseg_object_hook)
        assert data2["value1"] == RadiusType.R2D
        assert data2["value2"] == RadiusType.NO
        assert data2["value3"] == NotificationSeverity.ERROR

    def test_dataclass_serialze(self, tmp_path):
        data = {"value": SampleDataclass(1, "text")}
        with (tmp_path / "test.json").open("w") as f_p:
            json.dump(data, f_p, cls=PartSegEncoder)
        with (tmp_path / "test.json").open("r") as f_p:
            data2 = json.load(f_p, object_hook=partseg_object_hook)

        assert isinstance(data2["value"], SampleDataclass)
        assert data2["value"] == SampleDataclass(1, "text")

    def test_pydantic_serialize(self, tmp_path):
        data = {
            "color1": Colormap(colors=[[0, 0, 0], [0, 0, 0]], controls=[0, 1]),
            "other": SamplePydantic(sample_int=1, sample_str="text", sample_dataclass=SampleDataclass(1, "text")),
        }
        with (tmp_path / "test.json").open("w") as f_p:
            json.dump(data, f_p, cls=PartSegEncoder)
        with (tmp_path / "test.json").open("r") as f_p:
            data2 = json.load(f_p, object_hook=partseg_object_hook)
        assert data2["color1"] == Colormap(colors=[[0, 0, 0], [0, 0, 0]], controls=[0, 1])
        assert isinstance(data2["other"], SamplePydantic)
        assert isinstance(data2["other"].sample_dataclass, SampleDataclass)

    def test_numpy_serialize(self, tmp_path):
        data = {"arr": np.arange(10), "f": np.float32(0.1), "i": np.int16(1000)}
        with (tmp_path / "test.json").open("w") as f_p:
            json.dump(data, f_p, cls=PartSegEncoder)
        with (tmp_path / "test.json").open("r") as f_p:
            data2 = json.load(f_p, object_hook=partseg_object_hook)
        assert data2["arr"] == list(range(10))
        assert np.isclose(data["f"], 0.1)
        assert data2["i"] == 1000

    def test_class_with_as_dict(self, tmp_path):
        data = {"d": SampleAsDict(1, 10)}
        with (tmp_path / "test.json").open("w") as f_p:
            json.dump(data, f_p, cls=PartSegEncoder)
        with (tmp_path / "test.json").open("r") as f_p:
            data2 = json.load(f_p, object_hook=partseg_object_hook)
        assert isinstance(data2["d"], SampleAsDict)
        assert data2["d"].value1 == data["d"].value1
        assert data2["d"].value2 == data["d"].value2

    def test_sub_class_serialization(self, tmp_path):
        ob = DummyClassForTest.DummySubClassForTest(1, 2)
        with (tmp_path / "test.json").open("w") as f_p:
            json.dump(ob, f_p, cls=PartSegEncoder)
        with (tmp_path / "test.json").open("r") as f_p:
            ob2 = json.load(f_p, object_hook=partseg_object_hook)
        assert ob2.data1 == 1
        assert ob2.data2 == 2


def test_add_class_info_pydantic(clean_register):
    @register_class
    class SampleClass(BaseModel):
        field: int = 1

    dkt = {}
    add_class_info(SampleClass(), dkt)
    assert "__class__" in dkt
    assert class_to_str(SampleClass) == dkt["__class__"]
    assert len(dkt["__class_version_dkt__"]) == 1
    assert dkt["__class_version_dkt__"][class_to_str(SampleClass)] == "0.0.0"


def test_add_class_info_enum(clean_register):
    @register_class
    class SampleEnum(Enum):
        field = 1

    dkt = {}
    add_class_info(SampleEnum.field, dkt)
    assert "__class__" in dkt
    assert class_to_str(SampleEnum) == dkt["__class__"]
    assert len(dkt["__class_version_dkt__"]) == 1
    assert dkt["__class_version_dkt__"][class_to_str(SampleEnum)] == "0.0.0"


class TestPartSegObjectHook:
    def test_no_inheritance_read(self, clean_register, tmp_path):
        @register_class(version="0.0.1", migrations=[("0.0.1", rename_key("field", "field1"))])
        class BaseClass(BaseModel):
            field1: int = 1

        @register_class(
            version="0.0.1", migrations=[("0.0.1", rename_key("field", "field1"))], use_parent_migrations=False
        )
        class MainClass(BaseClass):
            field2: int = 5

        data_str = """
        {"field": 1, "field2": 5,
         "__class__":
         "test_PartSegCore.test_json_hooks.TestPartSegObjectHook.test_no_inheritance_read.<locals>.MainClass",
         "__class_version_dkt__": {
         "test_PartSegCore.test_json_hooks.TestPartSegObjectHook.test_no_inheritance_read.<locals>.MainClass": "0.0.0",
         "test_PartSegCore.test_json_hooks.TestPartSegObjectHook.test_no_inheritance_read.<locals>.BaseClass": "0.0.0"
         }}
         """

        ob = json.loads(data_str, object_hook=partseg_object_hook)
        assert isinstance(ob, MainClass)


class DummyClassForTest:
    class DummySubClassForTest:
        def __init__(self, data1, data2):
            self.data1, self.data2 = data1, data2

        def as_dict(self):
            return {"data1": self.data1, "data2": self.data2}
