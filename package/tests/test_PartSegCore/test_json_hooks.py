# pylint: disable=R0201

import json
from dataclasses import dataclass

from PartSegCore.image_operations import RadiusType
from PartSegCore.json_hooks import PartSegEncoder, partseg_object_hook
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
        json.load(f_p, object_hook=partseg_object_hook)


def test_dump_custom_types():
    prof_dict = ProfileDict()
    prof_dict.set("a.b.c", 1)
    data = {"a": RadiusType.R2D, "b": prof_dict}
    text = json.dumps(data, cls=PartSegEncoder)
    loaded = json.loads(text, object_hook=partseg_object_hook)
    assert loaded["a"] == RadiusType.R2D
    assert loaded["b"].get("a.b.c") == 1
