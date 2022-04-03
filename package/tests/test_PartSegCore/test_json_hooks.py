import json

from PartSegCore.image_operations import RadiusType
from PartSegCore.json_hooks import PartSegEncoder, partseg_object_hook
from PartSegCore.utils import ProfileDict


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
