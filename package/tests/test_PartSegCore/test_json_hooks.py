import json

from PartSegCore import Units
from PartSegCore.image_operations import RadiusType
from PartSegCore.io_utils import find_problematic_leafs
from PartSegCore.json_hooks import PartSegEncoder, partseg_object_hook
from PartSegCore.utils import ProfileDict


def test_profile_hook_colormap_load(bundle_test_dir):
    with open(bundle_test_dir / "view_settings_v0.12.6.json") as f_p:
        res = json.load(f_p, object_hook=partseg_object_hook)

    assert isinstance(res, ProfileDict)
    assert res.verify_data()


def test_dump_custom_types():
    prof_dict = ProfileDict()
    prof_dict.set("a.b.c", 1)
    data = {"a": RadiusType.R2D, "b": prof_dict}
    text = json.dumps(data, cls=PartSegEncoder)
    loaded = json.loads(text, object_hook=partseg_object_hook)
    assert loaded["a"] == RadiusType.R2D
    assert loaded["b"].get("a.b.c") == 1


def test_error_reported(bundle_test_dir):
    with open(bundle_test_dir / "problematic_profile_dict.json") as f_p:
        res = json.load(f_p, object_hook=partseg_object_hook)

    assert isinstance(res, ProfileDict)
    assert not res.verify_data()
    error_data = res.pop_errors()
    assert len(error_data) == 2
    assert len(find_problematic_leafs(error_data[0][1])) == 1
    assert len(find_problematic_leafs(error_data[1][1])) == 2


def test_plugin_bugfix():
    data = '{"__class__": "plugins.PartSegCore.mask_partition_utils.BorderRimParameters", "__class_version_dkt__": {"plugins.PartSegCore.mask_partition_utils.BorderRimParameters": "0.0.0", "PartSegCore.utils.BaseModel": "0.0.0"}, "__values__": {"distance": 500, "units": {"__class__": "PartSegCore.universal_const.Units", "__class_version_dkt__": {"plugins.PartSegCore.universal_const.Units": "0.0.0"}, "__values__": {"value": 2}}}}'  # noqa: E501
    res = json.loads(data, object_hook=partseg_object_hook)
    assert res.distance == 500
    assert res.units == Units.nm


def _test_prepare(bundle_test_dir):  # pragma: no cover
    """Prepare test data for test_error_reported. After call this the names of class should be manually broken"""
    from PartSegCore.segmentation.noise_filtering import BilateralNoiseFilteringParams, GaussNoiseFilteringParams

    dkt = ProfileDict()
    dkt.set("a.b", GaussNoiseFilteringParams())
    dkt.set("a.c", BilateralNoiseFilteringParams())
    dkt.set("b.a.d", GaussNoiseFilteringParams())
    dkt.set("b.a.e", BilateralNoiseFilteringParams())
    dkt.set("b.a.f", GaussNoiseFilteringParams())

    with open(bundle_test_dir / "problematic_profile_dict.json", "w") as f_p:
        json.dump(dkt, f_p, cls=PartSegEncoder)
