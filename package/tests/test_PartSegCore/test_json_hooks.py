import json

import numpy as np
import pytest
from napari.utils import Colormap

from PartSegCore.image_operations import RadiusType
from PartSegCore.json_hooks import ProfileDict, ProfileEncoder, profile_hook, recursive_update_dict


def test_recursive_update_dict_basic():
    dict1 = {"a": 1, "b": 2}
    dict2 = {"b": 3, "c": 4}
    dict1_copy = dict1.copy()
    dict1.update(dict2)
    recursive_update_dict(dict1_copy, dict2)
    assert dict1 == dict1_copy


def test_recursive_update_dict():
    dict1 = {"a": {"k": 1, "l": 2}, "b": {"k": 1, "l": 2}}
    dict2 = {"a": {"m": 3, "l": 4}, "b": 3, "c": 4}
    recursive_update_dict(dict1, dict2)
    assert dict1 == {"a": {"k": 1, "l": 4, "m": 3}, "b": 3, "c": 4}


class TestProfileDict:
    def test_simple(self):
        dkt = ProfileDict()
        dkt.set("a.b.c", 1)
        dkt.set("a.b.a", 2)
        assert dkt.get("a.b.c") == 1
        with pytest.raises(KeyError):
            dkt.get("a.b.d")
        dkt.get("a.b.d", 3)
        assert dkt.get("a.b.d") == 3
        assert dkt.get("a.b") == {"a": 2, "c": 1, "d": 3}
        with pytest.raises(TypeError):
            dkt.set("a.b.c.d", 3)

    def test_update(self):
        dkt = ProfileDict()
        dkt.update(a=1, b=2, c=3)
        assert dkt.my_dict == {"a": 1, "b": 2, "c": 3}
        dkt2 = ProfileDict()
        dkt2.update(c=4, d={"a": 2, "e": 7})
        assert dkt2.get("d.e") == 7
        dkt.update(dkt2)
        assert dkt.get("d.e") == 7
        assert dkt.get("c") == 4
        dkt.update({"g": 1, "h": 4})
        assert dkt.get("g") == 1
        dkt.update({"w": 1, "z": 4}, w=3)
        assert dkt.get("w") == 3
        assert dkt.verify_data()
        assert dkt.filter_data() == []
        dkt.set("e.h.l", {"aaa": 1, "__error__": True})
        assert not dkt.verify_data()
        assert dkt.filter_data() == ["e.h"]

    def test_serialize(self, tmp_path):
        dkt = ProfileDict()
        dkt.set("a.b.c", 1)
        dkt.set("a.b.a", 2)
        with open(tmp_path / "test.json", "w") as f_p:
            json.dump(dkt, f_p, cls=ProfileEncoder)
        with open(tmp_path / "test.json") as f_p:
            dkt2 = json.load(f_p, object_hook=profile_hook)

        assert dkt.my_dict == dkt2.my_dict


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

    cmap_list = [Colormap([(0, 0, 0), (1, 1, 1)]), Colormap([(0, 0, 0), (1, 1, 1)], controls=[0.1, 0.8])]
    with open(tmp_path / "test2.json", "w") as f_p:
        json.dump(cmap_list, f_p, cls=ProfileEncoder)

    with open(tmp_path / "test2.json") as f_p:
        cmap_list2 = json.load(f_p, object_hook=profile_hook)

    assert np.array_equal(cmap_list[0].colors, cmap_list2[0].colors)
    assert np.array_equal(cmap_list[0].controls, cmap_list2[0].controls)
    assert np.array_equal(cmap_list[1].colors, cmap_list2[1].colors[1:3])
    assert np.array_equal(cmap_list[1].controls, cmap_list2[1].controls[1:3])
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
