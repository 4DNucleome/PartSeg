import json

import pytest

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
