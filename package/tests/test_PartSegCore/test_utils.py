# pylint: disable=no-self-use
import json
from unittest.mock import MagicMock

import pytest

from PartSegCore.json_hooks import PartSegEncoder, partseg_object_hook
from PartSegCore.utils import (
    BaseModel,
    CallbackFun,
    CallbackMethod,
    EventedDict,
    ProfileDict,
    get_callback,
    iterate_names,
    recursive_update_dict,
)


def test_callback_fun():
    call_list = []

    def call_fun():
        call_list.append(1)

    callback = CallbackFun(call_fun)
    assert not call_list
    callback()
    assert call_list == [1]
    assert callback.is_alive()


def test_callback_method():
    call_list = []

    class A:
        def fun(self):  # pylint: disable=no-self-use
            call_list.append(1)

    a = A()
    callback = CallbackMethod(a.fun)
    assert not call_list
    callback()
    assert call_list == [1]
    assert callback.is_alive()
    del a  # skipcq: PTC-W0043
    assert not callback.is_alive()
    callback()
    assert call_list == [1]


def test_get_callback():
    def call_fun():
        raise NotImplementedError

    class A:
        def fun(self):  # pylint: disable=no-self-use
            raise NotImplementedError

    a = A()

    assert isinstance(get_callback(call_fun), CallbackFun)
    assert isinstance(get_callback(a.fun), CallbackMethod)


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


class TestEventedDict:
    def test_simple_add(self):
        receiver = MagicMock()

        dkt = EventedDict()
        dkt.setted.connect(receiver.set)
        dkt.deleted.connect(receiver.delete)
        dkt["a"] = 1
        assert dkt["a"] == 1
        assert receiver.set.call_count == 1
        assert "'a': 1" in str(dkt)
        assert "'a': 1" in repr(dkt)
        dkt["a"] = 2
        assert dkt["a"] == 2
        assert receiver.set.call_count == 2
        assert len(dkt) == 1
        del dkt["a"]
        assert receiver.set.call_count == 2
        assert receiver.delete.call_count == 1
        assert len(dkt) == 0

    def test_simple_add_remove(self):
        callback_list = []

        def callback_add():
            callback_list.append(1)

        def callback_delete():
            callback_list.append(2)

        dkt = EventedDict()
        dkt.setted.connect(callback_add)
        dkt.deleted.connect(callback_delete)

        dkt[1] = 1
        dkt[2] = 1
        assert len(dkt) == 2
        assert callback_list == [1, 1]
        del dkt[1]
        assert len(dkt) == 1
        assert callback_list == [1, 1, 2]

    def test_nested_evented(self):
        dkt = EventedDict(bar={"foo": {"baz": 1}})
        assert isinstance(dkt["bar"], EventedDict)
        assert isinstance(dkt["bar"]["foo"], EventedDict)
        assert dkt["bar"]["foo"]["baz"] == 1

        dkt["baz"] = {"bar": {"foo": 1}}
        assert isinstance(dkt["baz"], EventedDict)
        assert isinstance(dkt["baz"]["bar"], EventedDict)
        assert dkt["baz"]["bar"]["foo"] == 1

    def test_serialize(self, tmp_path):
        dkt = EventedDict(a={"b": {"c": 1, "d": 2, "e": 3}, "f": 1}, g={"h": {"i": 1, "j": 2}, "k": [6, 7, 8]})
        with (tmp_path / "test_dict.json").open("w") as f_p:
            json.dump(dkt, f_p, cls=PartSegEncoder)

        with (tmp_path / "test_dict.json").open("r") as f_p:
            dkt2 = json.load(f_p, object_hook=partseg_object_hook)

        assert isinstance(dkt2, EventedDict)
        assert isinstance(dkt2["a"], EventedDict)
        assert dkt["g"]["k"] == [6, 7, 8]

    def test_signal_names(self):
        receiver = MagicMock()
        dkt = EventedDict(baz={"foo": 1})
        dkt.setted.connect(receiver.set)
        dkt.deleted.connect(receiver.deleted)
        dkt["foo"] = 1
        assert receiver.set.call_count == 1
        receiver.set.assert_called_with("foo")
        dkt["bar"] = EventedDict()
        assert receiver.set.call_count == 2
        receiver.set.assert_called_with("bar")
        dkt["bar"]["baz"] = 1
        assert receiver.set.call_count == 3
        receiver.set.assert_called_with("bar.baz")
        dkt["baz"]["foo"] = 2
        assert receiver.set.call_count == 4
        receiver.set.assert_called_with("baz.foo")

        del dkt["bar"]["baz"]
        assert receiver.deleted.call_count == 1
        receiver.deleted.assert_called_with("bar.baz")
        del dkt["bar"]
        assert receiver.deleted.call_count == 2
        receiver.deleted.assert_called_with("bar")

    def test_propagate_signal(self):
        receiver = MagicMock()
        dkt = EventedDict(baz={"foo": 1})
        dkt.setted.connect(receiver.set)
        dkt.deleted.connect(receiver.deleted)
        dkt["baz"].base_key = ""
        dkt["baz"]["foo"] = 2
        receiver.set.assert_called_with("foo")
        receiver.set.assert_called_once()
        del dkt["baz"]["foo"]
        receiver.deleted.assert_called_with("foo")
        receiver.deleted.assert_called_once()

    def test_dict_force_class(self):
        dkt = EventedDict(int)
        dkt["a"] = 1
        with pytest.raises(TypeError):
            dkt["b"] = "a"
        assert dkt["a"] == 1
        assert set(dkt) == {"a"}

    def test_dict_force_class_dkt(self):
        dkt = EventedDict({"a": int, "b": tuple})
        dkt["a"] = 1
        with pytest.raises(TypeError):
            dkt["b"] = "a"
        dkt["b"] = (1, 2)
        dkt["c"] = "A"
        assert set(dkt) == {"a", "b", "c"}

    def test_dict_force_class_dkt2(self):
        dkt = EventedDict({"a": int, "b": tuple, "*": str})
        dkt["a"] = 1
        with pytest.raises(TypeError):
            dkt["b"] = "a"
        dkt["b"] = (1, 2)
        with pytest.raises(TypeError):
            dkt["c"] = 1
        dkt["c"] = "A"
        assert set(dkt) == {"a", "b", "c"}

    def test_dict_nested_type(self):
        dkt = EventedDict({"a": {"b": int}})
        dkt["a"] = {"b": 1}
        with pytest.raises(TypeError):
            dkt["a"] = {"b": "a"}
        assert dkt["a"]["b"] == 1
        assert set(dkt) == {"a"}
        assert set(dkt["a"]) == {"b"}

    def test_dict_repr(self):
        assert repr(EventedDict(a=1, b=2)) == "EventedDict(klass={'*': <class 'object'>}, {'a': 1, 'b': 2})"
        assert str(EventedDict(a=1, b=2)) == "EventedDict[{'*': <class 'object'>}]({'a': 1, 'b': 2})"


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
        assert dkt.pop_errors() == []
        dkt.set("e.h.l", {"aaa": 1, "__error__": True})
        assert not dkt.verify_data()
        assert dkt.pop_errors()[0][0] == "e.h"

    def test_serialize(self, tmp_path):
        dkt = ProfileDict()
        dkt.set("a.b.c", 1)
        dkt.set("a.b.a", 2)
        with open(tmp_path / "test.json", "w") as f_p:
            json.dump(dkt, f_p, cls=PartSegEncoder)
        with open(tmp_path / "test.json") as f_p:
            dkt2 = json.load(f_p, object_hook=partseg_object_hook)

        assert dkt.my_dict == dkt2.my_dict

    def test_callback(self):
        def dummy_call():
            receiver.dummy()

        receiver = MagicMock()

        dkt = ProfileDict()
        dkt.connect("", receiver.empty)
        dkt.connect("", dummy_call)
        dkt.connect("b", receiver.b)
        dkt.connect(["d", "c"], receiver.dc)

        dkt.set("test.a", 1)
        assert receiver.empty.call_count == 1
        assert receiver.dummy.call_count == 1
        receiver.empty.assert_called_with("a")
        receiver.dummy.assert_called_with()
        dkt.set("test.a", 1)
        assert receiver.empty.call_count == 1
        receiver.b.assert_not_called()
        dkt.set("test2.a", 1)
        assert receiver.empty.call_count == 2
        receiver.b.assert_not_called()
        dkt.set(["test", "b"], 1)
        assert receiver.empty.call_count == 3
        assert receiver.b.call_count == 1
        dkt.set("test.d.c", 1)
        receiver.dc.assert_called_once()
        dkt.set("test.a", 2)
        assert receiver.empty.call_count == 5

    def test_dict_add(self):
        dkt = ProfileDict()
        data = dkt.get("foo", default={})
        assert isinstance(data, EventedDict)
        dkt.update({"foo": {"bar": 1}})
        data["a"] = 1
        assert dkt.get("foo") == {"a": 1, "bar": 1}

    def test_force_type(self):
        dkt = ProfileDict(int)
        dkt.set("a", 1)
        with pytest.raises(TypeError):
            dkt.set("b", "a")

    def test_force_type_nested(self):
        dkt = ProfileDict({"a": int, "b": {"a": int, "*": str}})
        dkt.set("a", 1)
        with pytest.raises(TypeError):
            dkt.set("b", "a")
        dkt.set("b.a", 1)
        with pytest.raises(TypeError):
            dkt.set("b.c", 1)
        with pytest.raises(TypeError):
            dkt.set("a", "a")
        assert dkt.get("a") == 1
        assert dkt.get("b") == {"a": 1}


def test_iterate_names():
    assert iterate_names("aaaa", {}) == "aaaa"
    assert iterate_names("aaaa", {"aaaa"}) == "aaaa (1)"
    assert iterate_names("aaaa", {"aaaa", "aaaa (1)", "aaaa (3)"}) == "aaaa (2)"

    assert iterate_names("a" * 10, {}, 10) == "a" * 10
    assert iterate_names("a" * 11, {}, 10) == "a" * 10
    assert iterate_names("a" * 9, {"a" * 9}, 10) == "a" * 5 + " (1)"

    input_set = {"aaaaa"}
    for _ in range(15):
        input_set.add(iterate_names("a" * 5, input_set))
    assert iterate_names("a" * 5, input_set) == "a" * 5 + " (16)"
    for _ in range(85):
        input_set.add(iterate_names("a" * 5, input_set))
    assert iterate_names("a" * 5, input_set) is None


def test_base_model_getitem():
    class SampleModel(BaseModel):
        a: int = 1
        b: float = 2.0
        c: str = "3"

    ob = SampleModel()
    with pytest.warns(FutureWarning, match=r"Access to attribute by \[\] is deprecated\. Use \. instead"):
        assert ob["a"] == 1
    with pytest.warns(FutureWarning, match=r"Access to attribute by \[\] is deprecated\. Use \. instead"):
        assert ob["b"] == 2.0
    with pytest.warns(FutureWarning, match=r"Access to attribute by \[\] is deprecated\. Use \. instead"):
        assert ob["c"] == "3"
    with pytest.raises(KeyError):
        ob["d"]  # pylint: disable=pointless-statement
