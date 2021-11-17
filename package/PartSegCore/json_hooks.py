import copy
import importlib
import itertools
import typing
from collections import defaultdict
from collections.abc import MutableMapping
from contextlib import suppress

import numpy as np
from napari.utils import Colormap
from psygnal import Signal

from PartSegCore.algorithm_describe_base import ROIExtractionProfile

from .class_generator import SerializeClassEncoder, serialize_hook
from .image_operations import RadiusType
from .utils import CallbackBase, get_callback


class EventedDict(MutableMapping):
    setted = Signal(str)
    deleted = Signal(str)

    def __init__(self, **kwargs):
        # TODO add positional only argument when drop python 3.7
        super().__init__()
        self._dict = {}
        for key, dkt in kwargs.items():
            if isinstance(dkt, dict):
                dkt = EventedDict(**dkt)
            if isinstance(dkt, EventedDict):
                dkt.base_key = key
                dkt.setted.connect(self._propagate_setitem)
                dkt.deleted.connect(self._propagate_del)
            self._dict[key] = dkt
        self.base_key = ""

    def __setitem__(self, k, v) -> None:
        if isinstance(v, dict):
            v = EventedDict(**v)
        if isinstance(v, EventedDict):
            v.base_key = k
            v.setted.connect(self._propagate_setitem)
            v.deleted.connect(self._propagate_del)
        if k in self._dict and isinstance(self._dict[k], EventedDict):
            self._dict[k].setted.disconnect(self._propagate_setitem)
            self._dict[k].deleted.disconnect(self._propagate_del)
        old_value = self._dict[k] if k in self._dict else None
        with suppress(ValueError):
            if old_value == v:
                return
        self._dict[k] = v
        self.setted.emit(k)

    def __delitem__(self, k) -> None:
        if k in self._dict and isinstance(self._dict[k], EventedDict):
            self._dict[k].setted.disconnect(self._propagate_setitem)
            self._dict[k].deleted.disconnect(self._propagate_del)
        del self._dict[k]
        self.deleted.emit(k)

    def __getitem__(self, k):
        return self._dict[k]

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> typing.Iterator:
        return iter(self._dict)

    def as_dict(self):
        return copy.copy(self._dict)

    def as_dict_deep(self):
        return {k: v.as_dict_deep() if isinstance(v, EventedDict) else v for k, v in self._dict.items()}

    def __str__(self):
        return f"EventedDict({self._dict})"

    def __repr__(self):
        return f"EventedDict({repr(self._dict)})"

    def _propagate_setitem(self, key):
        # Fixme when partial disconnect will work
        sender: EventedDict = Signal.sender()
        if sender.base_key:
            self.setted.emit(f"{sender.base_key}.{key}")
        else:
            self.setted.emit(key)

    def _propagate_del(self, key):
        # Fixme when partial disconnect will work
        sender: EventedDict = Signal.sender()
        if sender.base_key:
            self.deleted.emit(f"{sender.base_key}.{key}")
        else:
            self.deleted.emit(key)


def recursive_update_dict(main_dict: typing.Union[dict, EventedDict], other_dict: typing.Union[dict, EventedDict]):
    """
    recursive update main_dict with elements of other_dict

    :param main_dict: dict to be updated recursively
    :param other_dict: source dict

    >>> dkt1 = {"test": {"test": 1, "test2": {"test4": 1}}}
    >>> dkt2 = {"test": {"test": 4, "test2": {"test2": 1}}}
    >>> recursive_update_dict(dkt1, dkt2)
    >>> dkt1
        {"test": {"test": 1, "test2": {"test4": 1, "test2": 1}}}
    """
    for key, val in other_dict.items():
        if key in main_dict and isinstance(main_dict[key], dict) and isinstance(val, dict):
            recursive_update_dict(main_dict[key], val)
        else:
            main_dict[key] = val


class ProfileDict:
    """
    Dict for storing recursive data. The path are dot separated.

    >>> dkt = ProfileDict()
    >>> dkt.set(["aa", "bb", "c1"], 7)
    >>> dkt.get(["aa", "bb", "c1"])
        7
    >>> dkt.get("aa.bb.c2", 8)
        8
    >>> dkt.get("aa.bb")
        {'c1': 7, 'c2': 8}
    """

    def __init__(self, **kwargs):
        self._my_dict = EventedDict(**kwargs)
        self._callback_dict: typing.Dict[str, typing.List[CallbackBase]] = defaultdict(list)

        self._my_dict.setted.connect(self._call_callback)
        self._my_dict.deleted.connect(self._call_callback)

    def as_dict(self):
        return self.my_dict.as_dict()

    @property
    def my_dict(self) -> EventedDict:
        return self._my_dict

    @my_dict.setter
    def my_dict(self, value: typing.Union[dict, EventedDict]):
        if isinstance(value, dict):
            value = EventedDict(**value)

        self._my_dict.setted.disconnect(self._call_callback)
        self._my_dict.deleted.disconnect(self._call_callback)
        self._my_dict = value
        self._my_dict.setted.connect(self._call_callback)
        self._my_dict.deleted.connect(self._call_callback)

    def update(self, ob: typing.Union["ProfileDict", dict, None] = None, **kwargs):
        """
        Update dict recursively. Use :py:func:`~.recursive_update_dict`

        :param ob: data source
        :param kwargs: data source, as keywords
        """
        if isinstance(ob, ProfileDict):
            recursive_update_dict(self.my_dict, ob.my_dict)
            recursive_update_dict(self.my_dict, kwargs)
        elif isinstance(ob, dict):
            recursive_update_dict(self.my_dict, ob)
            recursive_update_dict(self.my_dict, kwargs)
        elif ob is None:
            recursive_update_dict(self.my_dict, kwargs)

    def profile_change(self):
        for callback in itertools.chain(*self._callback_dict.values()):
            callback()

    def connect(
        self, key_path: typing.Union[typing.Sequence[str], str], callback: typing.Callable[[], typing.Any], maxargs=None
    ) -> typing.Callable:
        """
        Connect function to receive information when object on path was changed using :py:meth:`.set`

        :param key_path: path for which signal should be emitted
        :param callback: parameterless function which should be called

        :return: callback function itself.
        """
        if not isinstance(key_path, str):
            key_path = ".".join(key_path)

        self._callback_dict[key_path].append(get_callback(callback, maxargs))
        return callback

    def set(self, key_path: typing.Union[typing.Sequence[str], str], value):
        """
        Set value from dict

        :param key_path: Path to element. If is string then will be split on '.' (`key_path.split('.')`)
        :param value: Value to set.
        """
        if isinstance(key_path, str):
            key_path = key_path.split(".")
        curr_dict = self.my_dict

        for i, key in enumerate(key_path[:-1]):
            try:
                # TODO add check if next step element is dict and create custom information
                curr_dict = curr_dict[key]
            except KeyError:
                for key2 in key_path[i:-1]:
                    with curr_dict.setted.blocked():
                        curr_dict[key2] = EventedDict()
                    curr_dict = curr_dict[key2]
                break
        if isinstance(value, dict):
            value = EventedDict(**value)
        curr_dict[key_path[-1]] = value
        return value

    def _call_callback(self, key_path: typing.Union[typing.Sequence[str], str]):
        if isinstance(key_path, str):
            key_path = key_path.split(".")
        full_path = ".".join(key_path[1:])
        callback_path = ""
        callback_list = []
        if callback_path in self._callback_dict:
            callback_list = self._callback_dict[callback_path]

        for callback_path in itertools.accumulate(key_path[1:], lambda x, y: f"{x}.{y}"):
            if callback_path in self._callback_dict:
                li = self._callback_dict[callback_path]
                li = [x for x in li if x.is_alive()]
                self._callback_dict[callback_path] = li
                callback_list.extend(li)
        for callback in callback_list:
            callback(full_path)

    def get(self, key_path: typing.Union[list, str], default=None):
        """
        Get value from dict.

        :param key_path: Path to element. If is string then will be split on . (`key_path.split('.')`).
        :param default: default value if element missed in dict.
        :raise KeyError: on missed element if default is not provided.
        :return: requested value
        """
        if isinstance(key_path, str):
            key_path = key_path.split(".")
        curr_dict = self.my_dict
        for key in key_path:
            try:
                curr_dict = curr_dict[key]
            except KeyError as e:
                if default is None:
                    raise e

                val = copy.deepcopy(default)
                return self.set(key_path, val)

        return curr_dict

    def verify_data(self) -> bool:
        """
        Call :py:func:`~.check_loaded_dict` on inner structures
        """
        return check_loaded_dict(self.my_dict)

    def filter_data(self):
        error_list = []
        for group, up_dkt in list(self.my_dict.items()):
            if not isinstance(up_dkt, (dict, EventedDict)):
                continue
            for key, dkt in list(up_dkt.items()):
                if not check_loaded_dict(dkt):
                    error_list.append(f"{group}.{key}")
                    del up_dkt[key]
        return error_list


class ProfileEncoder(SerializeClassEncoder):
    """
    Json encoder for :py:class:`ProfileDict`, :py:class:`RadiusType`,
     :py:class:`.SegmentationProfile` classes

    >>> import json
    >>> data = ProfileDict()
    >>> data.set("aa.bb.cc", 7)
    >>> with open("some_file", 'w') as fp:
    >>>     json.dump(data, fp, cls=ProfileEncoder)
    """

    # pylint: disable=E0202
    def default(self, o):
        """encoder implementation"""
        if isinstance(o, RadiusType):
            return {"__RadiusType__": True, "value": o.value}
        if isinstance(o, ROIExtractionProfile):
            return {"__SegmentationProfile__": True, "name": o.name, "algorithm": o.algorithm, "values": o.values}
        if isinstance(o, Colormap):
            return {
                "__Colormap__": True,
                "name": o.name,
                "colors": o.colors.tolist(),
                "interpolation": o.interpolation,
                "controls": o.controls.tolist(),
            }
        if hasattr(o, "as_dict"):
            dkt = o.as_dict()
            dkt["__class__"] = o.__module__ + "." + o.__class__.__name__
            return dkt
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super().default(o)


def profile_hook(dkt):
    """
    hook for json loading

    >>> import json
    >>> with open("some_file", 'r') as fp:
    ...     data = json.load(fp, object_hook=profile_hook)

    """
    if "__class__" in dkt:
        module_name, class_name = dkt["__class__"].rsplit(".", maxsplit=1)
        # the migration code should be called here
        try:
            del dkt["__class__"]
            module = importlib.import_module(module_name)
            return getattr(module, class_name)(**dkt)
        except Exception as e:  # skipcq: PTC-W0703`  # pylint: disable=W0703    # pragma: no cover
            dkt["__class__"] = module_name + "." + class_name
            dkt["__error__"] = e
    if "__ProfileDict__" in dkt:
        del dkt["__ProfileDict__"]
        res = ProfileDict(**dkt)
        return res
    if "__RadiusType__" in dkt:
        return RadiusType(dkt["value"])
    if "__SegmentationProperty__" in dkt:
        del dkt["__SegmentationProperty__"]
        res = ROIExtractionProfile(**dkt)
        return res
    if "__SegmentationProfile__" in dkt:
        del dkt["__SegmentationProfile__"]
        res = ROIExtractionProfile(**dkt)
        return res
    if (
        "__Serializable__" in dkt and dkt["__subtype__"] == "HistoryElement" and "algorithm_name" in dkt
    ):  # pragma: no cover
        # old code fix
        name = dkt["algorithm_name"]
        par = dkt["algorithm_values"]
        del dkt["algorithm_name"]
        del dkt["algorithm_values"]
        dkt["segmentation_parameters"] = {"algorithm_name": name, "values": par}
    if "__Serializable__" in dkt and dkt["__subtype__"] == "PartSegCore.color_image.base_colors.ColorMap":
        positions, colors = list(zip(*dkt["colormap"]))
        return Colormap(colors, controls=positions)
    if "__Serializable__" in dkt and dkt["__subtype__"] == "PartSegCore.color_image.base_colors.ColorPosition":
        return (dkt["color_position"], dkt["color"])
    if "__Serializable__" in dkt and dkt["__subtype__"] == "PartSegCore.color_image.base_colors.Color":
        return (dkt["red"] / 255, dkt["green"] / 255, dkt["blue"] / 255)
    if "__Colormap__" in dkt:
        del dkt["__Colormap__"]
        if dkt["controls"][0] != 0:
            dkt["controls"].insert(0, 0)
            dkt["colors"].insert(0, dkt["colors"][0])
        if dkt["controls"][-1] != 1:
            dkt["controls"].append(1)
            dkt["colors"].append(dkt["colors"][-1])
        return Colormap(**dkt)

    return serialize_hook(dkt)


def check_loaded_dict(dkt) -> bool:
    """
    Recursive check if dict `dkt` or any sub dict contains '__error__' key.

    :param dkt: dict to check
    """
    if not isinstance(dkt, (dict, EventedDict)):
        return True
    if "__error__" in dkt:
        return False
    return all(check_loaded_dict(val) for val in dkt.values())
