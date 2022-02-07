import copy
import dataclasses
import enum
import itertools
import json
import typing
from collections import defaultdict
from collections.abc import MutableMapping
from contextlib import suppress

import numpy as np
import pydantic
from psygnal import Signal

from ._old_json_hooks import part_hook
from .class_register import REGISTER, class_to_str
from .utils import CallbackBase, get_callback


class EventedDict(MutableMapping):
    setted = Signal(str)
    deleted = Signal(str)

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        cls = self.__class__
        result = cls(**copy.deepcopy(self._dict))
        result.base_key = self.base_key
        memodict[id(self)] = result
        return result

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


def add_class_info(obj, dkt):
    dkt["__class__"] = class_to_str(obj.__class__)
    dkt["__version__"] = str(REGISTER.get_version(obj.__class__))


class PartSegEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, enum.Enum):
            return {
                "__class__": class_to_str(o.__class__),
                "__version__": str(REGISTER.get_version(o.__class__)),
                "value": o.value,
            }
        if dataclasses.is_dataclass(o):
            fields = dataclasses.fields(o)
            dkt = {x.name: getattr(o, x.name) for x in fields}
            add_class_info(o, dkt)
            return dkt

        if isinstance(o, np.ndarray):
            return o.tolist()

        if isinstance(o, pydantic.BaseModel):
            try:
                dkt = dict(o)
            except (ValueError, TypeError):
                dkt = o.dict()
            add_class_info(o, dkt)
            return dkt

        if hasattr(o, "as_dict"):
            dkt = o.as_dict()
            add_class_info(o, dkt)
            return dkt

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, dict) and "__error__" in o:
            del o["__error__"]  # different environments without same plugins installed
        return super().default(o)


def partseg_object_hook(dkt: dict):
    if "__class__" in dkt:
        # the migration code should be called here
        cls_str = dkt.pop("__class__")
        version_str = dkt.pop("__version__") if "__version__" in dkt else "0.0.0"
        try:
            dkt_migrated = REGISTER.migrate_data(cls_str, version_str, dkt)
            cls = REGISTER.get_class(cls_str)
            return cls(**dkt_migrated)
        except Exception as e:  # pylint: disable=W0703
            dkt["__class__"] = cls_str
            dkt["__version__"] = version_str
            dkt["__error__"] = e

    if "__ReadOnly__" in dkt or "__Serializable__" in dkt:
        if "__Serializable__" in dkt:
            del dkt["__Serializable__"]
        else:
            del dkt["__ReadOnly__"]
        cls_str = dkt["__subtype__"]
        del dkt["__subtype__"]
        try:
            dkt_migrated = REGISTER.migrate_data(cls_str, "0.0.0", dkt)
            cls = REGISTER.get_class(cls_str)
            return cls(**dkt_migrated)
        except Exception:  # pylint: disable=W0703
            dkt["__subtype__"] = cls_str
            dkt["__Serializable__"] = True
    return part_hook(dkt)
