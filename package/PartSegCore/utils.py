import copy
import inspect
import itertools
import typing
import warnings
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import suppress
from types import MethodType

import numpy as np
from local_migrator import register_class
from psygnal import Signal
from pydantic import BaseModel as PydanticBaseModel
from sentry_sdk.utils import safe_repr as _safe_repr

__author__ = "Grzegorz Bokota"

if typing.TYPE_CHECKING:
    from napari.layers import Image


def bisect(arr, val, comp):
    left = -1
    right = len(arr)
    while right - left > 1:
        mid = (left + right) >> 1
        if comp(arr[mid], val):
            left = mid
        else:
            right = mid
    return right


def numpy_repr(val: np.ndarray):
    if val is None:  # pragma: no cover
        return repr(val)
    if val.size < 20:
        return repr(val)
    return f"array(size={val.size}, shape={val.shape}, dtype={val.dtype}, min={val.min()}, max={val.max()})"


class CallbackBase(ABC):
    @abstractmethod
    def is_alive(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwarg):
        raise NotImplementedError


class CallbackFun(CallbackBase):
    def __init__(self, fun: typing.Callable, max_args: typing.Optional[int] = None):
        self.fun = fun
        self.count = _inspect_signature(fun) if max_args is None else max_args

    def is_alive(self):
        return True

    def __call__(self, *args, **kwarg):
        self.fun(*args[: self.count], **kwarg)


class CallbackMethod(CallbackBase):
    def __init__(self, method, max_args: typing.Optional[int] = None):
        obj, name = self._get_proper_name(method)
        self.ref = weakref.ref(obj)
        self.name = name
        self.count = _inspect_signature(method) if max_args is None else max_args

    @staticmethod
    def _get_proper_name(callback):
        obj = callback.__self__
        if not hasattr(obj, callback.__name__) or getattr(obj, callback.__name__) != callback:
            # some decorators will alter method.__name__, so that obj.method
            # will not be equal to getattr(obj, obj.method.__name__). We check
            # for that case here and traverse to find the right method here.
            for name in dir(obj):
                meth = getattr(obj, name)
                if inspect.ismethod(meth) and meth == callback:
                    return obj, name
            raise RuntimeError(f"During bind method {callback} of object {obj} an error happen")
        return obj, callback.__name__

    def is_alive(self):
        return self.ref() is not None

    def __call__(self, *args, **kwarg):
        obj = self.ref()
        if obj is not None:
            getattr(obj, self.name)(*args[: self.count], **kwarg)


def _inspect_signature(slot: typing.Callable) -> typing.Optional[int]:
    """
    count maximal number of positional argument
    :param slot: callable to be checked
    :return: number of parameters which could be passed to callable, None if unbound
    """
    if hasattr(slot, "__module__") and isinstance(slot.__module__, str) and slot.__module__.startswith("superqt"):
        return 0
    try:
        signature = inspect.signature(slot)
    except ValueError:
        return 0
    count = 0
    for parameter in signature.parameters.values():
        if parameter.kind in [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD]:
            count += 1
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            count = None
            break
    return count


def get_callback(callback: typing.Union[typing.Callable, MethodType], max_args=None) -> CallbackBase:
    if inspect.ismethod(callback):
        return CallbackMethod(callback, max_args)

    return CallbackFun(callback, max_args)


@register_class(old_paths=["PartSegCore.json_hooks.EventedDict"], allow_errors_in_values=True)
class EventedDict(typing.MutableMapping):
    """
    Class for storing data in dictionary with possibility to connect to change of data.

    :param klass: class of stored data. It could be Type or dict with key as name of key and value as class of data.
        Key "*" is used as default class, for key different form specified one. .
    """

    setted = Signal(str)
    deleted = Signal(str)

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        cls = self.__class__
        result = cls(self._klass, **copy.deepcopy(self._dict))
        result.base_key = self.base_key
        memodict[id(self)] = result
        return result

    def __init__(self, klass=None, /, **kwargs):
        super().__init__()
        self._dict = {}
        if klass is None:
            klass = object
        self._klass = klass if isinstance(klass, dict) else {"*": klass}

        for key, dkt in kwargs.items():
            self[key] = dkt
        self.base_key = ""

    def __setitem__(self, k, v) -> None:
        klass = self._klass.get(k, self._klass.get("*", object))
        if isinstance(klass, dict):
            if not isinstance(v, typing.MutableMapping):
                raise TypeError(f"Value for key {k} should be dict")
        elif not isinstance(v, klass):
            raise TypeError(f"Value {v} for key {k} is not instance of {klass}")

        if isinstance(v, dict):
            v = EventedDict(klass, **v)
        if isinstance(v, EventedDict):
            v.base_key = k
            v._klass = klass if isinstance(klass, dict) else {"*": klass}
            v.setted.connect(self._propagate_setitem)
            v.deleted.connect(self._propagate_del)
        if k in self._dict and isinstance(self._dict[k], EventedDict):
            self._dict[k].setted.disconnect(self._propagate_setitem)
            self._dict[k].deleted.disconnect(self._propagate_del)
        old_value = self._dict.get(k)
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
        return f"EventedDict[{self._klass}]({self._dict})"

    def __repr__(self):
        return f"EventedDict(klass={self._klass}, {self._dict!r})"

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


def recursive_update_dict(main_dict: typing.MutableMapping, other_dict: typing.MutableMapping):
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
        if (
            key in main_dict
            and isinstance(main_dict[key], typing.MutableMapping)
            and isinstance(val, typing.MutableMapping)
        ):
            recursive_update_dict(main_dict[key], val)
        else:
            main_dict[key] = val


@register_class(old_paths=["PartSegCore.json_hooks.ProfileDict"], allow_errors_in_values=True)
class ProfileDict:
    """
    Dict for storing recursive data. The path is dot separated.

    :param klass: class of stored data. Same as in :py:class:`EventedDict`
    :param kwargs: initial data

    >>> dkt = ProfileDict()
    >>> dkt.set(["aa", "bb", "c1"], 7)
    >>> dkt.get(["aa", "bb", "c1"])
        7
    >>> dkt.get("aa.bb.c2", 8)
        8
    >>> dkt.get("aa.bb")
        {'c1': 7, 'c2': 8}
    """

    def __init__(self, klass=None, **kwargs):
        self._my_dict = EventedDict(klass, **kwargs)
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
        i = 0
        try:
            for i, key in enumerate(key_path[:-1]):  # noqa: B007
                # TODO add check if next step element is dict and create custom information
                curr_dict = curr_dict[key]
        except KeyError:
            for key2 in key_path[i:-1]:
                with curr_dict.setted.blocked():
                    curr_dict[key2] = {}
                curr_dict = curr_dict[key2]
        if isinstance(value, dict):
            value = EventedDict(**value)
        curr_dict[key_path[-1]] = value
        return curr_dict[key_path[-1]]

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
        try:
            for key in key_path:
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

    def filter_data(self):  # pragma: no cover
        warnings.warn("Deprecated, use pop errors instead", FutureWarning, stacklevel=2)
        self.pop_errors()

    def pop_errors(self) -> typing.List[typing.Tuple[str, dict]]:
        """Remove problematic entries from dict"""
        error_list = []
        for group, up_dkt in list(self.my_dict.items()):
            if not isinstance(up_dkt, (dict, EventedDict)):
                continue
            error_list.extend(
                (f"{group}.{key}", up_dkt.pop(key)) for key, dkt in list(up_dkt.items()) if not check_loaded_dict(dkt)
            )

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


class BaseModel(PydanticBaseModel):
    class Config:
        extra = "forbid"

    def __getitem__(self, item):
        if item in self.__fields__:
            warnings.warn("Access to attribute by [] is deprecated. Use . instead", FutureWarning, stacklevel=2)
            return getattr(self, item)
        raise KeyError(f"{item} not found in {self.__class__.__name__}")

    def copy(self: PydanticBaseModel, *, validate: bool = True, **kwargs: typing.Any) -> PydanticBaseModel:
        copy_res = super().copy(**kwargs)
        if validate:
            return self.validate(
                dict(
                    copy_res._iter(  # pylint: disable=protected-access
                        to_dict=False, by_alias=False, exclude_unset=True
                    )
                )
            )
        return copy_res


def iterate_names(base_name: str, data_dict, max_length=None) -> typing.Optional[str]:
    if base_name not in data_dict:
        return base_name[:max_length]
    if max_length is not None:
        max_length -= 5
    for i in range(1, 100):
        res_name = f"{base_name[:max_length]} ({i})"
        if res_name not in data_dict:
            return res_name
    return None


def napari_image_repr(image: "Image") -> str:
    return (
        f"<Image of shape: {image.data.shape}, dtype: {image.data.dtype}, "
        f"slice {getattr(image, '_slice_indices', None)}>"
    )


def safe_repr(val):
    from napari.layers import Image

    if isinstance(val, np.ndarray):
        return numpy_repr(val)
    if isinstance(val, Image):
        return napari_image_repr(val)
    return _safe_repr(val)
