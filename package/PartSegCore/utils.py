import inspect
import typing
import weakref
from abc import ABC, abstractmethod
from types import MethodType

import numpy as np

__author__ = "Grzegorz Bokota"


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
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, *args, **kwarg):
        raise NotImplementedError()


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
