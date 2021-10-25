import inspect
import weakref
from abc import ABC, abstractmethod
from typing import Callable

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
    def __init__(self, fun: Callable):
        self.fun = fun

    def is_alive(self):
        return True

    def __call__(self, *args, **kwarg):
        self.fun(*args, **kwarg)


class CallbackMethod(CallbackBase):
    def __init__(self, method):
        obj, name = self._get_proper_name(method)
        self.ref = weakref.ref(obj)
        self.name = name

    @staticmethod
    def _get_proper_name(callback):
        assert inspect.ismethod(callback)
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
            getattr(obj, self.name)(*args, **kwarg)


def get_callback(callable: Callable) -> CallbackBase:
    if inspect.ismethod(callable):
        return CallbackMethod(callable)

    return CallbackFun(callable)
