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
