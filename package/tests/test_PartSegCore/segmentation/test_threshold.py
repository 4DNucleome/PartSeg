import operator

import numpy as np
import pytest

from PartSegCore.segmentation.threshold import (
    BaseThreshold,
    IntermodesThreshold,
    KittlerIllingworthThreshold,
    double_threshold_dict,
    threshold_dict,
)

square = np.zeros((21, 21))
for i, v in [(2, 1000), (4, 10000), (6, 20000), (7, 21000), (8, 22000), (9, 23000)]:
    square[i:-i, i:-i] = v

cube = np.zeros((21, 21, 21))
for i, v in [(2, 1000), (4, 10000), (6, 20000), (7, 21000), (8, 22000), (9, 23000)]:
    cube[i:-i, i:-i, i:-i] = v


@pytest.mark.parametrize("method", threshold_dict.values())
@pytest.mark.parametrize("data", [square, cube], ids=["square", "cube"])
@pytest.mark.parametrize("op", [operator.lt, operator.gt])
@pytest.mark.parametrize("masking", [True, False])
def test_threshold(method: BaseThreshold, data, op, masking):
    mask = (data > 0) if masking else None

    try:
        data, thr_info = method.calculate_mask(data=data, mask=mask, arguments=method.get_default_values(), operator=op)
    except RuntimeError:
        if method is KittlerIllingworthThreshold:
            pytest.xfail("KittlerIllingworth sigma problem")
        if method is IntermodesThreshold:
            pytest.xfail("IntermodesThreshold sigma problem")
        raise
    assert isinstance(data, np.ndarray)
    assert isinstance(thr_info, (int, float))


@pytest.mark.parametrize("method", double_threshold_dict.values())
@pytest.mark.parametrize("data", [square, cube], ids=["square", "cube"])
@pytest.mark.parametrize("op", [operator.lt, operator.gt])
@pytest.mark.parametrize("masking", [True, False])
def test_double_threshold(method: BaseThreshold, data, op, masking):
    mask = (data > 0) if masking else None

    data, thr_info = method.calculate_mask(data=data, mask=mask, arguments=method.get_default_values(), operator=op)

    assert isinstance(data, np.ndarray)
    assert isinstance(thr_info[0], (int, float))
    assert isinstance(thr_info[1], (int, float))
