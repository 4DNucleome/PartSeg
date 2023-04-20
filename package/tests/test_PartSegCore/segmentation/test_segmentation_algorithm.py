from typing import Type

import numpy as np
import pytest

from PartSegCore.algorithm_describe_base import base_model_to_algorithm_property
from PartSegCore.segmentation import ROIExtractionAlgorithm
from PartSegCore.segmentation.algorithm_base import ROIExtractionResult, SegmentationLimitException
from PartSegCore.segmentation.restartable_segmentation_algorithms import (
    final_algorithm_list as restartable_list,
)
from PartSegCore.segmentation.restartable_segmentation_algorithms import (
    remove_object_touching_border,
)
from PartSegCore.segmentation.segmentation_algorithm import (
    CellFromNucleusFlow,
    ThresholdFlowAlgorithm,
)
from PartSegCore.segmentation.segmentation_algorithm import final_algorithm_list as algorithm_list
from PartSegCore.segmentation.utils import close_small_holes


def empty(*_):
    """Empty function to pass as a callback"""


@pytest.fixture(autouse=True)
def _fix_threshold_flow(monkeypatch):
    values = ThresholdFlowAlgorithm.get_default_values()
    values.threshold.values.core_threshold.values.threshold = 10
    values.threshold.values.base_threshold.values.threshold = 5

    def _param(self):
        return values

    monkeypatch.setattr(ThresholdFlowAlgorithm, "get_default_values", _param)

    values2 = CellFromNucleusFlow.get_default_values()
    values2.nucleus_threshold.values.threshold = 10
    values2.cell_threshold.values.threshold = 5

    def _param2(self):
        return values2

    monkeypatch.setattr(CellFromNucleusFlow, "get_default_values", _param2)


@pytest.mark.parametrize("algorithm", restartable_list + algorithm_list)
@pytest.mark.parametrize("masking", [True, False])
def test_segmentation_algorithm(image, algorithm: Type[ROIExtractionAlgorithm], masking):
    assert algorithm.support_z() is True
    assert algorithm.support_time() is False
    assert isinstance(algorithm.get_steps_num(), int)
    instance = algorithm()
    instance.set_image(image)
    if masking:
        instance.set_mask(image.get_channel(0) > 0)
    if instance.__new_style__:
        # FIXME when migrate whole code
        instance.set_parameters(instance.get_default_values())
    else:
        instance.set_parameters(**instance.get_default_values())
    if not masking and "Need mask" in base_model_to_algorithm_property(instance.__argument_class__):
        with pytest.raises(SegmentationLimitException):
            instance.calculation_run(empty)
    else:
        res = instance.calculation_run(empty)
        assert isinstance(instance.get_info_text(), str)
        assert isinstance(res, ROIExtractionResult)
    instance.clean()


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("dtype", [np.uint8, bool])
def test_close_small_holes(ndim, dtype):
    data = np.zeros((10,) * ndim, dtype=dtype)
    data[(slice(1, -1),) * ndim] = 1
    copy = data.copy()
    data[(slice(3, -3),) * ndim] = 0
    res = close_small_holes(data, 5**2)
    assert np.all(res == copy)


def test_remove_object_touching_border():
    data = np.zeros((10, 10), dtype=np.uint8)
    data[3:-3, 3:-3] = 1
    res = remove_object_touching_border(data)
    assert np.all(res == data)

    res = remove_object_touching_border(np.reshape(data, (1, 10, 10)))
    assert np.all(res == np.reshape(data, (1, 10, 10)))

    new_data = np.copy(data)
    new_data[:2] = 2

    res = remove_object_touching_border(new_data)
    assert np.all(res == data)

    res = remove_object_touching_border(np.reshape(new_data, (1, 10, 10)))
    assert np.all(res == np.reshape(data, (1, 10, 10)))
