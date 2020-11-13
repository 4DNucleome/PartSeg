from typing import Type

import pytest

from PartSegCore.segmentation import SegmentationAlgorithm
from PartSegCore.segmentation.algorithm_base import SegmentationLimitException, SegmentationResult
from PartSegCore.segmentation.restartable_segmentation_algorithms import final_algorithm_list as restartable_list
from PartSegCore.segmentation.segmentation_algorithm import ThresholdFlowAlgorithm
from PartSegCore.segmentation.segmentation_algorithm import final_algorithm_list as algorithm_list


def empty(*args):
    pass


@pytest.fixture(autouse=True)
def fix_threshold_flow(monkeypatch):
    values = ThresholdFlowAlgorithm.get_default_values()
    values["threshold"]["values"]["core_threshold"]["values"]["threshold"] = 10
    values["threshold"]["values"]["base_threshold"]["values"]["threshold"] = 5

    def _param(self):
        return values

    monkeypatch.setattr(ThresholdFlowAlgorithm, "get_default_values", _param)


@pytest.mark.parametrize("algorithm", restartable_list + algorithm_list)
@pytest.mark.parametrize("masking", [True, False])
def test_segmentation_algorithm(image, algorithm: Type[SegmentationAlgorithm], masking):
    assert algorithm.support_z() is True
    assert algorithm.support_time() is False
    instance = algorithm()
    instance.set_image(image)
    if masking:
        instance.set_mask(image.get_channel(0) > 0)
    instance.set_parameters(**instance.get_default_values())
    if not masking and "Need mask" in algorithm.get_fields():
        with pytest.raises(SegmentationLimitException):
            instance.calculation_run(empty)
    else:
        res = instance.calculation_run(empty)
        assert isinstance(instance.get_info_text(), str)
        assert isinstance(res, SegmentationResult)
    instance.clean()
