import os
from copy import deepcopy

import numpy as np
import pytest

from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis import (
    ProjectTuple,
    SegmentationPipeline,
    SegmentationPipelineElement,
    analysis_algorithm_dict,
)
from PartSegCore.analysis.calculate_pipeline import calculate_pipeline
from PartSegCore.analysis.load_functions import LoadProject
from PartSegCore.analysis.save_functions import SaveProject
from PartSegCore.mask_create import calculate_mask


@pytest.mark.parametrize("channel", [0, 1])
def test_simple(image, algorithm_parameters, channel):
    algorithm_parameters["values"]["channel"] = channel
    algorithm = analysis_algorithm_dict[algorithm_parameters["algorithm_name"]]()
    algorithm.set_image(image)
    algorithm.set_parameters(**algorithm_parameters["values"])
    result = algorithm.calculation_run(lambda x, y: None)
    assert np.max(result.roi) == 1


def test_pipeline_manual(image, algorithm_parameters, mask_property):
    algorithm = analysis_algorithm_dict[algorithm_parameters["algorithm_name"]]()
    algorithm.set_image(image)
    algorithm.set_parameters(**algorithm_parameters["values"])
    result = algorithm.calculation_run(lambda x, y: None)
    mask = calculate_mask(mask_property, result.roi, None, image.spacing)
    algorithm_parameters["values"]["channel"] = 1
    algorithm.set_parameters(**algorithm_parameters["values"])
    algorithm.set_mask(mask)
    result2 = algorithm.calculation_run(lambda x, y: None)
    assert np.max(result2.roi) == 2


@pytest.mark.parametrize("use_mask", [True, False])
def test_pipeline(image, algorithm_parameters, mask_property, tmp_path, use_mask):
    elem = SegmentationPipelineElement(
        segmentation=ROIExtractionProfile(
            name="", algorithm=algorithm_parameters["algorithm_name"], values=algorithm_parameters["values"]
        ),
        mask_property=mask_property,
    )
    algorithm_parameters = deepcopy(algorithm_parameters)
    algorithm_parameters["values"]["channel"] = 1
    pipeline = SegmentationPipeline(
        name="",
        segmentation=ROIExtractionProfile(
            name="", algorithm=algorithm_parameters["algorithm_name"], values=algorithm_parameters["values"]
        ),
        mask_history=[elem],
    )
    mask = np.ones(image.get_channel(0).shape, dtype=np.uint8) if use_mask else None
    result = calculate_pipeline(image, mask, pipeline, lambda x, y: None)
    assert np.max(result.roi_info.roi) == 2
    pt = ProjectTuple(
        file_path=image.file_path,
        image=image,
        roi_info=result.roi_info,
        mask=result.mask,
        history=result.history,
        algorithm_parameters=algorithm_parameters,
    )
    SaveProject.save(tmp_path / "project.tgz", pt)
    assert os.path.exists(tmp_path / "project.tgz")
    loaded = LoadProject.load([tmp_path / "project.tgz"])
    assert np.all(loaded.roi_info.roi == result.roi_info.roi)
