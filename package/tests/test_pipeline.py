import os
from copy import deepcopy

import numpy as np
import pytest

from PartSegCore.algorithm_describe_base import SegmentationProfile
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
    assert np.max(result.segmentation) == 1


def test_pipeline_manual(image, algorithm_parameters, mask_property):
    algorithm = analysis_algorithm_dict[algorithm_parameters["algorithm_name"]]()
    algorithm.set_image(image)
    algorithm.set_parameters(**algorithm_parameters["values"])
    result = algorithm.calculation_run(lambda x, y: None)
    mask = calculate_mask(mask_property, result.segmentation, None, image.spacing)
    algorithm_parameters["values"]["channel"] = 1
    algorithm.set_parameters(**algorithm_parameters["values"])
    algorithm.set_mask(mask)
    result2 = algorithm.calculation_run(lambda x, y: None)
    assert np.max(result2.segmentation) == 2


def test_pipeline(image, algorithm_parameters, mask_property, tmp_path):
    elem = SegmentationPipelineElement(
        segmentation=SegmentationProfile(
            name="", algorithm=algorithm_parameters["algorithm_name"], values=algorithm_parameters["values"]
        ),
        mask_property=mask_property,
    )
    algorithm_parameters = deepcopy(algorithm_parameters)
    algorithm_parameters["values"]["channel"] = 1
    pipeline = SegmentationPipeline(
        name="",
        segmentation=SegmentationProfile(
            name="", algorithm=algorithm_parameters["algorithm_name"], values=algorithm_parameters["values"]
        ),
        mask_history=[elem],
    )
    result = calculate_pipeline(image, None, pipeline, lambda x, y: None)
    assert np.max(result.segmentation) == 2
    pt = ProjectTuple(
        file_path=image.file_path,
        image=image,
        segmentation=result.segmentation,
        mask=result.mask,
        history=result.history,
        algorithm_parameters=algorithm_parameters,
    )
    SaveProject.save(tmp_path / "project.tgz", pt)
    assert os.path.exists(tmp_path / "project.tgz")
    loaded = LoadProject.load([tmp_path / "project.tgz"])
    assert np.all(loaded.segmentation == result.segmentation)
