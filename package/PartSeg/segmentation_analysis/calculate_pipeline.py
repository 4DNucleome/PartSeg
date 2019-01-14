import typing

from ..partseg_utils.class_generator import BaseReadonlyClass
from ..partseg_utils.segmentation.restartable_segmentation_algorithms import RestartableAlgorithm
from .analysis_utils import SegmentationPipeline, HistoryElement
from .algorithm_description import part_algorithm_dict
from ..partseg_utils.mask_create import calculate_mask
from PartSeg.tiff_image import Image
import numpy as np


def _empty_fun(_a1, _a2):
    pass


class PipelineResult(BaseReadonlyClass):
    segmentation: np.ndarray
    full_segmentation: np.ndarray
    mask: np.ndarray
    history: typing.List[HistoryElement]
    description: str


def calculate_pipeline(image: Image, mask: typing.Optional[np.ndarray], pipeline: SegmentationPipeline, report_fun):
    history = []
    report_fun("max", 2 * len(pipeline.mask_history) + 1)
    for i, el in enumerate(pipeline.mask_history):
        algorithm: RestartableAlgorithm = part_algorithm_dict[el.segmentation.algorithm]()
        algorithm.set_image(image)
        algorithm.set_mask(mask)
        algorithm.set_parameters(**el.segmentation.values)
        segmentation, full_segmentation, _ = algorithm.calculation_run(_empty_fun)
        report_fun("step", 2 * i + 1)
        new_mask = calculate_mask(el.mask_property, segmentation, mask, image.spacing)
        history.append(
            HistoryElement.create(segmentation, full_segmentation, mask, el.segmentation.name, el.segmentation.values,
                                  el.mask_property)
        )
        report_fun("step", 2 * i + 2)
        mask = new_mask
    algorithm: RestartableAlgorithm = part_algorithm_dict[pipeline.segmentation.algorithm]()
    algorithm.set_image(image)
    algorithm.set_mask(mask)
    algorithm.set_parameters(**pipeline.segmentation.values)
    segmentation, full_segmentation, _ = algorithm.calculation_run(_empty_fun)
    report_fun("step", 2 * len(pipeline.mask_history) + 1)
    return PipelineResult(segmentation, full_segmentation, mask, history, algorithm.get_info_text())
