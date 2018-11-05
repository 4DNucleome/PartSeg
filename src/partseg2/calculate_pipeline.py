import typing
from .partseg_utils import SegmentationPipeline, HistoryElement
from .algorithm_description import part_algorithm_dict
from project_utils.mask_create import calculate_mask
from tiff_image import Image
import numpy as np


def _empty_fun(*_args, **_kwargs):
    pass


def calculate_pipeline(image: Image, mask: typing.Union[np.ndarray, None], pipeline: SegmentationPipeline, report_fun):
    history = []
    report_fun("max", 2 * len(pipeline.mask_history) + 1)
    for i, el in enumerate(pipeline.mask_history):
        algorithm = part_algorithm_dict[el.segmentation.name][0]()
        algorithm.set_image(image)
        algorithm.set_mask(mask)
        algorithm.set_parameters(**el.segmentation.values)
        segmentation, full_segmentation = algorithm.calculation_run(_empty_fun)
        report_fun("step", 2 * i + 1)
        new_mask = calculate_mask(el.mask_property, segmentation, mask, image.spacing)
        history.append(
            HistoryElement.create(segmentation, full_segmentation, mask, el.segmentation.name, el.segmentation.values,
                                  el.mask_property)
        )
        report_fun("step", 2 * i + 2)
        mask = new_mask
    algorithm = part_algorithm_dict[pipeline.segmentation.name][0]()
    algorithm.set_image(image)
    algorithm.set_mask(mask)
    algorithm.set_parameters(**pipeline.segmentation.values)
    segmentation, full_segmentation = algorithm.calculation_run(_empty_fun)
    report_fun("step", 2 * len(pipeline.mask_history) + 1)
    return segmentation, full_segmentation, history
