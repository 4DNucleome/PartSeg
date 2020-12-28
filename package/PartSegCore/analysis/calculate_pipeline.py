import typing
from dataclasses import dataclass

import numpy as np

from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis.algorithm_description import analysis_algorithm_dict
from PartSegCore.analysis.analysis_utils import SegmentationPipeline
from PartSegCore.io_utils import HistoryElement
from PartSegCore.mask_create import calculate_mask
from PartSegCore.segmentation import RestartableAlgorithm
from PartSegCore.segmentation.algorithm_base import AdditionalLayerDescription
from PartSegImage import Image


def _empty_fun(_a1, _a2):
    pass


@dataclass(frozen=True)
class PipelineResult:
    roi: np.ndarray
    additional_layers: typing.Dict[str, AdditionalLayerDescription]
    mask: np.ndarray
    history: typing.List[HistoryElement]
    description: str


def calculate_pipeline(image: Image, mask: typing.Optional[np.ndarray], pipeline: SegmentationPipeline, report_fun):
    history = []
    report_fun("max", 2 * len(pipeline.mask_history) + 1)
    for i, el in enumerate(pipeline.mask_history):
        result, _ = calculate_segmentation_step(el.segmentation, image, mask)
        segmentation = image.fit_array_to_image(result.roi)
        report_fun("step", 2 * i + 1)
        new_mask = calculate_mask(
            mask_description=el.mask_property,
            segmentation=segmentation,
            old_mask=mask,
            spacing=image.spacing,
            time_axis=image.time_pos,
        )
        segmentation_parameters = {"algorithm_name": el.segmentation.name, "values": el.segmentation.values}
        history.append(HistoryElement.create(segmentation, mask, segmentation_parameters, el.mask_property))
        report_fun("step", 2 * i + 2)
        mask = image.fit_array_to_image(new_mask)
    result, text = calculate_segmentation_step(pipeline.segmentation, image, mask)
    report_fun("step", 2 * len(pipeline.mask_history) + 1)
    return PipelineResult(result.roi, result.additional_layers, mask, history, text)


def calculate_segmentation_step(profile: ROIExtractionProfile, image: Image, mask: typing.Optional[np.ndarray]):
    algorithm: RestartableAlgorithm = analysis_algorithm_dict[profile.algorithm]()
    algorithm.set_image(image)
    algorithm.set_mask(mask)
    parameters = profile.values
    algorithm.set_parameters(**parameters)
    return algorithm.calculation_run(_empty_fun), algorithm.get_info_text()
