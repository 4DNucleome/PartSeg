"""
This module is designed to provide more stable api for PartSeg plugins.
All operations ar class based. If, in base object, some function is @classmethod, the it can be called
without creating class instance. So also plugins need to satisfy it.

RegisterEnum: used as key in all other members. Elements are described in its docstring

base_class_dict: Dict in which information about base class for given operation is stored.
The inheritance is not checked, but api must be implemented.

register: function for registering operation in inner structures.

register_dict: holds information where register given operation type. Strongly suggest to use register function instead.
"""
from enum import Enum
from typing import Type

from . import io_utils
from .algorithm_describe_base import AlgorithmDescribeBase
from .analysis import algorithm_description as analysis_algorithm_description
from .analysis import load_functions, measurement_base, measurement_calculation, save_functions
from .image_transforming import TransformBase, image_transform_dict
from .mask import algorithm_description as mask_algorithm_description
from .mask import io_functions
from .segmentation import (
    noise_filtering,
    restartable_segmentation_algorithms,
    segmentation_algorithm,
    threshold,
    watershed,
)

# from .mask.io_functions import

qss_list = []


class RegisterEnum(Enum):
    """
    Given types of operation are supported as plugins
    """

    sprawl = 0  #: algorithm for calculation sprawl from core object to borders. For spiting touching objects
    threshold = 1  #: threshold algorithms. From greyscale array to binary array
    noise_filtering = 2  #: filter noise from image
    analysis_algorithm = 3  #: algorithm for creating segmentation in analysis PartSeg part
    mask_algorithm = 4  #: algorithm for creating segmentation in mask PartSeg part
    analysis_save = 5  #: save functions for analysis part
    analysis_load = 6  #: load functions for analysis part
    mask_load = 7  #: load functions for mask part
    image_transform = 8  #: transform image, like interpolation
    mask_save_parameters = 9  #: save metadata for mask part (currently creating json file)
    mask_save_components = 10  #: save each segmentation component in separate file. Save location is directory
    mask_save_segmentation = 11  #: save project (to one file) in mask part
    analysis_measurement = 12  #: measurements algorithms (analysis mode)
    roi_analysis_segmentation_algorithm = 13  #: algorithm for creating segmentation in analysis PartSeg part
    roi_mask_segmentation_algorithm = 14  #: algorithm for creating segmentation in mask PartSeg part
    _qss_register = 15  #: new qss styles


# noinspection DuplicatedCode
register_dict = {
    RegisterEnum.sprawl: watershed.sprawl_dict,
    RegisterEnum.threshold: threshold.threshold_dict,
    RegisterEnum.noise_filtering: noise_filtering.noise_filtering_dict,
    RegisterEnum.analysis_algorithm: analysis_algorithm_description.analysis_algorithm_dict,
    RegisterEnum.mask_algorithm: mask_algorithm_description.mask_algorithm_dict,
    RegisterEnum.analysis_save: save_functions.save_dict,
    RegisterEnum.analysis_load: load_functions.load_dict,
    RegisterEnum.mask_load: io_functions.load_dict,
    RegisterEnum.image_transform: image_transform_dict,
    RegisterEnum.mask_save_parameters: io_functions.save_parameters_dict,
    RegisterEnum.mask_save_components: io_functions.save_components_dict,
    RegisterEnum.mask_save_segmentation: io_functions.save_segmentation_dict,
    RegisterEnum.analysis_measurement: measurement_calculation.MEASUREMENT_DICT,
    RegisterEnum.roi_analysis_segmentation_algorithm: analysis_algorithm_description.analysis_algorithm_dict,
    RegisterEnum.roi_mask_segmentation_algorithm: mask_algorithm_description.mask_algorithm_dict,
}

# noinspection DuplicatedCode
base_class_dict = {
    RegisterEnum.sprawl: watershed.BaseWatershed,
    RegisterEnum.threshold: threshold.BaseThreshold,
    RegisterEnum.noise_filtering: noise_filtering.NoiseFilteringBase,
    RegisterEnum.analysis_algorithm: restartable_segmentation_algorithms.RestartableAlgorithm,
    RegisterEnum.mask_algorithm: segmentation_algorithm.SegmentationAlgorithm,
    RegisterEnum.analysis_save: io_utils.SaveBase,
    RegisterEnum.analysis_load: io_utils.LoadBase,
    RegisterEnum.mask_load: io_utils.LoadBase,
    RegisterEnum.image_transform: TransformBase,
    RegisterEnum.mask_save_parameters: io_utils.SaveBase,
    RegisterEnum.mask_save_components: io_utils.SaveBase,
    RegisterEnum.mask_save_segmentation: io_utils.SaveBase,
    RegisterEnum.analysis_measurement: measurement_base.MeasurementMethodBase,
}  # dict with base class for given type of algorithm, keys are :py:class:`RegisterEnum`

reload_module_list = [
    threshold,
    watershed,
    segmentation_algorithm,
    restartable_segmentation_algorithms,
    noise_filtering,
    io_functions,
    mask_algorithm_description,
    analysis_algorithm_description,
    measurement_calculation,
    save_functions,
    load_functions,
]


def register(target: Type[AlgorithmDescribeBase], target_type: RegisterEnum, replace=False):
    """
    Function for registering new operations in PartSeg inner structures.

    :param target: operation to register in PartSeg inner structures
    :param target_type: Which type of operation.
    :param replace: force to replace operation if same name is defined. Dangerous.
    """
    if target_type == RegisterEnum._qss_register:  # pylint: disable=W0212
        qss_list.append(target)
    else:
        register_dict[target_type].register(target, replace=replace)
