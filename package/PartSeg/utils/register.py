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

from .algorithm_describe_base import AlgorithmDescribeBase
from PartSeg.utils.analysis import save_functions, load_functions, \
    algorithm_description as analysis_algorithm_description, statistics_calculation, measurement_base
from .image_transforming import image_transform_dict, TransformBase
from . import io_utils
from .segmentation import threshold, sprawl, segmentation_algorithm, restartable_segmentation_algorithms, \
    noise_filtering
from .mask import io_functions, algorithm_description as mask_algorithm_description


# from .mask.io_functions import


class RegisterEnum(Enum):
    """
    Given types of operation are supported as plugins:
    sprawl: algorithm for calculation sprawl from core object to borders. For spiting touching objects
    threshold: threshold algorithms. From greyscale array to binary array
    noise_filtering: filter noise
    image_transform = 8
    analysis_algorithm: algorithm for creating segmentation in analysis PartSeg part
    mask_algorithm: algorithm for creating segmentation in mask PartSeg part
    analysis_save: save functions for analysis part
    analysis_load: load functions for analysis part
    mask_load: load functions for mask part
    mask_save_parameters = save metadata for mask part (currently creating json file)
    mask_save_components = save each segmentation component in separate file. Save location is directory
    mask_save_segmentation = save project (to one file) in mask part
    """
    sprawl = 0
    threshold = 1
    noise_filtering = 2
    analysis_algorithm = 3
    mask_algorithm = 4
    analysis_save = 5
    analysis_load = 6
    mask_load = 7
    image_transform = 8
    mask_save_parameters = 9
    mask_save_components = 10
    mask_save_segmentation = 11
    analysis_measurement = 12


register_dict = {
    RegisterEnum.sprawl: sprawl.sprawl_dict, RegisterEnum.threshold: threshold.threshold_dict,
    RegisterEnum.noise_filtering: noise_filtering.noise_removal_dict,
    RegisterEnum.analysis_algorithm: analysis_algorithm_description.analysis_algorithm_dict,
    RegisterEnum.mask_algorithm: mask_algorithm_description.mask_algorithm_dict,
    RegisterEnum.analysis_save: save_functions.save_dict,
    RegisterEnum.analysis_load: load_functions.load_dict, RegisterEnum.mask_load: io_functions.load_dict,
    RegisterEnum.image_transform: image_transform_dict,
    RegisterEnum.mask_save_parameters: io_functions.save_parameters_dict,
    RegisterEnum.mask_save_components: io_functions.save_components_dict,
    RegisterEnum.mask_save_segmentation: io_functions.save_segmentation_dict,
    RegisterEnum.analysis_measurement: statistics_calculation.STATISTIC_DICT
}

base_class_dict = {
    RegisterEnum.sprawl: sprawl.BaseSprawl, RegisterEnum.threshold: threshold.BaseThreshold,
    RegisterEnum.noise_filtering: noise_filtering.NoiseFilteringBase,
    RegisterEnum.analysis_algorithm: restartable_segmentation_algorithms.RestartableAlgorithm,
    RegisterEnum.mask_algorithm: segmentation_algorithm.SegmentationAlgorithm,
    RegisterEnum.analysis_save: io_utils.SaveBase,
    RegisterEnum.analysis_load: io_utils.LoadBase, RegisterEnum.mask_load: io_utils.LoadBase,
    RegisterEnum.image_transform: TransformBase,  RegisterEnum.mask_save_parameters: io_utils.SaveBase,
    RegisterEnum.mask_save_components: io_utils.SaveBase, RegisterEnum.mask_save_segmentation: io_utils.SaveBase,
    RegisterEnum.analysis_measurement: measurement_base.StatisticMethodBase
}

reload_module_list = \
    [threshold, sprawl, segmentation_algorithm, restartable_segmentation_algorithms, noise_filtering, io_functions,
     mask_algorithm_description, analysis_algorithm_description, statistics_calculation, save_functions, load_functions,
     ]


def register(target: Type[AlgorithmDescribeBase], target_type: RegisterEnum, replace=False):
    """
    Function for registering new operations in PartSeg inner structures.
    :param target: operation to register in PartSeg inner structures
    :param target_type: Which type of operation.
    :param replace: force to replace operation if same name is defined. Dangerous.
    :return:
    """
    register_dict[target_type].register(target, replace=replace)
