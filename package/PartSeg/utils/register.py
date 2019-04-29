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
from .analysis.algorithm_description import analysis_algorithm_dict
from .analysis.load_functions import load_dict as analysis_load_dict
from .analysis.save_functions import save_dict as analysis_save_dict
from .image_transforming import image_transform_dict, TransformBase
from .io_utils import SaveBase, LoadBase
from .mask.algorithm_description import mask_algorithm_dict
from .mask.io_functions import load_dict as mask_load_dict, save_parameters_dict as mask_save_parameters_dict, \
    save_components_dict as mask_save_components_dict, save_segmentation_dict as mask_save_segmentation_dict
from .segmentation.noise_filtering import noise_removal_dict, NoiseFilteringBase
from .segmentation.restartable_segmentation_algorithms import RestartableAlgorithm
from .segmentation.segmentation_algorithm import SegmentationAlgorithm
from .segmentation.sprawl import sprawl_dict, BaseSprawl
from .segmentation.threshold import threshold_dict, BaseThreshold


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


register_dict = {
    RegisterEnum.sprawl: sprawl_dict, RegisterEnum.threshold: threshold_dict,
    RegisterEnum.noise_filtering: noise_removal_dict, RegisterEnum.analysis_algorithm: analysis_algorithm_dict,
    RegisterEnum.mask_algorithm: mask_algorithm_dict, RegisterEnum.analysis_save: analysis_save_dict,
    RegisterEnum.analysis_load: analysis_load_dict, RegisterEnum.mask_load: mask_load_dict,
    RegisterEnum.image_transform: image_transform_dict, RegisterEnum.mask_save_parameters: mask_save_parameters_dict,
    RegisterEnum.mask_save_components: mask_save_components_dict,
    RegisterEnum.mask_save_segmentation: mask_save_segmentation_dict
}

base_class_dict = {
    RegisterEnum.sprawl: BaseSprawl, RegisterEnum.threshold: BaseThreshold,
    RegisterEnum.noise_filtering: NoiseFilteringBase, RegisterEnum.analysis_algorithm: RestartableAlgorithm,
    RegisterEnum.mask_algorithm: SegmentationAlgorithm, RegisterEnum.analysis_save: SaveBase,
    RegisterEnum.analysis_load: LoadBase, RegisterEnum.mask_load: LoadBase,
    RegisterEnum.image_transform: TransformBase,  RegisterEnum.mask_save_parameters: SaveBase,
    RegisterEnum.mask_save_components: SaveBase, RegisterEnum.mask_save_segmentation: SaveBase
}


def register(target: Type[AlgorithmDescribeBase], target_type: RegisterEnum, replace=False):
    """
    Function for registering new operations in PartSeg inner structures.
    :param target: operation to register in PartSeg inner structures
    :param target_type: Which type of operation.
    :param replace: force to replace operation if same name is defined. Dangerous.
    :return:
    """
    register_dict[target_type].register(target, replace=replace)
