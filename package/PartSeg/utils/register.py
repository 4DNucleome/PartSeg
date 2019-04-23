from enum import Enum
from typing import Type
from .algorithm_describe_base import AlgorithmDescribeBase
from .segmentation.sprawl import sprawl_dict, BaseSprawl
from .segmentation.threshold import threshold_dict, BaseThreshold
from .segmentation.noise_filtering import noise_removal_dict, NoiseFilteringBase
from .segmentation.restartable_segmentation_algorithms import RestartableAlgorithm
from .segmentation.segmentation_algorithm import SegmentationAlgorithm
from .analysis.algorithm_description import analysis_algorithm_dict
from .mask.algorithm_description import mask_algorithm_dict
from .analysis.save_functions import save_dict as analysis_save_dict
from .analysis.load_functions import load_dict as analysis_load_dict
from .mask.io_functions import load_dict as mask_load_dict, save_parameters_dict as mask_save_parameters_dict, \
    save_components_dict as mask_save_components_dict
from .image_transforming import image_transform_dict, TransformBase
from .io_utils import SaveBase, LoadBase
# from .mask.io_functions import


class RegisterEnum(Enum):
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


register_dict = {RegisterEnum.sprawl: sprawl_dict, RegisterEnum.threshold: threshold_dict,
                 RegisterEnum.noise_filtering: noise_removal_dict,
                 RegisterEnum.analysis_algorithm: analysis_algorithm_dict,
                 RegisterEnum.mask_algorithm: mask_algorithm_dict, RegisterEnum.analysis_save: analysis_save_dict,
                 RegisterEnum.analysis_load: analysis_load_dict, RegisterEnum.mask_load: mask_load_dict,
                 RegisterEnum.image_transform: image_transform_dict,
                 RegisterEnum.mask_save_parameters: mask_save_parameters_dict,
                 RegisterEnum.mask_save_components: mask_save_components_dict
                 }

base_class_dict = {RegisterEnum.sprawl: BaseSprawl, RegisterEnum.threshold: BaseThreshold,
                   RegisterEnum.noise_filtering: NoiseFilteringBase,
                   RegisterEnum.analysis_algorithm: RestartableAlgorithm,
                   RegisterEnum.mask_algorithm: SegmentationAlgorithm, RegisterEnum.analysis_save: SaveBase,
                   RegisterEnum.analysis_load: LoadBase, RegisterEnum.mask_load: LoadBase,
                   RegisterEnum.image_transform: TransformBase,
                   RegisterEnum.mask_save_parameters: SaveBase, RegisterEnum.mask_save_components: SaveBase
                   }


def register(target: Type[AlgorithmDescribeBase], place: RegisterEnum, replace=False):
    register_dict[place].register(target, replace=replace)
