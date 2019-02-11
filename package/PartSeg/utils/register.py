from enum import Enum
from typing import Type
from .segmentation.algorithm_describe_base import AlgorithmDescribeBase
from .segmentation.sprawl import sprawl_dict
from .segmentation.threshold import threshold_dict
from .segmentation.noise_filtering import noise_removal_dict
from .analysis.algorithm_description import analysis_algorithm_dict
from .mask.algorithm_description import mask_algorithm_dict
from .analysis.save_functions import save_dict as analysis_save_dict
from .analysis.load_functions import load_dict as analysis_load_dict
from .mask.io_functions import load_dict as mask_load_dict
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


register_dict = {RegisterEnum.sprawl: sprawl_dict, RegisterEnum.threshold: threshold_dict,
                 RegisterEnum.noise_filtering: noise_removal_dict,
                 RegisterEnum.analysis_algorithm: analysis_algorithm_dict,
                 RegisterEnum.mask_algorithm: mask_algorithm_dict, RegisterEnum.analysis_save: analysis_save_dict,
                 RegisterEnum.analysis_load: analysis_load_dict, RegisterEnum.mask_load: mask_load_dict
                 }


def register(target: Type[AlgorithmDescribeBase], place: RegisterEnum, replace=False):
    register_dict[place].register(target, replace=replace)
