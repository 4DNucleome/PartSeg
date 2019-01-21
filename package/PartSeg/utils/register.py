from enum import Enum
from .segmentation.sprawl import sprawl_dict
from .segmentation.threshold import threshold_dict
from .segmentation.noise_filtering import noise_removal_dict
from .analysis.algorithm_description import analysis_algorithm_dict
from .mask.algorithm_description import mask_algorithm_dict
from .analysis.save_functions import save_register
# from .mask.io_functions import

class RegisterEnum(Enum):
    sprawl = 0
    threshold = 1
    noise_filtering = 2
    analysis_algorithm = 3
    mask_algorithm = 4
    analysis_save = 5




register_dict = {RegisterEnum.sprawl: sprawl_dict, RegisterEnum.threshold: threshold_dict,
                 RegisterEnum.noise_filtering: noise_removal_dict,
                 RegisterEnum.analysis_algorithm: analysis_algorithm_dict,
                 RegisterEnum.mask_algorithm: mask_algorithm_dict, RegisterEnum.analysis_save: save_register
                 }



