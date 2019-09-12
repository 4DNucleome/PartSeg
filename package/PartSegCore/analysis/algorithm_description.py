from ..algorithm_describe_base import Register
from ..segmentation.restartable_segmentation_algorithms import final_algorithm_list

analysis_algorithm_dict = Register()

assert hasattr(analysis_algorithm_dict, "register")

for el in final_algorithm_list:
    analysis_algorithm_dict.register(el)
