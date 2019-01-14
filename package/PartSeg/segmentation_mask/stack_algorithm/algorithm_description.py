from ...partseg_utils.segmentation.segmentation_algorithm import final_algorithm_list
from ...partseg_utils.segmentation.algorithm_describe_base import Register

stack_algorithm_dict = Register()
for el in final_algorithm_list:
    stack_algorithm_dict.register(el)
