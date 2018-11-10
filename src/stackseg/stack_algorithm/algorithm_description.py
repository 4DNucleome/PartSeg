from project_utils.segmentation.segmentation_algorithm import final_algorithm_list
from collections import OrderedDict

stack_algorithm_dict = OrderedDict(((x.get_name(), x) for x in final_algorithm_list))
