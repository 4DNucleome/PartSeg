from ..algorithm_describe_base import Register
from ..segmentation.restartable_segmentation_algorithms import final_algorithm_list

analysis_algorithm_dict = Register(
    *final_algorithm_list,
    class_methods=["support_time", "support_z"],
    methods=["set_image", "set_mask", "get_info_text", "calculation_run"],
)
"""Register for segmentation method designed for separate specific areas."""
