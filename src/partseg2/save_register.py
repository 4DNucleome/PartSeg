import typing

from partseg_utils.io_utils import SaveBase
from partseg_utils.segmentation.algorithm_describe_base import Register

save_register: typing.Dict[str, typing.Type[SaveBase]] = Register(class_methods=["save", "get_name_with_suffix", "get_short_name"])
