from abc import ABC
from typing import Callable, List, Optional, Tuple

from PartSegCore.algorithm_describe_base import AlgorithmDescribeBase
from PartSegCore.roi_info import ROIInfo
from PartSegImage import Image


class TransformBase(AlgorithmDescribeBase, ABC):
    @classmethod
    def transform(
        cls,
        image: Image,
        roi_info: ROIInfo,
        arguments: dict,
        callback_function: Optional[Callable[[str, int], None]] = None,
    ) -> Tuple[Image, Optional[ROIInfo]]:
        raise NotImplementedError

    @classmethod
    def get_fields_per_dimension(cls, component_list: List[str]):
        raise NotImplementedError

    @classmethod
    def calculate_initial(cls, image: Image):
        raise NotImplementedError
