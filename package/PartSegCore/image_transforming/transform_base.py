from abc import ABC
from typing import List, Callable, Optional

from PartSegImage import Image
from PartSegCore.algorithm_describe_base import AlgorithmDescribeBase


class TransformBase(AlgorithmDescribeBase, ABC):
    @classmethod
    def transform(
        cls, image: Image, arguments: dict, callback_function: Optional[Callable[[str, int], None]] = None
    ) -> Image:
        raise NotImplementedError

    @classmethod
    def get_fields_per_dimension(cls, component_list: List[str]):
        raise NotImplementedError

    @classmethod
    def calculate_initial(cls, image: Image):
        raise NotImplementedError
