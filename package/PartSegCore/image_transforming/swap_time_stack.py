import typing

from PartSegCore.algorithm_describe_base import AlgorithmProperty
from PartSegCore.image_transforming.transform_base import TransformBase
from PartSegCore.roi_info import ROIInfo
from PartSegImage import Image


class SwapTimeStack(TransformBase):
    @classmethod
    def transform(
        cls,
        image: Image,
        roi_info: ROIInfo,
        arguments: dict,
        callback_function: typing.Optional[typing.Callable[[str, int], None]] = None,
    ) -> typing.Tuple[Image, typing.Optional[ROIInfo]]:
        return image.swap_time_and_stack(), None

    @classmethod
    def get_fields_per_dimension(cls, image: Image):
        return cls.get_fields()

    @classmethod
    def calculate_initial(cls, image: Image):
        return cls.get_default_values()

    @classmethod
    def get_name(cls) -> str:
        return "Swap time and Z dim"

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return []
