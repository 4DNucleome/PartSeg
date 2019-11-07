from ..algorithm_describe_base import Register
from .interpolate_image import InterpolateImage
from .transform_base import TransformBase
from .swap_time_stack import SwapTimeStack

image_transform_dict = Register(InterpolateImage, SwapTimeStack)

__all__ = ("image_transform_dict", "TransformBase")
