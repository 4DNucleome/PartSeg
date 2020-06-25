from ..algorithm_describe_base import Register
from .interpolate_image import InterpolateImage
from .swap_time_stack import SwapTimeStack
from .transform_base import TransformBase

image_transform_dict = Register(InterpolateImage, SwapTimeStack)

__all__ = ("image_transform_dict", "TransformBase")
