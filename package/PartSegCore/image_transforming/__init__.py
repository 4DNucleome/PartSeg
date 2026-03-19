from PartSegCore.algorithm_describe_base import Register
from PartSegCore.image_transforming.combine_channels import CombineChannels, CombineMode
from PartSegCore.image_transforming.image_projection import ImageProjection
from PartSegCore.image_transforming.interpolate_image import InterpolateImage
from PartSegCore.image_transforming.swap_time_stack import SwapTimeStack
from PartSegCore.image_transforming.transform_base import TransformBase

image_transform_dict = Register(CombineChannels, ImageProjection, InterpolateImage, SwapTimeStack)

__all__ = (
    "CombineChannels",
    "CombineMode",
    "ImageProjection",
    "InterpolateImage",
    "SwapTimeStack",
    "TransformBase",
    "image_transform_dict",
)
