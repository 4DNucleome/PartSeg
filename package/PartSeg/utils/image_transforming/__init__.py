from ..algorithm_describe_base import Register
from .interpolate_image import InterpolateImage
from .transform_base import TransformBase


image_transform_dict = Register(InterpolateImage)