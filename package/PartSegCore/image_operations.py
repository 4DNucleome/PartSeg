from enum import Enum

import numpy as np
import SimpleITK as sitk

from .class_generator import enum_register


class RadiusType(Enum):
    """
    If operation should be performed and if on each layer separately on globally
    """
    NO = 0  #: No operation
    R2D = 1  #: operation in each layer separately
    R3D = 2  #: operation on whole stack

    def __str__(self):
        return self.name


enum_register.register_class(RadiusType)


def _generic_image_operation(image, radius, fun, layer):
    if image.ndim == 2:
        layer = False
    if image.dtype == np.bool:
        image = image.astype(np.uint8)
    if isinstance(radius, (list, tuple)):
        radius = list(reversed(radius))
    if not layer or image.ndim == 2:
        return sitk.GetArrayFromImage(fun(sitk.GetImageFromArray(image), radius))
    else:
        res = np.copy(image)
        for layer in res:
            layer[...] = sitk.GetArrayFromImage(fun(sitk.GetImageFromArray(layer), radius))
        return res


def gaussian(image: np.ndarray, radius: float, layer=True):
    """
    Gaussian blur of image

    :param image: image to apply gaussian filter
    :param radius: radius for gaussian kernel
    :param layer: if operation should be run on each layer separately
    :return:
    """
    return _generic_image_operation(image, radius, sitk.DiscreteGaussian, layer)


def median(image, radius, layer=True):
    """
    Median blur of image

    :param image: image to apply median filter
    :param radius: radius for median kernel
    :param layer: if operation should be run on each layer separately
    :return:
    """
    return _generic_image_operation(image, radius, sitk.Median, layer)


def dilate(image, radius, layer=True):
    """
    Dilate of image

    :param image: image to apply dilation
    :param radius: dilation radius
    :param layer: if operation should be run on each layer separately
    :return:
    """
    return _generic_image_operation(image, radius, sitk.GrayscaleDilate, layer)


def erode(image, radius, layer=True):
    """
    Erosion of image

    :param image: image to apply erosion
    :param radius: erosion radius
    :param layer: if operation should be run on each layer separately
    :return:
    """
    return _generic_image_operation(image, radius, sitk.GrayscaleErode, layer)


def to_binary_image(image):
    """Convert image to binary. All positive values are set to 1."""
    return np.array(image > 0).astype(np.uint8)
