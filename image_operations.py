import numpy as np
import SimpleITK as sitk
from enum import Enum


def gaussian(image, radius):
    """
    :param image: image to apply gaussian filter
    :param radius: radius for gaussian kernel
    :return:
    """
    if len(image.shape) == 2:
        return sitk.GetArrayFromImage(sitk.DiscreteGaussian(sitk.GetImageFromArray(image), radius))
    res = np.copy(image)
    for layer in res:
        layer[...] = sitk.GetArrayFromImage(sitk.DiscreteGaussian(sitk.GetImageFromArray(layer), radius))
    return res


def dilate(image, radius):
    """
    :param image: image to apply gaussian filter
    :param radius: radius for gaussian kernel
    :return:
    """
    if len(image.shape) == 2:
        return sitk.GetArrayFromImage(sitk.GrayscaleDilate(sitk.GetImageFromArray(image), radius))
    res = np.copy(image)
    for layer in res:
        layer[...] = sitk.GetArrayFromImage(sitk.GrayscaleDilate(sitk.GetImageFromArray(layer), radius))
    return res


def to_binary_image(image):
    return np.array(image > 0).astype(np.uint8)


class DrawType(Enum):
    draw = 1
    erase = 2
    force_show = 3
    force_hide = 4
