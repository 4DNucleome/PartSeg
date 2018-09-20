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
    if isinstance(radius, (tuple, list)):
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
    if image.dtype == np.bool:
        image = image.astype(np.uint8)
    if len(image.shape) == 2:
        return sitk.GetArrayFromImage(sitk.GrayscaleDilate(sitk.GetImageFromArray(image), radius))
    res = np.copy(image)
    print(image.shape, image.dtype)
    for layer in res:
        layer[...] = sitk.GetArrayFromImage(sitk.GrayscaleDilate(sitk.GetImageFromArray(layer), radius))
    return res


def erode(image, radius):
    """
    :param image: image to apply gaussian filter
    :param radius: radius for gaussian kernel
    :return:
    """
    if image.dtype == np.bool:
        image = image.astype(np.uint8)
    if len(image.shape) == 2:
        return sitk.GetArrayFromImage(sitk.GrayscaleErode(sitk.GetImageFromArray(image), radius))
    res = np.copy(image)
    for layer in res:
        layer[...] = sitk.GetArrayFromImage(sitk.GrayscaleErode(sitk.GetImageFromArray(layer), radius))
    return res


def to_binary_image(image):
    return np.array(image > 0).astype(np.uint8)


class DrawType(Enum):
    draw = 1
    erase = 2
    force_show = 3
    force_hide = 4


def normalize_shape(image):
    if len(image.shape) == 4:
        if image.shape[-1] > 10:
            image = np.swapaxes(image, 1, 3)
            image = np.swapaxes(image, 1, 2)
    elif len(image.shape) == 3:
        if image.shape[-1] > 10:
            image = image.reshape(image.shape + (1,))
        else:
            image = image.reshape((1,) + image.shape)
    else:
        image = image.reshape((1,) + image.shape + (1,))
    return image
