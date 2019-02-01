import numpy as np
import SimpleITK as sitk
from enum import Enum
from .class_generator import enum_register

class RadiusType(Enum):
    NO = 0
    R2D = 1
    R3D = 2

    def __str__(self):
        return self.name

enum_register.register_class(RadiusType)


to_radius_type_dict = {"No": RadiusType.NO, "2d": RadiusType.R2D, "3d": RadiusType.R3D}


def _generic_image_operation(image, radius, fun, layer):
    if image.ndim == 2:
        layer = False
    if image.dtype == np.bool:
        image = image.astype(np.uint8)
    if isinstance(radius, (list, tuple)):
        radius = list(reversed(radius))
    if not layer:
        return sitk.GetArrayFromImage(fun(sitk.GetImageFromArray(image), radius))
    else:
        res = np.copy(image)
        for layer in res:
            layer[...] = sitk.GetArrayFromImage(fun(sitk.GetImageFromArray(layer), radius))
        return res

def gaussian(image, radius, layer=True):
    """
    :param image: image to apply gaussian filter
    :param radius: radius for gaussian kernel
    :return:
    """
    return _generic_image_operation(image, radius, sitk.DiscreteGaussian, layer)


def dilate(image, radius, layer=True):
    """
    :param image: image to apply gaussian filter
    :param radius: radius for gaussian kernel
    :return:
    """
    return _generic_image_operation(image, radius, sitk.GrayscaleDilate, layer)



def erode(image, radius, layer=True):
    """
    :param image: image to apply gaussian filter
    :param radius: radius for gaussian kernel
    :return:
    """
    return _generic_image_operation(image, radius, sitk.GrayscaleErode, layer)


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
