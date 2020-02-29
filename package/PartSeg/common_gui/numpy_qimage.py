from qtpy.QtGui import QImage
import numpy as np
import typing
from PartSegCore.color_image import color_image_fun, create_color_map, BaseColormap
from PartSegCore.color_image import resolution

ColorMapDict = typing.MutableMapping[str, typing.Tuple[BaseColormap, bool]]


class NumpyQImage(QImage):
    """
    Class for fix problem with PySide2 QImage implementation (non copied buffer)
    """

    def __init__(self, image: np.ndarray):
        super().__init__(
            image.data,
            image.shape[1],
            image.shape[0],
            image.dtype.itemsize * image.shape[1] * image.shape[2],
            QImage.Format_RGB888,
        )
        self.image = image


def colormap_array_to_image(array: np.ndarray) -> NumpyQImage:
    """
    Convert colormap in array format (:py:data:`.resolution`, 3) to :py:class:`~.NumpyQImage` instance
    """
    if array.shape != (resolution, 3):
        raise ValueError(f"Wrong shape ({array.shape}) of colormap")
    img = color_image_fun(np.linspace(0, 256, 512, endpoint=False).reshape((1, 512, 1)), [array], [(0, 255)])
    return NumpyQImage(img)


def convert_colormap_to_image(colormap: BaseColormap) -> NumpyQImage:
    """
    convert colormap to image of size (512, 1)

    :param colormap: colormap to convert
    :return: Color Bar image
    """
    color_array = create_color_map(colormap)
    return colormap_array_to_image(color_array)


def create_colormap_image(colormap: str, color_dict: ColorMapDict) -> NumpyQImage:
    """
    Convert named colormap to image of szie (512, 1)

    :param colormap: must be key in color_dict
    :param color_dict: dict mapping name to definition of colormap
    :return: Color Bar image
    """
    return convert_colormap_to_image((color_dict[colormap][0]))
