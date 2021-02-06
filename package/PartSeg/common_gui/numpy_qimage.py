import typing

import numpy as np
from napari.utils import Colormap
from napari.utils.colormaps import make_colorbar
from qtpy.QtGui import QImage

ColorMapDict = typing.MutableMapping[str, typing.Tuple[Colormap, bool]]


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
            QImage.Format_RGBA8888,
        )
        self.image = image


def convert_colormap_to_image(colormap: Colormap) -> NumpyQImage:
    """
    convert colormap to image of size (512, 1)

    :param colormap: colormap to convert
    :return: Color Bar image
    """
    return NumpyQImage(np.array(make_colorbar(colormap, size=(1, 512))))


def create_colormap_image(colormap: str, color_dict: ColorMapDict) -> NumpyQImage:
    """
    Convert named colormap to image of szie (512, 1)

    :param colormap: must be key in color_dict
    :param color_dict: dict mapping name to definition of colormap
    :return: Color Bar image
    """
    return convert_colormap_to_image(color_dict[colormap][0])
