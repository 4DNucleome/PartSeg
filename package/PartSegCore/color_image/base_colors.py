import typing

from napari.utils.colormaps.colormap import Colormap
from napari.utils.colormaps.colormap_utils import AVAILABLE_COLORMAPS

from PartSegCore.color_image.color_data import sitk_labels

Num = typing.Union[int, float]


class Color(typing.NamedTuple):
    """
    store color information

    :param red: red color value
    :param green: green color value
    :param blue: blue color value
    :param alpha: alpha value
    """

    red: float
    green: float
    blue: float
    alpha: float = 1


starting_colors = ["red", "green", "blue", "magenta", "inferno", "magma"]

default_colormap_dict = {name: AVAILABLE_COLORMAPS[name] for name in starting_colors}
default_colormap_dict.update(AVAILABLE_COLORMAPS)
default_colormap_dict.update(
    {
        k + "_reversed": Colormap(v.colors[::-1], controls=1 - v.controls[::-1])
        for k, v in AVAILABLE_COLORMAPS.items()
        if not k.endswith("_k")
    }
)


default_label_dict = {"default": sitk_labels}
