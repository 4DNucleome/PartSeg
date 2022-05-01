import typing

from napari.utils.colormaps.colormap import Colormap
from napari.utils.colormaps.colormap_utils import AVAILABLE_COLORMAPS
from pydantic import Field

from PartSegCore.color_image.color_data import sitk_labels
from PartSegCore.utils import BaseModel

Num = typing.Union[int, float]


class Color(BaseModel):
    """
    store color information

    :param red: red color value
    :param green: green color value
    :param blue: blue color value
    :param alpha: alpha value
    """

    red: float = Field(ge=0.0, le=1.0)
    green: float = Field(ge=0.0, le=1.0)
    blue: float = Field(ge=0.0, le=1.0)
    alpha: float = Field(1, ge=0.0, le=1.0)

    def as_tuple(self):
        return (self.red, self.green, self.blue, self.alpha)

    @classmethod
    def from_tuple(cls, tup):
        """
        create color from tuple

        :param tup: tuple of color values
        :return: color
        """
        return cls(red=tup[0], green=tup[1], blue=tup[2], alpha=tup[3] if len(tup) > 3 else 1)

    def is_close(self, other: "Color", epsilon: float = 1e-6) -> bool:
        """
        check if color is close to other color

        :param other: other color
        :param epsilon: epsilon value
        :return: True if colors are close
        """
        return (
            abs(self.red - other.red) < epsilon
            and abs(self.green - other.green) < epsilon
            and abs(self.blue - other.blue) < epsilon
            and abs(self.alpha - other.alpha) < epsilon
        )


starting_colors = ["red", "green", "blue", "magenta", "inferno", "magma"]

default_colormap_dict = {name: AVAILABLE_COLORMAPS[name] for name in starting_colors}
default_colormap_dict.update(AVAILABLE_COLORMAPS)
default_colormap_dict.update(
    {
        f"{k}_reversed": Colormap(v.colors[::-1], controls=1 - v.controls[::-1])
        for k, v in AVAILABLE_COLORMAPS.items()
        if not k.endswith("_k")
    }
)


default_label_dict = {"default": sitk_labels}
