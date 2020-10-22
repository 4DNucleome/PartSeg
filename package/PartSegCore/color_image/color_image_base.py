import typing

import numpy as np

import PartSegData

from .base_colors import BaseColormap
from .color_image_cython import resolution

color_maps = np.load(PartSegData.colors_file)

color_array_dict: typing.Dict[BaseColormap, np.ndarray] = {}
# TODO maybe replace with better cache structure


def create_color_map(colormap_definition: BaseColormap, power: float = 1.0) -> np.ndarray:
    """
    Calculate array with approximation of colormap used by :py:func`.color_image_fun` function.
    If first or last color do not have position 0 or 1 respectively then begin or end will be filled with given color

    Greyscale colormap
    ``res = create_color_map([ColorPosition(0, Color(0, 0, 0)), ColorPosition(1, Color(255, 255, 255))])``

    Black in first 25% of colormap then Greyscale from 25% to 75% and then white from 75% of colormap
    ``res = create_color_map([ColorPosition(0.25, Color(0, 0, 0)), ColorPosition(0.75, Color(255, 255, 255))])``

    :param colormap_definition: list defining base colormap
    :param power: power normalization of colormap. Will be ignored if points_and_colors.points be non empty
    :return: array of size (1024, 3)
    """
    # TODO Rethink
    colormap = np.zeros((1024, 3), dtype=np.uint8)
    bounds = colormap_definition.bounds()
    values = colormap_definition.color_values()
    if len(bounds) == 0:
        return colormap
    if len(bounds) == 1:
        colormap[:] = values[0]
        return colormap
    _bounds = [x ** power * ((resolution - 1) / resolution) for x in bounds]
    bounds = [_bounds[0]]
    _values = values
    values = [_values[0]]
    if len(bounds) < 10:
        for i, (x, y) in enumerate(zip(_bounds, _bounds[1:]), start=1):
            dist = (y - x) * resolution / 256
            if dist > 1:
                bounds.append(y - dist / resolution)
                values.append(_values[i])
            bounds.append(y)
            values.append(_values[i])
    values = np.array(values)
    points = list(colormap_definition.get_points())
    if len(points) == 0:
        points = np.linspace(0, 1, resolution, endpoint=False)
    for i in range(3):
        colormap[:, i] = np.interp(points, bounds, values[:, i])
    return colormap


def color_bar_fun(bar: np.ndarray, colormap: typing.Union[BaseColormap, np.ndarray]):
    if isinstance(colormap, BaseColormap):
        if colormap not in color_array_dict:
            color_array_dict[colormap] = create_color_map(colormap)
        colormap = color_array_dict[colormap]
    if not isinstance(colormap, np.ndarray) or not colormap.shape == (resolution, 3):
        raise ValueError(f"colormap should be passed as numpy array with shape ({resolution},3)")
    min_val = np.min(bar)
    cords = ((bar - min_val) * ((colormap.shape[0] - 1) / (np.max(bar) - min_val))).astype(np.uint16)
    return colormap[cords]
