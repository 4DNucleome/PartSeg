import typing
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from PartSegCore_compiled_backend.color_image_cython import color_grayscale, resolution

from .base_colors import BaseColormap

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
    if not points:
        points = np.linspace(0, 1, resolution, endpoint=False)
    for i in range(3):
        colormap[:, i] = np.interp(points, bounds, values[:, i])
    return colormap


def color_bar_fun(bar: np.ndarray, colormap: typing.Union[BaseColormap, np.ndarray]):
    if isinstance(colormap, BaseColormap):
        if colormap not in color_array_dict:
            color_array_dict[colormap] = create_color_map(colormap)
        colormap = color_array_dict[colormap]
    if not isinstance(colormap, np.ndarray) or colormap.shape != (
        resolution,
        3,
    ):
        raise ValueError(f"colormap should be passed as numpy array with shape ({resolution},3)")
    min_val = np.min(bar)
    cords = ((bar - min_val) * ((colormap.shape[0] - 1) / (np.max(bar) - min_val))).astype(np.uint16)
    return colormap[cords]


def color_image_fun(
    image: np.ndarray,
    colors: typing.List[typing.Union[BaseColormap, np.ndarray]],
    min_max: typing.List[typing.Tuple[float, float]],
) -> np.ndarray:
    """
    Color given image layer.
    :param image: Single image layer (array of size (width, height, channels)
    :param colors: list of colormaps by name (from ``.color_maps`` dict) or array (its size ned to be size (1024, 3))
      array can be created by :py:func:`.create_color_map`
    :param min_max: bounds for each channel separately
    :return: colored image (array of size (width, height, 3) as RGB image
    """
    new_shape = image.shape[:-1] + (3,)

    result_images = []  # = np.zeros(new_shape, dtype=np.uint8)
    colored_channels = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i, colormap in enumerate(colors):
            if colormap is None:
                continue
            if isinstance(colormap, BaseColormap):
                if colormap not in color_array_dict:
                    color_array_dict[colormap] = create_color_map(colormap)
                colormap = color_array_dict[colormap]
            if not isinstance(colormap, np.ndarray) or not colormap.shape == (resolution, 3):
                raise ValueError(f"colormap should be passed as numpy array with shape ({resolution},3)")
            min_val, max_val = min_max[i]

            colored_channels[executor.submit(color_grayscale, colormap, image[..., i], min_val, max_val)] = i
    for res in as_completed(colored_channels):
        result_images.append(res.result())
    if len(result_images) > 0:
        if len(result_images) == 1:
            return result_images[0]
        # TODO use ColorMap additional information
        return np.max(result_images, axis=0)
    return np.zeros(new_shape, dtype=np.uint8)
