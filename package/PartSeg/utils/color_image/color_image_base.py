import typing

import PartSegData
import numpy as np

from .color_image import color_grayscale
from concurrent.futures import ThreadPoolExecutor, as_completed

color_maps = np.load(PartSegData.colors_file)


def color_chanel(cmap, chanel, max_val, min_val):
    cmap0 = cmap[:, 0]
    cmap1 = cmap[:, 1]
    cmap2 = cmap[:, 2]
    range_val = max_val - min_val
    norm_factor = range_val / 255.0
    temp_image = np.zeros(chanel.shape + (3,), dtype=np.uint8)

    def _norm_array0(x):
        return cmap0[x]

    def _norm_array1(x):
        return cmap1[x]

    def _norm_array2(x):
        return cmap2[x]

    vec_norm_array0 = np.vectorize(_norm_array0, otypes=[np.uint8])
    vec_norm_array1 = np.vectorize(_norm_array1, otypes=[np.uint8])
    vec_norm_array2 = np.vectorize(_norm_array2, otypes=[np.uint8])
    normed_image = (chanel - min_val / norm_factor).astype(np.uint8)
    temp_image[..., 0] = vec_norm_array0(normed_image)
    temp_image[..., 1] = vec_norm_array1(normed_image)
    temp_image[..., 2] = vec_norm_array2(normed_image)
    return temp_image


def color_image(image: np.ndarray, colors: typing.List[str], min_max: typing.List[typing.Tuple]) -> np.ndarray:
    color_maps_local = [color_maps[x] if isinstance(x, str) else x for x in colors]
    new_shape = image.shape[:-1] + (3,)

    result_images = []  # = np.zeros(new_shape, dtype=np.uint8)
    colored_channels = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i, cmap in enumerate(color_maps_local):
            if cmap is None:
                continue
            assert isinstance(cmap, np.ndarray) and cmap.shape == (1024, 3)
            min_val, max_val = min_max[i]

            colored_channels[executor.submit(color_grayscale, cmap, image[..., i], min_val, max_val)] = i
    for res in as_completed(colored_channels):
        result_images.append(res.result())
    if len(result_images) > 0:
        if len(result_images) == 1:
            return result_images[0]
        else:

            return np.max(result_images, axis=0)
    else:
        return np.zeros(new_shape, dtype=np.uint8)
