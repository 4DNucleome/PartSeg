import sys

import numpy as np
from ...utils.global_settings import static_file_folder
from os import path
import typing
from .color_image import color_grayscale

color_maps = np.load(path.join(static_file_folder, "colors.npz"))

def color_chanel(cmap, chanel, max_val, min_val):
    cmap0 = cmap[:, 0]
    cmap1 = cmap[:, 1]
    cmap2 = cmap[:, 2]
    range_val = max_val - min_val
    norm_factor = range_val / 255.0
    temp_image = np.zeros(chanel.shape+(3,), dtype=np.uint8)

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
    color_maps_local = [color_maps[x] if x is not None else None for x in colors]
    new_shape = image.shape[:-1] + (3,)

    result_images = [] # = np.zeros(new_shape, dtype=np.uint8)
    for i, cmap in enumerate(color_maps_local):
        if cmap is None:
            continue
        min_val, max_val = min_max[i] # min_max_calc_int(image[..., i])
        #chanel = (image[..., i] - min_val) / ((max_val - min_val) / 255)
        #chanel = chanel.astype(np.uint8)
        try:
            result_images.append([color_grayscale(cmap, image[..., i], min_val, max_val)])
        except TypeError as e:
            print(image.dtype, file=sys.stderr)
            print(e, file=sys.stderr)
        except IndexError as e:
            raise e
    if len(result_images) > 0:
        if len(result_images) == 1:
            return result_images[0][0]
        else:

            return np.vstack(result_images).max(axis=0)
    else:
         return np.zeros(new_shape, dtype=np.uint8)
