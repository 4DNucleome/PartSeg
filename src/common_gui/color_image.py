import numpy as np
from project_utils.global_settings import static_file_folder
from os import path
import typing
from numba import jit

color_maps = np.load(path.join(static_file_folder, "colors.npz"))

# @jit
def color_image(image: np.ndarray, colors: typing.List[str]):
    color_maps_local = [color_maps[x] for x in colors]
    new_shape = image.shape[:-1] + (3,)
    result_image = np.zeros(new_shape, dtype=np.uint8)
    min_val = image.min(axis=(0,1))
    max_val = image.max(axis=(0,1))
    for i, cmap in enumerate(color_maps_local):
        print(i)
        range_val = max_val[i] - min_val[i]
        norm_factor = range_val/255.0

        @jit
        def _norm_array(x):
            return cmap[x]
        vec_norm_array = np.vectorize(_norm_array, signature='()->(n)')
        temp_image = vec_norm_array((image[..., i] - min_val[i]/norm_factor).astype(np.uint8))
        result_image = np.max(result_image, temp_image )
    return result_image





