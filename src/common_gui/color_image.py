import numpy as np
from project_utils.global_settings import file_folder
from os import path
import typing
color_maps = np.load(path.join(file_folder, "static_files", "colors.npz"))

def color_image(image: np.ndarray, colors: typing.List[str]):
    color_maps_local = [color_maps[x] for x in colors]
    new_shape = image.shape[:-1] + (3,)
    result_image = np.zeros(new_shape, dtype=np.uint8)
    for i, cmap in enumerate(color_maps_local):
        min_val = result_image.min()
        max_val = result_image.max()
        range_val = max_val - min_val
        norm_factor = range_val/256
        def _norm_array(x):
            return cmap[int((x-min_val)/norm_factor)]
        vec_norm_array = np.vectorize(_norm_array, otypes=[np.uint8, np.uint8, np.uint8])
        result_image = np.max(result_image, vec_norm_array(image[..., i] - min_val))
    return result_image





