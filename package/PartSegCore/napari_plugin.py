import functools
import itertools
import os
import typing

import numpy as np
from napari_plugin_engine import napari_hook_implementation

from .analysis import ProjectTuple
from .analysis.load_functions import load_dict as analysis_load_dict
from .io_utils import LoadBase
from .mask.io_functions import LoadSegmentation, SaveSegmentation, SegmentationTuple
from .mask.io_functions import load_dict as mask_load_dict


def partseg_loader(loader: typing.Type[LoadBase], path: str):
    project_info = loader.load([path])
    if isinstance(project_info, (ProjectTuple, SegmentationTuple)):
        res_layers = []
        if project_info.image is not None and not isinstance(project_info.image, str):
            scale = project_info.image.normalized_scaling()
            for i in range(project_info.image.channels):
                res_layers.append(
                    (
                        project_info.image.get_channel(i),
                        {"scale": scale, "name": f"channel {i}", "blending": "additive"},
                        "image",
                    )
                )
            if project_info.segmentation is not None:
                res_layers.append(
                    (
                        project_info.image.fit_array_to_image(project_info.segmentation),
                        {"scale": scale, "name": "ROI"},
                        "labels",
                    )
                )
            if project_info.mask is not None:
                res_layers.append(
                    (
                        project_info.image.fit_array_to_image(project_info.mask),
                        {"scale": scale, "name": "Mask"},
                        "labels",
                    )
                )
        else:
            if isinstance(project_info, SegmentationTuple) and project_info.spacing is not None:
                scale = np.multiply(project_info.spacing, 10 ** 9)
            else:
                scale = None
            if project_info.segmentation is not None:
                res_layers.append((project_info.segmentation, {"scale": scale, "name": "ROI"}, "labels",))

        return res_layers
    else:
        return None


@napari_hook_implementation
def napari_get_reader(path: str):
    for loader in itertools.chain([LoadSegmentation], analysis_load_dict.values(), mask_load_dict.values()):
        if loader.partial():
            continue
        for extension in loader.get_extensions():
            if path.endswith(extension):
                return functools.partial(partseg_loader, loader)


@napari_hook_implementation
def napari_write_labels(path: str, data: typing.Any, meta: dict) -> typing.Optional[str]:
    if not os.path.splitext(path)[1] == ".seg":
        return
    components = np.unique(data.flat)
    if components[0] == 0:
        components = components[1:]

    st = SegmentationTuple(file_path="", image=None, mask=None, segmentation=data, selected_components=components)
    SaveSegmentation.save(path, st, {})
    return path
