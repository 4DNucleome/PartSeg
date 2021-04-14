import typing

import numpy as np

from PartSegCore.analysis import ProjectTuple
from PartSegCore.io_utils import LoadBase, WrongFileTypeException
from PartSegCore.mask.io_functions import MaskProjectTuple


def project_to_layers(project_info: typing.Union[ProjectTuple, MaskProjectTuple]):
    res_layers = []
    if project_info.image is not None and not isinstance(project_info.image, str):
        scale = project_info.image.normalized_scaling()
        for i in range(project_info.image.channels):
            res_layers.append(
                (
                    project_info.image.get_channel(i),
                    {"scale": scale, "name": project_info.image.channel_names[i], "blending": "additive"},
                    "image",
                )
            )
        if project_info.roi_info.roi is not None:
            res_layers.append(
                (
                    project_info.image.fit_array_to_image(project_info.roi_info.roi),
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
        if isinstance(project_info, MaskProjectTuple) and project_info.spacing is not None:
            scale = np.multiply(project_info.spacing, 10 ** 9)
        else:
            scale = None
        if project_info.roi_info.roi is not None:
            res_layers.append(
                (
                    project_info.roi_info.roi,
                    {"scale": scale, "name": "ROI"},
                    "labels",
                )
            )
    return res_layers


def partseg_loader(loader: typing.Type[LoadBase], path: str):
    load_locations = [path]
    for _ in range(1, loader.number_of_files()):
        load_locations.append(loader.get_next_file(load_locations))
    try:
        project_info = loader.load(load_locations)
    except WrongFileTypeException:
        return None

    if isinstance(project_info, (ProjectTuple, MaskProjectTuple)):
        return project_to_layers(project_info)
    return None
