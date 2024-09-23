import typing

import numpy as np

from PartSegCore.analysis import ProjectTuple
from PartSegCore.io_utils import LoadBase, WrongFileTypeException
from PartSegCore.mask.io_functions import MaskProjectTuple


def _image_to_layers(project_info, scale, translate):
    res_layers = []
    if project_info.image.name == "ROI" and project_info.image.channels == 1:
        res_layers.append(
            (
                project_info.image.get_channel(0),
                {"scale": scale, "name": project_info.image.channel_names[0], "translate": translate},
                "labels",
            )
        )
    else:
        res_layers.extend(
            (
                project_info.image.get_channel(i),
                {
                    "scale": scale,
                    "name": project_info.image.channel_names[i],
                    "blending": "additive",
                    "translate": translate,
                    "metadata": project_info.image.metadata,
                },
                "image",
            )
            for i in range(project_info.image.channels)
        )

    return res_layers


def project_to_layers(project_info: typing.Union[ProjectTuple, MaskProjectTuple]):
    res_layers = []
    if project_info.image is not None and not isinstance(project_info.image, str):
        scale = project_info.image.normalized_scaling()
        translate = project_info.image.shift
        translate = (0,) * (len(project_info.image.axis_order.replace("C", "")) - len(translate)) + translate
        res_layers.extend(_image_to_layers(project_info, scale, translate))
        if project_info.roi_info.roi is not None:
            res_layers.append(
                (
                    project_info.image.fit_array_to_image(project_info.roi_info.roi),
                    {"scale": scale, "name": "ROI", "translate": translate},
                    "labels",
                )
            )
        if project_info.roi_info.alternative:
            res_layers.extend(
                (
                    project_info.image.fit_array_to_image(roi),
                    {
                        "scale": scale,
                        "name": name,
                        "translate": translate,
                        "visible": False,
                    },
                    "labels",
                )
                for name, roi in project_info.roi_info.alternative.items()
            )

        if project_info.mask is not None:
            res_layers.append(
                (
                    project_info.image.fit_array_to_image(project_info.mask),
                    {"scale": scale, "name": "Mask", "translate": translate},
                    "labels",
                )
            )
    else:
        if isinstance(project_info, MaskProjectTuple) and project_info.spacing is not None:
            scale = np.multiply(project_info.spacing, 10**9)
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
    load_locations.extend(loader.get_next_file(load_locations) for _ in range(1, loader.number_of_files()))

    try:
        project_info = loader.load(load_locations)
    except WrongFileTypeException:  # pragma: no cover
        return None

    if isinstance(project_info, (ProjectTuple, MaskProjectTuple)):
        return project_to_layers(project_info)
    return None  # pragma: no cover
