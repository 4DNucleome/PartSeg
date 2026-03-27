import os
import typing
from importlib.metadata import version

import numpy as np
from packaging.version import parse as parse_version

from PartSeg.plugins.napari_widgets._settings import get_settings
from PartSegCore import UNIT_SCALE
from PartSegCore.analysis import ProjectTuple
from PartSegCore.io_utils import LoadBase, WrongFileTypeException
from PartSegCore.mask.io_functions import MaskProjectTuple
from PartSegCore.universal_const import format_layer_name
from PartSegImage import Image


@typing.overload
def adjust_color(color: str) -> str: ...


@typing.overload
def adjust_color(color: list[int]) -> list[float]: ...


def adjust_color(color: typing.Union[str, list[int]]) -> typing.Union[str, tuple[float]]:
    # as napari ignore alpha channel in color, and adding it to
    # color cause that napari fails to detect that such colormap is already present
    # in this function I remove alpha channel if it is present
    if isinstance(color, str) and color.startswith("#"):
        if len(color) == 9:
            # case when color is in format #RRGGBBAA
            return color[:7]
        if len(color) == 5:
            # case when color is in format #RGBA
            return color[:4]
    elif isinstance(color, list):
        return (color[i] / 255 for i in range(3))
    # If not fit to an earlier case, return as is.
    # Maybe napari will handle it
    return color


if parse_version(version("napari")) >= parse_version("0.4.19a1"):

    def add_color(image: Image, idx: int) -> dict:
        return {
            "colormap": adjust_color(image.get_colors()[idx]),
        }

else:

    def add_color(image: Image, idx: int) -> dict:  # noqa: ARG001
        # Do nothing, as napari is not able to pass hex color to image
        # the image and idx are present to keep the same signature
        return {}


def _image_to_layers(project_info, scale, translate):
    settings = get_settings()
    filename = os.path.basename(project_info.file_path)
    res_layers = []
    if project_info.image.name == "ROI" and project_info.image.channels == 1:
        res_layers.append(
            (
                project_info.image.get_channel(0),
                {
                    "scale": scale,
                    "name": format_layer_name(
                        settings.layer_naming_format, filename, project_info.image.channel_names[0]
                    ),
                    "translate": translate,
                },
                "labels",
            )
        )
    else:
        res_layers.extend(
            (
                project_info.image.get_channel(i),
                {
                    "scale": scale,
                    "name": format_layer_name(
                        settings.layer_naming_format, filename, project_info.image.channel_names[i]
                    ),
                    "blending": "additive",
                    "translate": translate,
                    "metadata": project_info.image.metadata,
                    **add_color(project_info.image, i),
                },
                "image",
            )
            for i in range(project_info.image.channels)
        )

    return res_layers


def project_to_layers(project_info: typing.Union[ProjectTuple, MaskProjectTuple]):
    res_layers = []
    if project_info.image is not None and not isinstance(project_info.image, str):
        settings = get_settings()
        scale = project_info.image.normalized_scaling(UNIT_SCALE[settings.io_units.value])
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


def partseg_loader(loader: type[LoadBase], path: str):
    load_locations = [path]
    load_locations.extend(loader.get_next_file(load_locations) for _ in range(1, loader.number_of_files()))

    try:
        project_info = loader.load(load_locations)
    except WrongFileTypeException:  # pragma: no cover
        return None

    if isinstance(project_info, (ProjectTuple, MaskProjectTuple)):
        return project_to_layers(project_info)
    return None  # pragma: no cover
