from io import BytesIO
from pathlib import Path
from copy import deepcopy
import typing
import os
import numpy as np

from PartSegCore.analysis.io_utils import ProjectTuple
from PartSegCore.analysis.save_functions import SaveCmap, SaveSegmentationAsTIFF, SaveSegmentationAsNumpy
from PartSegCore.channel_class import Channel
from PartSegCore.io_utils import SaveBase
from PartSegCore.algorithm_describe_base import AlgorithmProperty
from PartSegCore.universal_const import Units


class SaveModeling(SaveBase):
    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return [
            AlgorithmProperty("channel", "Channel", 0, property_type=Channel),
            AlgorithmProperty("clip", "Clip area", False),
            AlgorithmProperty(
                "reverse", "Reverse", False, help_text="Reverse brightness off image (for electron microscopy)"
            ),
            AlgorithmProperty("units", "Units", Units.nm, property_type=Units),
        ]

    @classmethod
    def get_name(cls):
        return "Modeling Data"

    @classmethod
    def get_default_extension(cls):
        return ""

    @classmethod
    def get_short_name(cls):
        return "modeling data"

    @classmethod
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info: ProjectTuple,
        parameters: dict,
        range_changed=None,
        step_changed=None,
    ):
        if not os.path.exists(save_location):
            os.makedirs(save_location)
        if not os.path.isdir(save_location):
            raise IOError("save location exist and is not a directory")
        parameters = deepcopy(parameters)
        if parameters["clip"]:
            points = np.nonzero(project_info.segmentation)
            lower_bound = np.min(points, axis=1)
            lower_bound = np.max([lower_bound - 3, [0, 0, 0]], axis=0)
            upper_bound = np.max(points, axis=1)
            upper_bound = np.max([upper_bound + 3, np.array(project_info.segmentation.shape) - 1], axis=0)
            cut_area = tuple([slice(x, y) for x, y in zip(lower_bound, upper_bound)])
            # WARNING time
            image = project_info.image.cut_image((slice(None),) + cut_area)
            segmentation = project_info.segmentation[cut_area]
            full_segmentation = (
                project_info.full_segmentation[cut_area] if project_info.full_segmentation is not None else None
            )
            mask = project_info.mask[cut_area] if project_info.mask else None
            project_info = project_info._replace(
                image=image, segmentation=segmentation, full_segmentation=full_segmentation, mask=mask
            )
            parameters["clip"] = False

        parameters.update({"separated_objects": False})
        SaveCmap.save(
            os.path.join(save_location, "density.cmap"), project_info, parameters, range_changed, step_changed
        )
        parameters.update({"separated_objects": True})
        SaveCmap.save(
            os.path.join(save_location, "density.cmap"), project_info, parameters, range_changed, step_changed
        )
        SaveSegmentationAsTIFF.save(
            os.path.join(save_location, "segmentation.tiff"), project_info, {}, range_changed, step_changed
        )
        SaveSegmentationAsNumpy.save(
            os.path.join(save_location, "segmentation.npy"), project_info, {}, range_changed, step_changed
        )
