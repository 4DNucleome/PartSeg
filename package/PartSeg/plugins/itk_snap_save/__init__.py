from io import BytesIO
from pathlib import Path

import SimpleITK

from PartSegCore.analysis.save_functions import ProjectTuple
from PartSegCore.io_utils import SaveBase


class SaveITKSnap(SaveBase):
    @classmethod
    def get_name(cls):
        return "Mask for itk-snap (*.img)"

    @classmethod
    def get_short_name(cls):
        return "img"

    @classmethod
    def get_fields(cls):
        return []

    @classmethod
    def save(
        cls,
        save_location: str | BytesIO | Path,
        project_info: ProjectTuple,
        parameters: dict,
        range_changed=None,
        step_changed=None,
    ):
        if isinstance(save_location, BytesIO):  # pragma: no cover
            raise NotImplementedError("Cannot save to BytesIO")
        if project_info.roi_info.roi is None:
            raise ValueError("ROI is empty, cannot save")
        mask = SimpleITK.GetImageFromArray(project_info.roi_info.roi)
        SimpleITK.WriteImage(mask, save_location)


def register():
    from PartSegCore.register import RegisterEnum  # noqa: PLC0415
    from PartSegCore.register import register as register_fun  # noqa: PLC0415

    register_fun(SaveITKSnap, RegisterEnum.analysis_save)
