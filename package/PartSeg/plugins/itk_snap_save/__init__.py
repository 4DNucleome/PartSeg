import typing
from io import BytesIO
from pathlib import Path
import SimpleITK as sitk
from PartSeg.utils.io_utils import SaveBase
from PartSeg.utils.analysis.save_functions import ProjectTuple


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
    def save(cls, save_location: typing.Union[str, BytesIO, Path], project_info: ProjectTuple, parameters: dict,
             range_changed=None, step_changed=None):
        mask = sitk.GetImageFromArray(project_info.segmentation)
        sitk.WriteImage(save_location, mask)


def register():
    from PartSeg.utils.register import register, RegisterEnum
    register(SaveITKSnap, RegisterEnum.analysis_save)
