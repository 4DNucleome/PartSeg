import typing
from io import BytesIO
from pathlib import Path
import SimpleITK as sitk
from PartSeg.partseg_utils.io_utils import SaveBase
from PartSeg.segmentation_analysis.io_functions import ProjectTuple


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
    def save(cls, save_location: typing.Union[str, BytesIO, Path], project_info: ProjectTuple, parameters: dict):
        mask = sitk.GetImageFromArray(project_info.segmentation)
        sitk.WriteImage(save_location, mask)


def register():
    from PartSeg.segmentation_analysis.save_register import save_register
    save_register.register(SaveITKSnap)
