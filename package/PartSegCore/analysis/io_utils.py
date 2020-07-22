import typing
from dataclasses import dataclass, field, replace

import numpy as np
import packaging.version

from PartSegCore.io_utils import HistoryElement
from PartSegCore.mask_create import MaskProperty
from PartSegCore.project_info import ProjectInfoBase
from PartSegCore.segmentation.algorithm_base import AdditionalLayerDescription
from PartSegCore.segmentation_info import SegmentationInfo
from PartSegImage import Image

project_version_info = packaging.version.Version("1.1")


@dataclass(frozen=True)
class ProjectTuple(ProjectInfoBase):
    file_path: str
    image: Image
    segmentation: typing.Optional[np.ndarray] = None
    segmentation_info: SegmentationInfo = SegmentationInfo(None)
    additional_layers: typing.Dict[str, AdditionalLayerDescription] = field(default_factory=dict)
    mask: typing.Optional[np.ndarray] = None
    history: typing.List[HistoryElement] = field(default_factory=list)
    algorithm_parameters: dict = field(default_factory=dict)
    errors: str = ""

    def __post_init__(self):
        if self.segmentation_info.segmentation is not None:
            object.__setattr__(self, "segmentation_info", SegmentationInfo(self.segmentation))

    def get_raw_copy(self):
        return ProjectTuple(self.file_path, self.image.substitute(mask=None))

    def is_raw(self):
        return self.segmentation is None

    def replace_(self, **kwargs):
        return replace(self, **kwargs)

    def is_masked(self):
        return self.mask is not None

    def get_raw_mask_copy(self):
        return ProjectTuple(file_path=self.file_path, image=self.image.substitute(), mask=self.mask)


class MaskInfo(typing.NamedTuple):
    file_path: str
    mask_array: np.ndarray


def create_history_element_from_project(project_info: ProjectTuple, mask_property: MaskProperty):
    return HistoryElement.create(
        segmentation=project_info.segmentation,
        mask=project_info.mask,
        segmentation_parameters=project_info.algorithm_parameters,
        mask_property=mask_property,
    )
