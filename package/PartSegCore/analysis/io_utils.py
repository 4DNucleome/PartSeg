import typing
from dataclasses import dataclass, field, replace

import numpy as np
import packaging.version

from PartSegCore.mask_create import MaskProperty
from PartSegCore.project_info import AdditionalLayerDescription, HistoryElement, ProjectInfoBase
from PartSegCore.roi_info import ROIInfo
from PartSegCore.utils import numpy_repr
from PartSegImage import Image

project_version_info = packaging.version.Version("1.1")


@dataclass(frozen=True)
class ProjectTuple(ProjectInfoBase):
    file_path: str
    image: Image
    roi_info: ROIInfo = ROIInfo(None)
    additional_layers: typing.Dict[str, AdditionalLayerDescription] = field(default_factory=dict)
    mask: typing.Optional[np.ndarray] = None
    history: typing.List[HistoryElement] = field(default_factory=list)
    algorithm_parameters: dict = field(default_factory=dict)
    errors: str = ""
    points: typing.Optional[np.ndarray] = None

    def get_raw_copy(self):
        return ProjectTuple(self.file_path, self.image.substitute(mask=None))

    def is_raw(self):
        return self.roi is None

    def replace_(self, **kwargs):
        return replace(self, **kwargs)

    def is_masked(self):
        return self.mask is not None

    def get_raw_mask_copy(self):
        return ProjectTuple(file_path=self.file_path, image=self.image.substitute(), mask=self.mask)

    def __repr__(self):
        return (
            f"ProjectTuple(file_path={self.file_path},\nimage={repr(self.image)},\n"
            f"segmentation={numpy_repr(self.roi)},\nsegmentation_info={repr(self.roi_info)},\n"
            f"additional_layers={repr(self.additional_layers)},\nmask={numpy_repr(self.mask)},\n"
            f"history={repr(self.history)},\nalgorithm_parameters={self.algorithm_parameters},\nerrors={self.errors})"
        )


class MaskInfo(typing.NamedTuple):
    """
    Structure representing mask data

    :param str file_path: path to file with mask
    :param np.ndarray mask_array: numpy array with mask information
    """

    file_path: str
    mask_array: np.ndarray


def create_history_element_from_project(project_info: ProjectTuple, mask_property: MaskProperty):
    return HistoryElement.create(
        roi_info=project_info.roi_info,
        mask=project_info.mask,
        roi_extraction_parameters=project_info.algorithm_parameters,
        mask_property=mask_property,
    )
