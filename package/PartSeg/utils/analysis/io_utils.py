import typing
import numpy as np

from PartSeg.tiff_image import Image
from PartSeg.utils.io_utils import ProjectInfoBase
from .analysis_utils import HistoryElement


class ProjectTuple(ProjectInfoBase, typing.NamedTuple):
    file_path: str
    image: Image
    segmentation: typing.Optional[np.ndarray] = None
    full_segmentation: typing.Optional[np.ndarray] = None
    mask: typing.Optional[np.ndarray] = None
    history: typing.List[HistoryElement] = []
    algorithm_parameters: dict = {}

    def get_raw_copy(self):
        return ProjectTuple(self.file_path, self.image.substitute())

    def is_raw(self):
        return self.segmentation is None


class MaskInfo(typing.NamedTuple):
    file_path: str
    mask_array: np.ndarray
