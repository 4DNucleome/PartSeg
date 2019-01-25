import typing
import numpy as np

from PartSeg.tiff_image import Image
from .analysis_utils import HistoryElement


class ProjectTuple(typing.NamedTuple):
    file_path: str
    image: Image
    segmentation: typing.Optional[np.ndarray] = None
    full_segmentation: typing.Optional[np.ndarray] = None
    mask: typing.Optional[np.ndarray] = None
    history: typing.List[HistoryElement] = []
    algorithm_parameters: dict = {}


class MaskInfo(typing.NamedTuple):
    file_path: str
    mask_array: np.ndarray