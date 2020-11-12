import sys
from typing import Optional

import numpy as np

from PartSegCore.roi_info import ROIInfo
from PartSegImage import Image

if sys.version_info.minor < 8:
    from typing_extensions import Protocol, runtime_checkable
else:
    from typing import Protocol, runtime_checkable


@runtime_checkable
class ProjectInfoBase(Protocol):
    """
    This is base protocol for Project Information.

    :ivar str ~.file_path: path to current preceded file
    :ivar Image ~.image: project image
    :ivar numpy.ndarray ~.segmentation: numpy array representing current project ROI
    :ivar SegmentationInfo ~.segmentation_info: segmentation metadata
    :ivar Optional[numpy.ndarray] ~.mask: mask used in project
    :ivar str errors: information about problems with current project
    """

    file_path: str
    image: Image
    roi: np.ndarray
    roi_info: ROIInfo = ROIInfo(None)
    mask: Optional[np.ndarray]
    errors: str = ""

    def get_raw_copy(self):
        """
        Create copy with only image
        """
        raise NotImplementedError

    def get_raw_mask_copy(self):
        raise NotImplementedError

    def is_raw(self):
        raise NotImplementedError

    def is_masked(self):
        raise NotImplementedError
