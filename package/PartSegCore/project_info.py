import sys
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from PartSegCore.roi_info import ROIInfo
from PartSegCore.utils import numpy_repr
from PartSegImage import Image

if sys.version_info.minor < 8:
    from typing_extensions import Protocol, runtime_checkable
else:
    from typing import Protocol, runtime_checkable


@dataclass
class AdditionalLayerDescription:
    """
    Dataclass

    :ivar numpy.ndarray data: layer data
    :ivar str layer_type: napari layer type
    :ivar str name: layer name
    """

    data: np.ndarray
    layer_type: str
    name: str = ""

    def __repr__(self):
        return (
            f"AdditionalLayerDescription(data={numpy_repr(self.data)},"
            f" layer_type='{self.layer_type}', name='{self.name}')"
        )


@runtime_checkable
class ProjectInfoBase(Protocol):
    """
    This is base protocol for Project Information.

    :ivar str ~.file_path: path to current preceded file
    :ivar Image ~.image: project image
    :ivar numpy.ndarray ~.segmentation: numpy array representing current project ROI
    :ivar ROIInfo ~.roi_info: segmentation metadata
    :ivar Optional[numpy.ndarray] ~.mask: mask used in project
    :ivar str errors: information about problems with current project
    """

    file_path: str
    image: Image
    roi_info: ROIInfo = ROIInfo(None)
    additional_layers: Dict[str, AdditionalLayerDescription] = {}
    mask: Optional[np.ndarray] = None
    errors: str = ""
    points: Optional[np.ndarray] = None

    @property
    def roi(self):
        warnings.warn("roi is deprecated", DeprecationWarning, 2)
        return self.roi_info.roi

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
