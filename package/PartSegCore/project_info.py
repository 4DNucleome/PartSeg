import sys
import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from PartSegCore.class_generator import BaseSerializableClass
from PartSegCore.mask_create import MaskProperty, calculate_mask
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


class HistoryElement(BaseSerializableClass):
    roi_extraction_parameters: Dict[str, Any]
    annotations: Optional[Dict[int, Any]]
    mask_property: MaskProperty
    arrays: BytesIO

    @classmethod
    def create(
        cls,
        roi_info: ROIInfo,
        mask: Union[np.ndarray, None],
        roi_extraction_parameters: dict,
        mask_property: MaskProperty,
    ):
        if "name" in roi_extraction_parameters:  # pragma: no cover
            raise ValueError("name")
        arrays = BytesIO()
        arrays_dict = {"roi": roi_info.roi}
        for name, array in roi_info.alternative.items():
            arrays_dict[name] = array
        if mask is not None:
            arrays_dict["mask"] = mask

        np.savez_compressed(arrays, **arrays_dict)
        arrays.seek(0)
        return cls(
            roi_extraction_parameters=roi_extraction_parameters,
            mask_property=mask_property,
            arrays=arrays,
            annotations=roi_info.annotations,
        )

    def get_roi_info_and_mask(self) -> Tuple[ROIInfo, Optional[np.ndarray]]:
        self.arrays.seek(0)
        seg = np.load(self.arrays)
        self.arrays.seek(0)
        alternative = {name: array for name, array in seg.items() if name not in {"roi", "mask"}}
        roi_info = ROIInfo(seg["roi"], annotations=self.annotations, alternative=alternative)
        mask = seg["mask"] if "mask" in seg else None
        return roi_info, mask


@runtime_checkable
class ProjectInfoBase(Protocol):
    """
    This is base protocol for Project Information.

    :ivar str ~.file_path: path to current preceded file
    :ivar Image ~.image: project image
    :ivar numpy.ndarray ~.segmentation: numpy array representing current project ROI
    :ivar ROIInfo ~.roi_info: segmentation metadata
    :ivar Optional[numpy.ndarray] ~.mask: mask used in project
    :ivar List[HistoryElement] ~.history: history of calculation
    :ivar str errors: information about problems with current project
    """

    file_path: str
    image: Image
    roi_info: ROIInfo = ROIInfo(None)
    additional_layers: Dict[str, AdditionalLayerDescription] = {}
    mask: Optional[np.ndarray] = None
    history: List[HistoryElement] = []
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


def calculate_mask_from_project(
    mask_description: MaskProperty, project: ProjectInfoBase, components: Optional[List[int]] = None
) -> np.ndarray:
    """
    Function for calculate mask base on MaskProperty.
    This function calls :py:func:`calculate_mask` with arguments from project.

    :param MaskProperty mask_description: information how calculate mask
    :param ProjectInfoBase project: project with information about segmentation
    :param Optional[List[int]] components: If present inform which components
        should be used when calculation mask, otherwise use all.
    :return: new mask
    :rtype: np.ndarray
    """
    try:
        time_axis = project.image.time_pos
    except AttributeError:
        time_axis = None
    return calculate_mask(
        mask_description, project.roi_info.roi, project.mask, project.image.spacing, components, time_axis
    )


class HistoryProblem(Exception):
    pass
