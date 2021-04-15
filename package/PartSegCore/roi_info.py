from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np

from PartSegCore.utils import numpy_repr
from PartSegImage.image import Image, minimal_dtype


class BoundInfo(NamedTuple):
    """
    Information about bounding box
    """

    lower: np.ndarray
    upper: np.ndarray

    def box_size(self) -> np.ndarray:
        """Size of bounding box"""
        return self.upper - self.lower + 1

    def get_slices(self) -> List[slice]:
        return [slice(x, y + 1) for x, y in zip(self.lower, self.upper)]


class ROIInfo:
    """
    Object to storage meta information about given segmentation.
    Segmentation array is only referenced, not copied.

    :ivar numpy.ndarray ~.roi: reference to segmentation
    :ivar Dict[int,BoundInfo] bound_info: mapping from component number to bounding box
    :ivar numpy.ndarray sizes: array with sizes of components
    :ivar Dict[int, Any] annotations: annotations of roi
    :ivar Dict[str, np.ndarray] alternative: alternative representation of roi
    """

    def __init__(
        self,
        roi: Optional[np.ndarray],
        annotations: Optional[Dict[int, Any]] = None,
        alternative: Optional[Dict[str, np.ndarray]] = None,
    ):
        annotations = {} if annotations is None else annotations
        self.annotations = {int(k): v for k, v in annotations.items()}
        self.alternative = {} if alternative is None else alternative
        if roi is None:
            self.roi = None
            self.bound_info = {}
            self.sizes = []
            return
        max_val = np.max(roi)
        dtype = minimal_dtype(max_val)
        roi = roi.astype(dtype)
        self.roi = roi
        self.bound_info = self.calc_bounds(roi)
        self.sizes = np.bincount(roi.flat)

    def fit_to_image(self, image: Image) -> "ROIInfo":
        if self.roi is None:
            return ROIInfo(self.roi, self.annotations, self.alternative)
        roi = image.fit_array_to_image(self.roi)
        alternatives = {k: image.fit_array_to_image(v) for k, v in self.alternative.items()}
        return ROIInfo(roi, self.annotations, alternatives)

    def __str__(self):
        return f"SegmentationInfo; components: {len(self.bound_info)}, sizes: {self.sizes}"

    def __repr__(self):
        return (
            f"SegmentationInfo(segmentation={numpy_repr(self.roi)},"
            f" bound_info={self.bound_info}, sizes={repr(self.sizes)})"
        )

    @staticmethod
    def calc_bounds(roi: np.ndarray) -> Dict[int, BoundInfo]:
        """
        Calculate bounding boxes components

        :param np.ndarray roi: array for which bounds boxes should be calculated
        :return: mapping component number to bounding box
        :rtype: Dict[int, BoundInfo]
        """
        bound_info = {}
        count = np.max(roi)
        for i in range(1, count + 1):
            component = np.array(roi == i)
            if np.any(component):
                points = np.nonzero(component)
                lower = np.min(points, 1)
                upper = np.max(points, 1)
                bound_info[i] = BoundInfo(lower=lower, upper=upper)
        return bound_info
