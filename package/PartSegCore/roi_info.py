from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np

from PartSegCore.utils import numpy_repr
from PartSegCore_compiled_backend.utils import calc_bounds
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

    def get_slices(self, margin=0) -> List[slice]:
        return [slice(max(x - margin, 0), y + 1 + margin) for x, y in zip(self.lower, self.upper)]

    def del_dim(self, axis: int):
        return BoundInfo(np.delete(self.lower, axis), np.delete(self.upper, axis))

    def __str__(self):
        return f"{self.__class__.__name__}(lower={list(self.lower)}, upper={list(self.upper)})"


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
        return f"ROIInfo; components: {len(self.bound_info)}, sizes: {self.sizes}"

    def __repr__(self):
        return f"ROIInfo(roi={numpy_repr(self.roi)}, bound_info={self.bound_info}, sizes={repr(self.sizes)})"

    @staticmethod
    def calc_bounds(roi: np.ndarray) -> Dict[int, BoundInfo]:
        """
        Calculate bounding boxes components

        :param np.ndarray roi: array for which bounds boxes should be calculated
        :return: mapping component number to bounding box
        :rtype: Dict[int, BoundInfo]
        """
        try:
            min_bounds, max_bounds = calc_bounds(roi)
            return {
                num: BoundInfo(lower=lower, upper=upper)
                for num, (lower, upper) in enumerate(zip(min_bounds, max_bounds))
                if num != 0 and upper[0] != -1
            }
        except KeyError:
            bound_info = {}
            points = np.nonzero(roi)
            comp_num = roi[points]
            point_dict = defaultdict(list)
            for num, point in zip(comp_num, np.transpose(points)):
                point_dict[num].append(point)
            for num, points_for_num in point_dict.items():
                lower = np.min(points_for_num, 0)
                upper = np.max(points_for_num, 0)
                bound_info[num] = BoundInfo(lower=lower, upper=upper)
            return bound_info
