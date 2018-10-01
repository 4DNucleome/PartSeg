from collections import defaultdict
from enum import Enum
from functools import reduce

from PyQt5.QtCore import pyqtSignal

from project_utils.algorithm_base import SegmentationAlgorithm
from project_utils.distance_in_structure.find_split import distance_sprawl, path_minimum_sprawl, path_maximum_sprawl
from project_utils.image_operations import gaussian
import numpy as np
import SimpleITK as sitk
from project_utils import bisect
import operator
from project_utils.universal_const import UNIT_SCALE

def blank_operator(x, y):
    raise NotImplemented()


class RestartableAlgorithm(SegmentationAlgorithm):
    execution_done_extend = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()
        self.parameters = defaultdict(lambda: None)
        self.new_parameters = {}
        self.spacing = None
        self.use_psychical_unit = False

    def set_image(self, image):
        self.image = image
        self.parameters.clear()

    def set_size_information(self, spacing, use_physicla_unit):
        self.spacing = spacing
        self.use_psychical_unit = use_physicla_unit

    def get_info_text(self):
        return "No info [Report this ass error]"




class ThresholdBaseAlgorithm(RestartableAlgorithm):
    """
    :type segmentation: np.ndarray
    """

    threshold_operator = blank_operator

    def __init__(self):
        super(ThresholdBaseAlgorithm, self).__init__()
        self.mask = None
        self.gauss_image = None
        self.threshold_image = None
        self._sizes_array = []
        self.components_num = 0

    def get_info_text(self):
        return ", ".join(map(str, self._sizes_array[1:self.components_num+1]))

    def run(self):
        finally_segment = self.calculation_run()
        if finally_segment is not None:
            self.execution_done.emit(finally_segment)
            self.execution_done_extend.emit(finally_segment, self.segmentation)
            self.parameters.update(self.new_parameters)

    def calculation_run(self):
        restarted = False
        if self.new_parameters["use_gauss"]:
            if self.parameters["gauss_radius"] != self.new_parameters["gauss_radius"] or \
                    self.new_parameters["use_gauss"] != self.parameters["use_gauss"]:
                self.gauss_image = gaussian(self.image, self.new_parameters["gauss_radius"])
                restarted = True
        elif self.new_parameters["use_gauss"] != self.parameters["use_gauss"]:
            self.gauss_image = self.image
            restarted = True
        if restarted or self.new_parameters["threshold"] != self.parameters["threshold"]:
            threshold_image = self._threshold(self.gauss_image)
            if self.mask is not None:
                threshold_image *= (self.mask > 0)
            connect = sitk.ConnectedComponent(sitk.GetImageFromArray(threshold_image), True)
            self.segmentation = sitk.GetArrayFromImage(sitk.RelabelComponent(connect))
            self._sizes_array = np.bincount(self.segmentation.flat)
            restarted = True
        if restarted or self.new_parameters["minimum_size"] != self.parameters["minimum_size"]:
            minimum_size =  self.new_parameters["minimum_size"]
            if self.use_psychical_unit:
                minimum_size /= reduce((lambda x, y: x * y), self.spacing)
            ind = bisect(self._sizes_array[1:], minimum_size, lambda x, y: x > y)
            finally_segment = np.copy(self.segmentation)
            finally_segment[finally_segment > ind] = 0
            self.components_num = ind
            return finally_segment

    def _clean(self):
        super()._clean()
        self.parameters = defaultdict(lambda: None)
        self.gauss_image = None
        self.mask = None

    def _threshold(self, image, thr=None):
        if thr is None:
            thr = self.new_parameters["threshold"]
        return self.threshold_operator(image, thr).astype(np.uint8)

    def set_image(self, image):
        self.image = image
        self.parameters["gauss_radius"] = None
        self.parameters["use_gauss"] = None

    def set_mask(self, mask):
        self.mask = mask
        self.new_parameters["threshold"] = self.parameters["threshold"]
        self.parameters["threshold"] = None


class OneThresholdAlgorithm(ThresholdBaseAlgorithm):
    def set_parameters(self, threshold, minimum_size, use_gauss, gauss_radius):
        self.new_parameters["threshold"] = threshold
        self.new_parameters["minimum_size"] = minimum_size
        self.new_parameters["use_gauss"] = use_gauss
        self.new_parameters["gauss_radius"] = gauss_radius


class LowerThresholdAlgorithm(OneThresholdAlgorithm):
    threshold_operator = operator.gt
    """def _threshold(self, image, thr=None):
        return (image > self.new_parameters["threshold"]).astype(np.uint8)"""


class UpperThresholdAlgorithm(OneThresholdAlgorithm):
    threshold_operator = operator.lt
    """def _threshold(self, image, thr=None):
        return (image < self.new_parameters["threshold"]).astype(np.uint8)"""


class RangeThresholdAlgorithm(ThresholdBaseAlgorithm):
    def set_parameters(self, lower_threshold, upper_threshold, minimum_size, use_gauss, gauss_radius):
        self.new_parameters["threshold"] = lower_threshold, upper_threshold
        self.new_parameters["minimum_size"] = minimum_size
        self.new_parameters["use_gauss"] = use_gauss
        self.new_parameters["gauss_radius"] = gauss_radius

    def _threshold(self, image, thr=None):
        return ((image > self.new_parameters["threshold"][0]) * (image < self.new_parameters["threshold"][1])).astype(
            np.uint8)


class BaseThresholdFlowAlgorithm(ThresholdBaseAlgorithm):
    def path_sprawl(self, base_image, object_image) -> np.ndarray:
        raise NotImplementedError()

    def get_info_text(self):
        return "Mid sizes: "  + ", ".join(map(str, self._sizes_array[1:self.components_num+1])) + \
               "\nFinal sizes: " +  ", ".join(map(str, self.final_sizes[1:]))

    def __init__(self):
        super().__init__()
        self.finally_segment = None
        self.final_sizes = []

    def set_parameters(self, threshold, minimum_size, use_gauss, gauss_radius, base_threshold):
        self.new_parameters["threshold"] = threshold
        self.new_parameters["minimum_size"] = minimum_size
        self.new_parameters["use_gauss"] = use_gauss
        self.new_parameters["gauss_radius"] = gauss_radius
        self.new_parameters["base_threshold"] = base_threshold

    def run(self):
        finally_segment = self.calculation_run()
        if finally_segment is not None and self.components_num == 0:
            self.final_sizes = []
            self.execution_done.emit(finally_segment)
            self.execution_done_extend.emit(finally_segment, self.segmentation)
            self.parameters.update(self.new_parameters)
            return

        if finally_segment is None:
            restarted = False
            finally_segment =np.copy(self.finally_segment)
        else:
            self.finally_segment = finally_segment
            restarted = True

        if restarted or self.new_parameters["base_threshold"] != self.parameters["base_threshold"]:
            if self.threshold_operator(self.new_parameters["base_threshold"], self.new_parameters["threshold"]):
                print("buka1")
            else:
                print("buka2")
            threshold_image = self._threshold(self.gauss_image, self.new_parameters["base_threshold"])
            print(f"Sizes {np.count_nonzero(threshold_image)}, {np.count_nonzero(finally_segment)}", self.__class__)
            if self.mask is not None:
                print("maskkkk")
                threshold_image *= (self.mask > 0)
            new_segment = self.path_sprawl(threshold_image, finally_segment)
            self.final_sizes = np.bincount(new_segment.flat)
            self.execution_done.emit(new_segment)
            self.execution_done_extend.emit(new_segment, threshold_image)
            self.parameters.update(self.new_parameters)


class LowerThresholdFlowAlgorithm(BaseThresholdFlowAlgorithm):
    threshold_operator = operator.gt
    """def _threshold(self, image, thr=None):
        if thr is None:
            thr = self.new_parameters["threshold"]
        return (image > thr).astype(np.uint8)"""


class UpperThresholdFlowAlgorithm(BaseThresholdFlowAlgorithm):
    threshold_operator = operator.lt
    """def _threshold(self, image, thr=None):
        if thr is None:
            thr = self.new_parameters["threshold"]
        return (image < thr).astype(np.uint8)"""


class LowerThresholdDistanceFlowAlgorithm(LowerThresholdFlowAlgorithm):
    def path_sprawl(self, base_image, object_image):
        neigh, dist = calculate_distances_array(self.spacing, NeighType.edges)
        return distance_sprawl(base_image, object_image, self.components_num, neigh, dist)


class UpperThresholdDistanceFlowAlgorithm(UpperThresholdFlowAlgorithm):
    def path_sprawl(self, base_image, object_image):
        neigh, dist = calculate_distances_array(self.spacing, NeighType.edges)
        return distance_sprawl(base_image, object_image, self.components_num, neigh, dist)


class LowerThresholdPathFlowAlgorithm(LowerThresholdFlowAlgorithm):
    def path_sprawl(self, base_image, object_image):
        image = self.image.astype(np.float64)
        image[base_image == 0] = 0
        mid = path_maximum_sprawl(image, object_image, self.components_num)
        return path_maximum_sprawl(image, mid, self.components_num)


class UpperThresholdPathFlowAlgorithm(UpperThresholdFlowAlgorithm):
    def path_sprawl(self, base_image, object_image):
        image = self.image.astype(np.float64)
        image[base_image == 0] = 0
        mid = path_minimum_sprawl(image, object_image, self.components_num)
        return path_minimum_sprawl(image, mid, self.components_num)


class LowerThresholdPathDistanceFlowAlgorithm(LowerThresholdPathFlowAlgorithm):
    def path_sprawl(self, base_image, object_image):
        mid = super().path_sprawl(base_image, object_image)
        neigh, dist = calculate_distances_array(self.spacing, NeighType.edges)
        return distance_sprawl(base_image, mid, self.components_num, neigh, dist)


class UpperThresholdPathDistanceFlowAlgorithm(UpperThresholdPathFlowAlgorithm):
    def path_sprawl(self, base_image, object_image):
        mid = super().path_sprawl(base_image, object_image)
        neigh, dist = calculate_distances_array(self.spacing, NeighType.edges)
        return distance_sprawl(base_image, mid, self.components_num, neigh, dist)

class NeighType(Enum):
    sides = 6
    edges = 18
    vertex = 26


def calculate_distances_array(spacing, neigh_type: NeighType):
    min_dist = min(spacing)
    normalized_spacing = [x/min_dist for x in spacing]
    if len(normalized_spacing) == 2:
        neighbourhood_array = neighbourhood2d
        if neigh_type == NeighType.sides:
            neighbourhood_array = neighbourhood_array[:4]
        normalized_spacing = [0] + normalized_spacing
    else:
        neighbourhood_array = neighbourhood[:neigh_type.value]
    normalized_spacing = np.array(normalized_spacing)
    return neighbourhood_array, np.sqrt(np.sum((neighbourhood_array*normalized_spacing)**2, axis=1))




neighbourhood = np.array([[0,-1, 0], [0, 0,-1],
    [0, 1, 0], [0, 0, 1],
    [-1, 0, 0], [1, 0, 0],

    [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 1, 0],
    [-1, 0, -1], [1, 0, -1], [-1, 0, 1], [1, 0, 1],
    [0, -1, -1], [0, 1, -1], [0, -1, 1], [0, 1, 1],

    [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
    [1, 1, -1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]], dtype=np.int8)

neighbourhood2d = np.array( [
    [0,-1, 0], [0, 0,-1],
    [0, 1, 0], [0, 0, 1],
    [0, -1, -1], [0, 1, -1], [0, -1, 1], [0, 1, 1],
], dtype=np.int8)

