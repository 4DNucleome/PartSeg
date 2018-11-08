from abc import ABC
from collections import defaultdict
from enum import Enum
from project_utils.segmentation.algorithm_base import SegmentationAlgorithm, AlgorithmProperty
from project_utils.distance_in_structure.find_split import distance_sprawl, path_minimum_sprawl, path_maximum_sprawl
import numpy as np
import SimpleITK as sitk
from project_utils import bisect
import operator

from project_utils.image_operations import RadiusType


def blank_operator(_x, _y):
    raise NotImplemented()


class RestartableAlgorithm(SegmentationAlgorithm, ABC):

    def __init__(self, **kwargs):
        super().__init__()
        self.parameters = defaultdict(lambda: None)
        self.new_parameters = {}

    def set_image(self, image):
        super().set_image(image)
        self.parameters.clear()

    def get_info_text(self):
        return "No info [Report this ass error]"


class ThresholdBaseAlgorithm(RestartableAlgorithm, ABC):
    """
    :type segmentation: np.ndarray
    """

    threshold_operator = blank_operator

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("minimum_size", "Minimum size (pix)", 8000, (0, 10 ** 6), 1000),
                AlgorithmProperty("use_gauss", "Use gauss", RadiusType.NO, None),
                AlgorithmProperty("gauss_radius", "Gauss radius", 1.0, (0, 10), 0.1),
                AlgorithmProperty("side_connection", "Connect only sides", False, (True, False))]

    def __init__(self, **kwargs):
        super(ThresholdBaseAlgorithm, self).__init__()
        self.mask = None
        self.gauss_image = None
        self.threshold_image = None
        self._sizes_array = []
        self.components_num = 0

    def get_info_text(self):
        return ", ".join(map(str, self._sizes_array[1:self.components_num + 1]))

    def calculation_run(self, _report_fun):
        """main calculation function.  return segmentation, full_segmentation"""
        restarted = False
        if self.parameters["channel"] != self.new_parameters["channel"]:
            self.channel = self.get_channel(self.new_parameters["channel"])
            restarted = True
        if restarted or self.parameters["gauss_radius"] != self.new_parameters["gauss_radius"] or \
                self.new_parameters["use_gauss"] != self.parameters["use_gauss"]:
            self.gauss_image = self.get_gauss(self.new_parameters["use_gauss"], self.new_parameters["gauss_radius"])
            restarted = True
        if restarted or self.new_parameters["threshold"] != self.parameters["threshold"] \
                or self.new_parameters["side_connection"] != self.parameters["side_connection"]:
            threshold_image = self._threshold(self.gauss_image)
            if self.mask is not None:
                threshold_image *= (self.mask > 0)
            connect = sitk.ConnectedComponent(sitk.GetImageFromArray(threshold_image),
                                              self.new_parameters["side_connection"])
            self.segmentation = sitk.GetArrayFromImage(sitk.RelabelComponent(connect))
            self._sizes_array = np.bincount(self.segmentation.flat)
            restarted = True
        if restarted or self.new_parameters["minimum_size"] != self.parameters["minimum_size"]:
            minimum_size = self.new_parameters["minimum_size"]
            ind = bisect(self._sizes_array[1:], minimum_size, lambda x, y: x > y)
            finally_segment = np.copy(self.segmentation)
            finally_segment[finally_segment > ind] = 0
            self.components_num = ind
            return finally_segment, self.segmentation

    def _clean(self):
        super()._clean()
        self.parameters = defaultdict(lambda: None)
        self.gauss_image = None
        self.mask = None

    def _threshold(self, image, thr=None):
        if thr is None:
            thr = self.new_parameters["threshold"]
        return self.threshold_operator(image, thr).astype(np.uint8)

    def set_mask(self, mask):
        self.mask = mask
        self.new_parameters["threshold"] = self.parameters["threshold"]
        self.parameters["threshold"] = None

    def _set_parameters(self, channel, threshold, minimum_size, use_gauss, gauss_radius, side_connection):
        self.new_parameters["channel"] = channel
        self.new_parameters["threshold"] = threshold
        self.new_parameters["minimum_size"] = minimum_size
        self.new_parameters["use_gauss"] = use_gauss
        self.new_parameters["gauss_radius"] = gauss_radius
        self.new_parameters["side_connection"] = side_connection


class OneThresholdAlgorithm(ThresholdBaseAlgorithm, ABC):
    def set_parameters(self, *args, **kwargs):
        self._set_parameters(*args, **kwargs)

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("threshold", "Threshold", 10000, (0, 10 ** 6), 100)] + super().get_fields()


class LowerThresholdAlgorithm(OneThresholdAlgorithm):
    threshold_operator = operator.gt

    @classmethod
    def get_name(cls):
        return "Lower threshold"


class UpperThresholdAlgorithm(OneThresholdAlgorithm):
    threshold_operator = operator.lt

    @classmethod
    def get_name(cls):
        return "Upper threshold"


class RangeThresholdAlgorithm(ThresholdBaseAlgorithm):
    def set_parameters(self, lower_threshold, upper_threshold, *args, **kwargs):
        self._set_parameters(threshold=(lower_threshold, upper_threshold), *args, **kwargs)

    def _threshold(self, image, thr=None):
        return ((image > self.new_parameters["threshold"][0]) * (image < self.new_parameters["threshold"][1])).astype(
            np.uint8)

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("lower_threshold", "Lower threshold", 10000, (0, 10 ** 6), 100),
                AlgorithmProperty("upper_threshold", "Upper threshold", 10000, (0, 10 ** 6), 100)] + \
               super().get_fields()

    @classmethod
    def get_name(cls):
        return "Range threshold"


class BaseThresholdFlowAlgorithm(ThresholdBaseAlgorithm, ABC):
    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("threshold", "Threshold", 10000, (0, 10 ** 6), 100),
                AlgorithmProperty("base_threshold", "Base threshold", 10000, (0, 10 ** 6), 100)] + super().get_fields()

    def path_sprawl(self, base_image, object_image) -> np.ndarray:
        raise NotImplementedError()

    def get_info_text(self):
        return "Mid sizes: " + ", ".join(map(str, self._sizes_array[1:self.components_num + 1])) + \
               "\nFinal sizes: " + ", ".join(map(str, self.final_sizes[1:]))

    def __init__(self):
        super().__init__()
        self.finally_segment = None
        self.final_sizes = []

    def set_parameters(self, base_threshold, *args, **kwargs):
        self._set_parameters(*args, **kwargs)
        self.new_parameters["base_threshold"] = base_threshold

    def calculation_run(self, report_fun):
        segment_data = super().calculation_run(report_fun)
        if segment_data is not None and self.components_num == 0:
            self.final_sizes = []
            return segment_data

        if segment_data is None:
            restarted = False
            finally_segment = np.copy(self.finally_segment)
        else:
            self.finally_segment = segment_data[0]
            finally_segment = segment_data[0]
            restarted = True

        if restarted or self.new_parameters["base_threshold"] != self.parameters["base_threshold"]:
            if self.threshold_operator(self.new_parameters["base_threshold"], self.new_parameters["threshold"]):
                return self.finally_segment, self.segmentation
            threshold_image = self._threshold(self.gauss_image, self.new_parameters["base_threshold"])
            if self.mask is not None:
                threshold_image *= (self.mask > 0)
            new_segment = self.path_sprawl(threshold_image, finally_segment)
            self.final_sizes = np.bincount(new_segment.flat)
            return new_segment, threshold_image


class LowerThresholdFlowAlgorithm(BaseThresholdFlowAlgorithm, ABC):
    threshold_operator = operator.gt


class UpperThresholdFlowAlgorithm(BaseThresholdFlowAlgorithm, ABC):
    threshold_operator = operator.lt


class LowerThresholdDistanceFlowAlgorithm(LowerThresholdFlowAlgorithm):
    def path_sprawl(self, base_image, object_image):
        neigh, dist = calculate_distances_array(self.image.spacing, get_neigh(self.new_parameters["side_connection"]))
        return distance_sprawl(base_image, object_image, self.components_num, neigh, dist)

    @classmethod
    def get_name(cls):
        return "Lower threshold euclidean"


class UpperThresholdDistanceFlowAlgorithm(UpperThresholdFlowAlgorithm):
    def path_sprawl(self, base_image, object_image):
        neigh, dist = calculate_distances_array(self.image.spacing, get_neigh(self.new_parameters["side_connection"]))
        return distance_sprawl(base_image, object_image, self.components_num, neigh, dist)

    @classmethod
    def get_name(cls):
        return "Upper threshold euclidean"


class LowerThresholdPathFlowAlgorithm(LowerThresholdFlowAlgorithm):
    def path_sprawl(self, base_image, object_image):
        image = self.channel.astype(np.float64)
        image[base_image == 0] = 0
        neigh = get_neighbourhood(self.image.spacing, get_neigh(self.new_parameters["side_connection"]))
        mid = path_maximum_sprawl(image, object_image, self.components_num, neigh)
        return path_maximum_sprawl(image, mid, self.components_num, neigh)

    @classmethod
    def get_name(cls):
        return "Lower threshold path"


class UpperThresholdPathFlowAlgorithm(UpperThresholdFlowAlgorithm):
    def path_sprawl(self, base_image, object_image):
        image = self.channel.astype(np.float64)
        image[base_image == 0] = 0
        neigh = get_neighbourhood(self.image.spacing, get_neigh(self.new_parameters["side_connection"]))
        mid = path_minimum_sprawl(image, object_image, self.components_num, neigh)
        return path_minimum_sprawl(image, mid, self.components_num, neigh)

    @classmethod
    def get_name(cls):
        return "Upper threshold path"


class LowerThresholdPathDistanceFlowAlgorithm(LowerThresholdPathFlowAlgorithm):
    def path_sprawl(self, base_image, object_image):
        mid = super().path_sprawl(base_image, object_image)
        neigh, dist = calculate_distances_array(self.image.spacing, get_neigh(self.new_parameters["side_connection"]))
        return distance_sprawl(base_image, mid, self.components_num, neigh, dist)

    @classmethod
    def get_name(cls):
        return "Lower threshold path euclidean"


class UpperThresholdPathDistanceFlowAlgorithm(UpperThresholdPathFlowAlgorithm):
    def path_sprawl(self, base_image, object_image):
        mid = super().path_sprawl(base_image, object_image)
        neigh, dist = calculate_distances_array(self.image.spacing, get_neigh(self.new_parameters["side_connection"]))
        return distance_sprawl(base_image, mid, self.components_num, neigh, dist)

    @classmethod
    def get_name(cls):
        return "Upper threshold path euclidean"


def get_neigh(sides):
    if sides:
        return NeighType.sides
    else:
        return NeighType.edges


class NeighType(Enum):
    sides = 6
    edges = 18
    vertex = 26


def calculate_distances_array(spacing, neigh_type: NeighType):
    min_dist = min(spacing)
    normalized_spacing = [x / min_dist for x in spacing]
    if len(normalized_spacing) == 2:
        neighbourhood_array = neighbourhood2d
        if neigh_type == NeighType.sides:
            neighbourhood_array = neighbourhood_array[:4]
        normalized_spacing = [0] + normalized_spacing
    else:
        neighbourhood_array = neighbourhood[:neigh_type.value]
    normalized_spacing = np.array(normalized_spacing)
    return neighbourhood_array, np.sqrt(np.sum((neighbourhood_array * normalized_spacing) ** 2, axis=1))


def get_neighbourhood(spacing, neigh_type: NeighType):
    if len(spacing) == 2:
        if neigh_type == NeighType.sides:
            return neighbourhood2d[:4]
        return neighbourhood2d
    else:
        return neighbourhood[:neigh_type.value]


neighbourhood = \
    np.array([[0, -1, 0], [0, 0, -1],
              [0, 1, 0], [0, 0, 1],
              [-1, 0, 0], [1, 0, 0],

              [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 1, 0],
              [-1, 0, -1], [1, 0, -1], [-1, 0, 1], [1, 0, 1],
              [0, -1, -1], [0, 1, -1], [0, -1, 1], [0, 1, 1],

              [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
              [1, 1, -1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]], dtype=np.int8)

neighbourhood2d = \
    np.array([[0, -1, 0], [0, 0, -1], [0, 1, 0], [0, 0, 1],
              [0, -1, -1], [0, 1, -1], [0, -1, 1], [0, 1, 1]], dtype=np.int8)


final_algorithm_list = [LowerThresholdAlgorithm, UpperThresholdAlgorithm, RangeThresholdAlgorithm,
                        LowerThresholdDistanceFlowAlgorithm, UpperThresholdDistanceFlowAlgorithm,
                        LowerThresholdPathFlowAlgorithm, UpperThresholdPathFlowAlgorithm,
                        UpperThresholdPathDistanceFlowAlgorithm, LowerThresholdPathDistanceFlowAlgorithm]
