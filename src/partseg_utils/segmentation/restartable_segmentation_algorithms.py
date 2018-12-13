import operator
from abc import ABC
from collections import defaultdict
from enum import Enum

import SimpleITK as sitk
import numpy as np

from partseg_utils import bisect
from partseg_utils.border_rim import border_mask
from partseg_utils.channel_class import Channel
from partseg_utils.distance_in_structure.find_split import distance_sprawl, path_minimum_sprawl, path_maximum_sprawl
from partseg_utils.segmentation.algorithm_base import SegmentationAlgorithm, SegmentationResult
from partseg_utils.segmentation.algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty
from partseg_utils.segmentation.noise_removing import noise_removal_dict
from partseg_utils.segmentation.threshold import threshold_dict, BaseThreshold, double_threshold_dict
from partseg_utils.universal_const import UNITS_LIST


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

    def set_mask(self, mask):
        super().set_mask(mask)
        self.new_parameters["threshold"] = self.parameters["threshold"]
        self.parameters["threshold"] = None

    def get_info_text(self):
        return "No info [Report this ass error]"


class BorderRim(RestartableAlgorithm):
    @classmethod
    def get_name(cls):
        return "Border Rim"

    def __init__(self):
        super().__init__()
        self.distance = 0
        self.units = ""

    @classmethod
    def get_fields(cls):
        return ["Need mask",
                AlgorithmProperty("distance", "Distance", 700.0, options_range=(0, 100000), property_type=float),
                AlgorithmProperty("units", "Units", "nm", possible_values=UNITS_LIST)]

    def set_parameters(self, distance, units):
        self.distance = distance
        self.units = units

    def get_info_text(self):
        if self.mask is None:
            return "Need mask"
        else:
            return ""

    def calculation_run(self, _report_fun) -> SegmentationResult:
        if self.mask is not None:
            result =\
                border_mask(mask=self.mask, distance=self.distance, units=self.units, voxel_size=self.image.spacing)
            return SegmentationResult(result, result, None)


class ThresholdBaseAlgorithm(RestartableAlgorithm, ABC):
    """
    :type segmentation: np.ndarray
    """

    threshold_operator = blank_operator

    @classmethod
    def get_fields(cls):
        # TODO coś z noise removal zrobić
        return [AlgorithmProperty("channel", "Channel", 0, property_type=Channel),
                AlgorithmProperty("minimum_size", "Minimum size (pix)", 8000, (0, 10 ** 6), 1000),
                AlgorithmProperty("noise_removal", "Filtering", next(iter(noise_removal_dict.keys())),
                                  possible_values=noise_removal_dict, property_type=AlgorithmDescribeBase),
                AlgorithmProperty("side_connection", "Connect only sides", False, (True, False))]

    def __init__(self, **kwargs):
        super(ThresholdBaseAlgorithm, self).__init__()
        self.cleaned_image = None
        self.threshold_image = None
        self._sizes_array = []
        self.components_num = 0
        self.threshold_info = None
        self.old_threshold_info = None

    def set_image(self, image):
        super().set_image(image)
        self.threshold_info = None

    def get_info_text(self):
        return f"Threshold: {self.threshold_info}\nSizes: " + \
               ", ".join(map(str, self._sizes_array[1:self.components_num + 1]))

    def calculation_run(self, _report_fun) -> SegmentationResult:
        """main calculation function.  return segmentation, full_segmentation"""
        self.old_threshold_info = self.threshold_info
        restarted = False
        if self.parameters["channel"] != self.new_parameters["channel"]:
            self.channel = self.get_channel(self.new_parameters["channel"])
            restarted = True
        if restarted or self.parameters["noise_removal"] != self.new_parameters["noise_removal"]:
            noise_removal_parameters = self.new_parameters["noise_removal"]
            self.cleaned_image = noise_removal_dict[noise_removal_parameters["name"]]. \
                noise_remove(self.channel, self.image.spacing, noise_removal_parameters["values"])
            restarted = True
        if restarted or self.new_parameters["threshold"] != self.parameters["threshold"]:
            self.threshold_image = self._threshold(self.cleaned_image)
            if isinstance(self.threshold_info, (list, tuple)):
                if self.old_threshold_info[0] != self.threshold_info[0]:
                    restarted = True
            elif self.old_threshold_info != self.threshold_info:
                restarted = True
        if restarted or self.new_parameters["side_connection"] != self.parameters["side_connection"]:
            connect = sitk.ConnectedComponent(sitk.GetImageFromArray(self.threshold_image),
                                              not self.new_parameters["side_connection"])
            self.segmentation = sitk.GetArrayFromImage(sitk.RelabelComponent(connect))
            self._sizes_array = np.bincount(self.segmentation.flat)
            restarted = True
        if restarted or self.new_parameters["minimum_size"] != self.parameters["minimum_size"]:
            minimum_size = self.new_parameters["minimum_size"]
            ind = bisect(self._sizes_array[1:], minimum_size, lambda x, y: x > y)
            finally_segment = np.copy(self.segmentation)
            finally_segment[finally_segment > ind] = 0
            self.components_num = ind
            return SegmentationResult(finally_segment, self.segmentation, self.cleaned_image)

    def _clean(self):
        super()._clean()
        self.parameters = defaultdict(lambda: None)
        self.cleaned_image = None
        self.mask = None

    def _threshold(self, image, thr=None):
        if thr is None:
            thr: BaseThreshold = threshold_dict[self.new_parameters["threshold"]["name"]]
        mask, thr_val = thr.calculate_mask(image, self.mask, self.new_parameters["threshold"]["values"],
                                           self.threshold_operator)
        self.threshold_info = thr_val
        return mask

    def _set_parameters(self, channel, threshold, minimum_size, noise_removal, side_connection):
        self.new_parameters["channel"] = channel
        self.new_parameters["threshold"] = threshold
        self.new_parameters["minimum_size"] = minimum_size
        self.new_parameters["noise_removal"] = noise_removal
        self.new_parameters["side_connection"] = side_connection


class OneThresholdAlgorithm(ThresholdBaseAlgorithm, ABC):
    def set_parameters(self, *args, **kwargs):
        self._set_parameters(*args, **kwargs)

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("threshold", "Threshold", next(iter(threshold_dict.keys())),
                                  possible_values=threshold_dict, property_type=AlgorithmDescribeBase)] \
               + super().get_fields()


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
        return ((image > self.new_parameters["threshold"][0]) *
                np.array(image < self.new_parameters["threshold"][1])).astype(np.uint8)

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
        return [AlgorithmProperty("threshold", "Threshold", next(iter(double_threshold_dict.keys())),
                                  possible_values=double_threshold_dict, property_type=AlgorithmDescribeBase)] \
               + super().get_fields()

    def path_sprawl(self, base_image, object_image) -> np.ndarray:
        raise NotImplementedError()

    def get_info_text(self):
        return f"Threshold: " + ", ".join(map(str, self.threshold_info)) + \
               "\nMid sizes: " + ", ".join(map(str, self._sizes_array[1:self.components_num + 1])) + \
               "\nFinal sizes: " + ", ".join(map(str, self.final_sizes[1:]))

    def __init__(self):
        super().__init__()
        self.finally_segment = None
        self.final_sizes = []
        self.threshold_info = [None, None]

    def _threshold(self, image, thr=None):
        if thr is None:
            thr: BaseThreshold = double_threshold_dict[self.new_parameters["threshold"]["name"]]
        mask, thr_val = thr.calculate_mask(image, self.mask, self.new_parameters["threshold"]["values"],
                                           self.threshold_operator)
        self.threshold_info = thr_val
        self.sprawl_area = (mask >= 1).astype(np.uint8)
        return (mask == 2).astype(np.uint8)

    def set_parameters(self, *args, **kwargs):
        self._set_parameters(*args, **kwargs)

    def set_image(self, image):
        super().set_image(image)
        self.threshold_info = [None, None]

    def calculation_run(self, report_fun) -> SegmentationResult:
        segment_data = super().calculation_run(report_fun)
        if segment_data is not None and self.components_num == 0:
            self.final_sizes = []
            return segment_data

        if segment_data is None:
            restarted = False
            finally_segment = np.copy(self.finally_segment)
        else:
            self.finally_segment = segment_data.segmentation
            finally_segment = segment_data.segmentation
            restarted = True

        if restarted or self.old_threshold_info[1] != self.threshold_info[1]:
            if self.threshold_operator(self.threshold_info[1], self.threshold_info[0]):
                self.final_sizes = np.bincount(finally_segment.flat)
                return SegmentationResult(self.finally_segment, self.segmentation, self.cleaned_image)
            new_segment = self.path_sprawl(self.sprawl_area, finally_segment)
            self.final_sizes = np.bincount(new_segment.flat)
            return SegmentationResult(new_segment, self.sprawl_area, self.cleaned_image)


class BaseThresholdSprawlAlgorithm(BaseThresholdFlowAlgorithm, ABC):
    def __init__(self):
        super().__init__()
        self.sprawl_area = None

    def _clean(self):
        self.sprawl_area = None
        super()._clean()


class LowerThresholdFlowAlgorithm(BaseThresholdSprawlAlgorithm, ABC):
    threshold_operator = operator.gt


class UpperThresholdFlowAlgorithm(BaseThresholdSprawlAlgorithm, ABC):
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


class OtsuSegment(RestartableAlgorithm):
    @classmethod
    def get_name(cls):
        return "Multiple Otsu"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("channel", "Channel", 0, property_type=Channel),
                AlgorithmProperty("noise_removal", "Noise Removal", next(iter(noise_removal_dict.keys())),
                                  possible_values=noise_removal_dict, property_type=AlgorithmDescribeBase),
                AlgorithmProperty("components", "Number of Components", 2, (0, 100)),
                AlgorithmProperty("mask", "Use mask in calculation", True),
                AlgorithmProperty("valley", "Valley emphasis", True),
                AlgorithmProperty("hist_num", "Number of histogram bins", 128, (8, 2 ** 16))]

    def __init__(self):
        super().__init__()
        self._sizes_array = []
        self.threshold_info = []

    def set_parameters(self, channel, noise_removal, components, mask, valley, hist_num):
        self.new_parameters["components"] = components
        self.new_parameters["mask"] = mask
        self.new_parameters["hist_num"] = hist_num
        self.new_parameters["channel"] = channel
        self.new_parameters["valley"] = valley
        self.new_parameters["noise_removal"] = noise_removal

    def calculation_run(self, report_fun):
        channel = self.get_channel(self.new_parameters["channel"])
        noise_removal_parameters = self.new_parameters["noise_removal"]
        cleaned_image = noise_removal_dict[noise_removal_parameters["name"]]. \
            noise_remove(channel, self.image.spacing, noise_removal_parameters["values"])
        cleaned_image_sitk = sitk.GetImageFromArray(cleaned_image)
        res = sitk.OtsuMultipleThresholds(cleaned_image_sitk, self.new_parameters["components"], 0,
                                          self.new_parameters["hist_num"], self.new_parameters["valley"])
        res = sitk.GetArrayFromImage(res)
        self._sizes_array = np.bincount(res.flat)[1:]
        self.threshold_info = []
        for i in range(1, self.new_parameters["components"] + 1):
            val = cleaned_image[res == i]
            if val.size:
                self.threshold_info.append(np.min(val))
            elif self.threshold_info:
                self.threshold_info.append(self.threshold_info[-1])
            else:
                self.threshold_info.append(0)
        return SegmentationResult(res, res, cleaned_image)

    def get_info_text(self):
        return f"Threshold: " + ", ".join(map(str, self.threshold_info)) + \
               "\nSizes: " + ", ".join(map(str, self._sizes_array))


final_algorithm_list = [LowerThresholdAlgorithm, UpperThresholdAlgorithm, RangeThresholdAlgorithm,
                        LowerThresholdDistanceFlowAlgorithm, UpperThresholdDistanceFlowAlgorithm,
                        LowerThresholdPathFlowAlgorithm, UpperThresholdPathFlowAlgorithm,
                        LowerThresholdPathDistanceFlowAlgorithm, UpperThresholdPathDistanceFlowAlgorithm,
                        OtsuSegment, BorderRim]
