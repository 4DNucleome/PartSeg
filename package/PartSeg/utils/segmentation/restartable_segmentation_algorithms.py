import operator
from abc import ABC
from collections import defaultdict
from copy import deepcopy
import SimpleITK as sitk
import numpy as np

from ..multiscale_opening import PyMSO
from ..multiscale_opening import calculate_mu_mid
from ..border_rim import border_mask
from ..channel_class import Channel
from .algorithm_base import SegmentationAlgorithm, SegmentationResult
from ..algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty, SegmentationProfile
from .noise_filtering import noise_removal_dict
from .sprawl import sprawl_dict, BaseSprawl, calculate_distances_array, get_neigh
from .threshold import threshold_dict, BaseThreshold, double_threshold_dict
from ..universal_const import Units
from ..utils import bisect
from .mu_mid_point import mu_mid_dict, BaseMuMid


def blank_operator(_x, _y):
    raise NotImplemented()


class RestartableAlgorithm(SegmentationAlgorithm, ABC):

    def __init__(self, **kwargs):
        super().__init__()
        self.parameters = defaultdict(lambda: None)
        self.new_parameters = {}

    def set_image(self, image):
        super().set_image(image)

    def set_mask(self, mask):
        super().set_mask(mask)
        self.new_parameters["threshold"] = self.parameters["threshold"]
        self.parameters["threshold"] = None

    def get_info_text(self):
        return "No info [Report this ass error]"

    def get_segmentation_profile(self) -> SegmentationProfile:
        return SegmentationProfile("", self.get_name(), deepcopy(self.new_parameters))


class BorderRim(RestartableAlgorithm):
    @classmethod
    def get_name(cls):
        return "Border Rim"

    def __init__(self):
        super().__init__()
        self.distance = 0
        self.units = Units.nm

    @classmethod
    def get_fields(cls):
        return ["Need mask",
                AlgorithmProperty("distance", "Distance", 700.0, options_range=(0, 100000), property_type=float),
                AlgorithmProperty("units", "Units", Units.nm, property_type=Units)]

    def set_parameters(self, distance: float, units: Units):
        self.distance = distance
        self.units = units

    def get_segmentation_profile(self) -> SegmentationProfile:
        return SegmentationProfile("", self.get_name(), {"distance": self.distance, "units":self.units})

    def get_info_text(self):
        if self.mask is None:
            return "Need mask"
        else:
            return ""

    def calculation_run(self, _report_fun) -> SegmentationResult:
        if self.mask is not None:
            result = \
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
                AlgorithmProperty("minimum_size", "Minimum size (px)", 8000, (0, 10 ** 6), 1000),
                AlgorithmProperty("noise_removal", "Filter", next(iter(noise_removal_dict.keys())),
                                  possible_values=noise_removal_dict, property_type=AlgorithmDescribeBase),
                AlgorithmProperty("side_connection", "Connect only sides", False, (True, False),
                                  tool_tip="During calculation of connected components includes"
                                           " only side by side connected pixels")]

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
        if self.channel is None or self.parameters["channel"] != self.new_parameters["channel"]:
            self.parameters["channel"] = self.new_parameters["channel"]
            self.channel = self.get_channel(self.new_parameters["channel"])
            restarted = True
        if restarted or self.parameters["noise_removal"] != self.new_parameters["noise_removal"]:
            self.parameters["noise_removal"] = deepcopy(self.new_parameters["noise_removal"])
            noise_removal_parameters = self.new_parameters["noise_removal"]
            self.cleaned_image = noise_removal_dict[noise_removal_parameters["name"]]. \
                noise_remove(self.channel, self.image.spacing, noise_removal_parameters["values"])
            restarted = True
        if restarted or self.new_parameters["threshold"] != self.parameters["threshold"]:
            self.parameters["threshold"] = deepcopy(self.new_parameters["threshold"])
            self.threshold_image = self._threshold(self.cleaned_image)
            if isinstance(self.threshold_info, (list, tuple)):
                if self.old_threshold_info is None or self.old_threshold_info[0] != self.threshold_info[0]:
                    restarted = True
            elif self.old_threshold_info != self.threshold_info:
                restarted = True
        if restarted or self.new_parameters["side_connection"] != self.parameters["side_connection"]:
            self.parameters["side_connection"] = self.new_parameters["side_connection"]
            connect = sitk.ConnectedComponent(sitk.GetImageFromArray(self.threshold_image),
                                              not self.new_parameters["side_connection"])
            self.segmentation = sitk.GetArrayFromImage(sitk.RelabelComponent(connect))
            self._sizes_array = np.bincount(self.segmentation.flat)
            restarted = True
        if restarted or self.new_parameters["minimum_size"] != self.parameters["minimum_size"]:
            self.parameters["minimum_size"] = self.new_parameters["minimum_size"]
            minimum_size = self.new_parameters["minimum_size"]
            ind = bisect(self._sizes_array[1:], minimum_size, lambda x, y: x > y)
            finally_segment = np.copy(self.segmentation)
            finally_segment[finally_segment > ind] = 0
            self.components_num = ind
            return SegmentationResult(finally_segment, self.get_segmentation_profile(),
                                      self.segmentation, self.cleaned_image)

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

    def get_segmentation_profile(self) -> SegmentationProfile:
        resp = super().get_segmentation_profile()
        low, upp = resp.values["threshold"]
        del resp.values["threshold"]
        resp.values["lower_threshold"] = low
        resp.values["upper_threshold"] = upp
        return resp

    def _threshold(self, image, thr=None):
        self.threshold_info = self.new_parameters["threshold"]
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
                                  possible_values=double_threshold_dict, property_type=AlgorithmDescribeBase),
                AlgorithmProperty("sprawl_type", "Flow type", next(iter(sprawl_dict.keys())),
                                  possible_values=sprawl_dict,
                                  property_type=AlgorithmDescribeBase)] \
               + super().get_fields()

    def get_info_text(self):
        return f"Threshold: " + ", ".join(map(str, self.threshold_info)) + \
               "\nMid sizes: " + ", ".join(map(str, self._sizes_array[1:self.components_num + 1])) + \
               "\nFinal sizes: " + ", ".join(map(str, self.final_sizes[1:]))

    def __init__(self):
        super().__init__()
        self.finally_segment = None
        self.final_sizes = []
        self.threshold_info = [None, None]
        self.sprawl_area = None

    def _clean(self):
        self.sprawl_area = None
        super()._clean()

    def _threshold(self, image, thr=None):
        if thr is None:
            thr: BaseThreshold = double_threshold_dict[self.new_parameters["threshold"]["name"]]
        mask, thr_val = thr.calculate_mask(image, self.mask, self.new_parameters["threshold"]["values"],
                                           self.threshold_operator)
        self.threshold_info = thr_val
        self.sprawl_area = (mask >= 1).astype(np.uint8)
        return (mask == 2).astype(np.uint8)

    def set_parameters(self, sprawl_type, *args, **kwargs):
        self.new_parameters["sprawl_type"] = sprawl_type
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

        if restarted or self.old_threshold_info[1] != self.threshold_info[1] or \
                self.new_parameters["sprawl_type"] != self.parameters["sprawl_type"]:
            if self.threshold_operator(self.threshold_info[1], self.threshold_info[0]):
                self.final_sizes = np.bincount(finally_segment.flat)
                return SegmentationResult(self.finally_segment, self.segmentation, self.cleaned_image)
            path_sprawl: BaseSprawl = sprawl_dict[self.new_parameters["sprawl_type"]["name"]]
            self.parameters["sprawl_type"] = self.new_parameters["sprawl_type"]
            new_segment = path_sprawl.sprawl(self.sprawl_area, finally_segment, self.channel, self.components_num,
                                             self.image.spacing,
                                             self.new_parameters["side_connection"], self.threshold_operator,
                                             self.new_parameters["sprawl_type"]["values"], self.threshold_info[1],
                                             self.threshold_info[0])
            self.final_sizes = np.bincount(new_segment.flat)
            return SegmentationResult(new_segment, self.get_segmentation_profile(), self.sprawl_area, self.cleaned_image)


class LowerThresholdFlowAlgorithm(BaseThresholdFlowAlgorithm):
    threshold_operator = operator.gt

    @classmethod
    def get_name(cls):
        return "Lower threshold flow"


class UpperThresholdFlowAlgorithm(BaseThresholdFlowAlgorithm):
    threshold_operator = operator.lt

    @classmethod
    def get_name(cls):
        return "Upper threshold flow"


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
                # AlgorithmProperty("mask", "Use mask in calculation", True),
                AlgorithmProperty("valley", "Valley emphasis", True),
                AlgorithmProperty("hist_num", "Number of histogram bins", 128, (8, 2 ** 16))]

    def __init__(self):
        super().__init__()
        self._sizes_array = []
        self.threshold_info = []

    def set_parameters(self, channel, noise_removal, components, valley, hist_num):  # mask
        self.new_parameters["components"] = components
        # self.new_parameters["mask"] = mask
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


class BaseMultiScaleOpening(ThresholdBaseAlgorithm, ABC):
    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("threshold", "Threshold", next(iter(double_threshold_dict.keys())),
                                  possible_values=double_threshold_dict, property_type=AlgorithmDescribeBase),
                AlgorithmProperty("mu_mid", "Mu mid value", next(iter(mu_mid_dict.keys())),
                                  possible_values=mu_mid_dict,
                                  property_type=AlgorithmDescribeBase),
                AlgorithmProperty("step_limits", "Limits of Steps", 100, options_range=(1, 1000), property_type=int)] \
               + super().get_fields()

    def get_info_text(self):
        return f"Threshold: " + ", ".join(map(str, self.threshold_info)) + \
               "\nMid sizes: " + ", ".join(map(str, self._sizes_array[1:self.components_num + 1])) + \
               "\nFinal sizes: " + ", ".join(map(str, self.final_sizes[1:])) + \
               f"\nsteps: {self.steps}"

    def __init__(self):
        super().__init__()
        self.finally_segment = None
        self.final_sizes = []
        self.threshold_info = [None, None]
        self.sprawl_area = None
        self.steps = 0
        self.mso = PyMSO()
        self.mso.set_use_background(True)

    def _clean(self):
        self.sprawl_area = None
        self.mso = PyMSO()
        self.mso.set_use_background(True)
        super()._clean()

    def _threshold(self, image, thr=None):
        if thr is None:
            thr: BaseThreshold = double_threshold_dict[self.new_parameters["threshold"]["name"]]
        mask, thr_val = thr.calculate_mask(image, self.mask, self.new_parameters["threshold"]["values"],
                                           self.threshold_operator)
        self.threshold_info = thr_val
        self.sprawl_area = (mask >= 1).astype(np.uint8)
        return (mask == 2).astype(np.uint8)

    def set_parameters(self, mu_mid, step_limits, *args, **kwargs):
        self.new_parameters["mu_mid"] = mu_mid
        self.new_parameters["step_limits"] = step_limits
        self._set_parameters(*args, **kwargs)

    def set_image(self, image):
        super().set_image(image)
        self.threshold_info = [None, None]

    def calculation_run(self, report_fun) -> SegmentationResult:
        if self.new_parameters["side_connection"] != self.parameters["side_connection"]:
            neigh, dist = calculate_distances_array(self.image.spacing, get_neigh(self.new_parameters["side_connection"]))
            self.mso.set_neighbourhood(neigh, dist)
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
            assert finally_segment.max() < 250
            components = finally_segment.astype(np.uint8)
            components[components > 0] += 1
            components[self.sprawl_area == 0] = 1
            self.mso.set_components(components, self.components_num)
            restarted = True

        if restarted or self.old_threshold_info[1] != self.threshold_info[1] or \
                self.new_parameters["mu_mid"] != self.parameters["mu_mid"]:
            if self.threshold_operator(self.threshold_info[1], self.threshold_info[0]):
                self.final_sizes = np.bincount(finally_segment.flat)
                return SegmentationResult(self.finally_segment, self.segmentation, self.cleaned_image)
            mu_calc: BaseMuMid = mu_mid_dict[self.new_parameters["mu_mid"]["name"]]
            self.parameters["mu_mid"] = self.new_parameters["mu_mid"]
            sprawl_area = (self.sprawl_area > 0).astype(np.uint8)
            sprawl_area[finally_segment > 0] = 0
            mid_val = mu_calc.value(sprawl_area, self.channel, self.threshold_info[0], self.threshold_info[1],
                                    self.new_parameters["mu_mid"]["values"])
            mu_array = calculate_mu_mid(self.channel, self.threshold_info[0], mid_val, self.threshold_info[1])
            self.mso.set_mu_array(mu_array)
            restarted = True

        if restarted or self.new_parameters["step_limits"] != self.parameters["step_limits"]:
            self.parameters["step_limits"] = self.new_parameters["step_limits"]
            self.mso.run_MSO(self.new_parameters["step_limits"])
            self.steps = self.mso.steps_done()
            new_segment = self.mso.get_result_catted()
            new_segment[new_segment > 0] -= 1
            self.final_sizes = np.bincount(new_segment.flat)
            return SegmentationResult(new_segment, self.sprawl_area, self.cleaned_image)


class LowerThresholdMultiScaleOpening(BaseMultiScaleOpening):
    threshold_operator = operator.gt

    @classmethod
    def get_name(cls):
        return "Lower threshold MultiScale Opening"


class UpperThresholdMultiScaleOpening(BaseMultiScaleOpening):
    threshold_operator = operator.lt

    @classmethod
    def get_name(cls):
        return "Upper threshold MultiScale Opening"


final_algorithm_list = [LowerThresholdAlgorithm, UpperThresholdAlgorithm, RangeThresholdAlgorithm,
                        LowerThresholdFlowAlgorithm, UpperThresholdFlowAlgorithm, # LowerThresholdMultiScaleOpening,
                        # UpperThresholdMultiScaleOpening,
                        OtsuSegment, BorderRim]
