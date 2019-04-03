from abc import ABC

import SimpleITK as sitk
import numpy as np
from typing import Optional
import operator
from ..utils import bisect
from ..channel_class import Channel
from ..segmentation.algorithm_base import SegmentationAlgorithm, SegmentationResult
from ..convex_fill import convex_fill
from ..image_operations import RadiusType
from ..algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty, SegmentationProfile
from .noise_filtering import noise_removal_dict
from .threshold import threshold_dict, BaseThreshold
from .segment import close_small_holes, opening


class StackAlgorithm(SegmentationAlgorithm, ABC):
    def __init__(self):
        super().__init__()
        self.channel_num = 0

    def _clean(self):
        super()._clean()


class ThresholdPreview(StackAlgorithm):
    @classmethod
    def get_fields(cls):
        return [
                AlgorithmProperty("channel", "Channel", 0, property_type=Channel),
                AlgorithmProperty("noise_removal", "Noise Removal", next(iter(noise_removal_dict.keys())),
                                  possible_values=noise_removal_dict, property_type=AlgorithmDescribeBase),
                AlgorithmProperty("threshold", "Threshold", 1000, (0, 10 ** 6), 100)]

    @classmethod
    def get_name(cls):
        return "Only Threshold"

    def __init__(self):
        super(ThresholdPreview, self).__init__()
        self.noise_removal = None
        self.threshold = 0

    def calculation_run(self, report_fun) -> SegmentationResult:
        self.channel = self.get_channel(self.channel_num)
        image = noise_removal_dict[self.noise_removal["name"]].noise_remove(self.channel, self.image.spacing,
                                                                            self.noise_removal["values"])
        res = (image > self.threshold).astype(np.uint8)
        if self.mask is not None:
            res[self.mask == 0] = 0
        self.image = None
        self.channel = None
        return SegmentationResult(res, self.get_segmentation_profile(), res, cleaned_channel=self.channel)

    def set_parameters(self, channel, threshold, noise_removal):
        self.channel_num = channel
        self.threshold = threshold
        self.noise_removal = noise_removal

    def get_info_text(self):
        return ""

    def get_segmentation_profile(self) -> SegmentationProfile:
        return SegmentationProfile("", self.get_name(), {'channel': self.channel_num, "threshold": self.threshold,
                                                         'noise_removal': self.noise_removal})


class BaseThresholdAlgorithm(StackAlgorithm, ABC):
    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("channel", "Channel", 0, property_type=Channel),
                AlgorithmProperty("threshold", "Threshold", next(iter(threshold_dict.keys())),
                                  possible_values=threshold_dict, property_type=AlgorithmDescribeBase),
                AlgorithmProperty("minimum_size", "Minimum size", 8000, (20, 10 ** 6), 1000),
                AlgorithmProperty("close_holes", "Fill holes", True, (True, False)),
                AlgorithmProperty("close_holes_size", "Maximum holes size (px)", 200, (0, 10 ** 3), 10),
                AlgorithmProperty("smooth_border", "Smooth borders", True, (True, False)),
                AlgorithmProperty("smooth_border_radius", "Smooth borders radius", 2, (0, 20), 1),
                AlgorithmProperty("noise_removal", "Filter", next(iter(noise_removal_dict.keys())),
                                  possible_values=noise_removal_dict, property_type=AlgorithmDescribeBase),
                AlgorithmProperty("side_connection", "Side by Side connections", False, (True, False),
                                  tool_tip="During calculation of connected components includes"
                                           " only side by side connected pixels"),
                AlgorithmProperty("use_convex", "Use convex hull", False, (True, False))]

    def __init__(self):
        super().__init__()
        self.threshold = None
        self.minimum_size = None
        self.sizes = None
        self.noise_removal = None
        self.close_holes = False
        self.close_holes_size = 0
        self.smooth_border = False
        self.smooth_border_radius = 0
        self.gauss_2d = False
        self.edge_connection = True
        self.use_convex = False

    @staticmethod
    def get_steps_num():
        return 7

    def _threshold_image(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def _threshold_and_exclude(self, image, report_fun):
        raise NotImplementedError()

    def calculation_run(self, report_fun):
        report_fun("Noise removal", 0)
        self.channel = self.get_channel(self.channel_num)
        image = noise_removal_dict[self.noise_removal["name"]].noise_remove(self.channel, self.image.spacing,
                                                                            self.noise_removal["values"])
        mask = self._threshold_and_exclude(image, report_fun)
        if self.close_holes:
            report_fun("Filing holes", 3)
            mask = close_small_holes(mask, self.close_holes_size)
        report_fun("Components calculating", 4)
        self.segmentation = sitk.GetArrayFromImage(
            sitk.RelabelComponent(
                sitk.ConnectedComponent(
                    sitk.GetImageFromArray(mask), self.edge_connection
                ), 20
            )
        )
        if self.smooth_border:
            report_fun("Smoothing borders", 5)
            self.segmentation = opening(self.segmentation, self.smooth_border_radius, 20)

        self.sizes = np.bincount(self.segmentation.flat)
        ind = bisect(self.sizes[1:], self.minimum_size, lambda x, y: x > y)
        self.threshold = self.threshold
        self.minimum_size = self.minimum_size
        resp = np.copy(self.segmentation)
        resp[resp > ind] = 0
        if self.use_convex:
            report_fun("convex hull", 6)
            resp = convex_fill(resp)
        report_fun("Calculation done", 7)
        return SegmentationResult(resp, self.get_segmentation_profile(), self.segmentation, image)

    def _set_parameters(self, channel, threshold, minimum_size, close_holes, smooth_border, noise_removal,
                        close_holes_size, smooth_border_radius, side_connection, use_convex):
        self.channel_num = channel
        self.threshold = threshold
        self.minimum_size = minimum_size
        self.close_holes = close_holes
        self.smooth_border = smooth_border
        self.noise_removal = noise_removal
        self.close_holes_size = close_holes_size
        self.smooth_border_radius = smooth_border_radius
        self.edge_connection = not side_connection
        self.use_convex = use_convex

    def get_segmentation_profile(self) -> SegmentationProfile:
        return SegmentationProfile("", self.get_name(), {
            "channel": self.channel_num, "threshold": self.threshold, "minimum_size": self.minimum_size,
            "close_holes": self.close_holes, "smooth_border": self.smooth_border, "noise_removal": self.noise_removal,
            "close_holes_size": self.close_holes_size, "smooth_border_radius": self.smooth_border_radius,
            "side_connection": not self.edge_connection, "use_convex": self.use_convex
        })


class ThresholdAlgorithm(BaseThresholdAlgorithm):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_name(cls):
        return "Threshold"

    def _threshold_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        return None

    def _threshold_and_exclude(self, image, report_fun):
        report_fun("Threshold calculation", 1)
        threshold_algorithm: BaseThreshold = threshold_dict[self.threshold["name"]]
        mask, thr_val = threshold_algorithm.calculate_mask(image, self.mask, self.threshold["values"], operator.ge)
        report_fun("Threshold calculated", 2)
        return mask

    def set_parameters(self, *args, **kwargs):
        super()._set_parameters(*args, **kwargs)

    def get_info_text(self):
        return ""


class AutoThresholdAlgorithm(BaseThresholdAlgorithm):
    @classmethod
    def get_fields(cls):
        res = super().get_fields()
        res.insert(1, AlgorithmProperty("suggested_size", "Suggested size", 200000, (0, 10 ** 6), 1000))
        return res

    @classmethod
    def get_name(cls):
        return "Auto Threshold"

    def __init__(self):
        super().__init__()
        self.suggested_size = 0

    def get_gauss(self, gauss_type, gauss_radius):
        if gauss_type == RadiusType.NO:
            return np.copy(self.channel)
        else:
            return super().get_gauss(gauss_type, gauss_radius)

    def _threshold_image(self, image: np.ndarray) -> np.ndarray:
        sitk_image = sitk.GetImageFromArray(image)
        sitk_mask = sitk.ThresholdMaximumConnectedComponents(sitk_image, self.suggested_size)
        # TODO what exactly it returns. Maybe it is already segmented.
        mask = sitk.GetArrayFromImage(sitk_mask)
        min_val = np.min(image[mask > 0])
        threshold_algorithm: BaseThreshold = threshold_dict[self.threshold["name"]]
        mask2, thr_val = threshold_algorithm.calculate_mask(image, None, self.threshold["values"], operator.le)
        if thr_val < min_val:
            return mask
        else:
            return mask2

    def _threshold_and_exclude(self, image, report_fun):
        if self.mask is not None:
            report_fun("Components exclusion apply", 1)
            image[self.mask == 0] = 0
        report_fun("Threshold calculation", 2)
        mask = self._threshold_image(image)
        return mask

    def set_parameters(self, suggested_size, *args, **kwargs):
        self._set_parameters(*args, **kwargs)
        self.suggested_size = suggested_size

    def get_info_text(self):
        return ""

    def get_segmentation_profile(self):
        resp = super().get_segmentation_profile()
        resp.values["suggested_size"] = self.suggested_size
        return resp


final_algorithm_list = [ThresholdAlgorithm, ThresholdPreview, AutoThresholdAlgorithm]
