import operator
from abc import ABC
from typing import Callable, Optional

import numpy as np
import SimpleITK as sitk

from PartSegCore.segmentation.border_smoothing import smooth_dict
from PartSegCore.segmentation.watershed import BaseWatershed, sprawl_dict

from ..algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty, ROIExtractionProfile
from ..channel_class import Channel
from ..convex_fill import convex_fill
from ..project_info import AdditionalLayerDescription
from ..segmentation.algorithm_base import SegmentationAlgorithm, SegmentationResult
from ..utils import bisect
from .noise_filtering import noise_filtering_dict
from .threshold import BaseThreshold, double_threshold_dict, threshold_dict


class StackAlgorithm(SegmentationAlgorithm, ABC):
    def __init__(self):
        super().__init__()
        self.channel_num = 0

    @classmethod
    def support_time(cls):
        return False

    @classmethod
    def support_z(cls):
        return True


class ThresholdPreview(StackAlgorithm):
    @classmethod
    def get_fields(cls):
        return [
            AlgorithmProperty("channel", "Channel", 0, value_type=Channel),
            AlgorithmProperty(
                "noise_filtering",
                "Filter",
                noise_filtering_dict.get_default(),
                possible_values=noise_filtering_dict,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty("threshold", "Threshold", 1000, (0, 10 ** 6), 100),
        ]

    @classmethod
    def get_name(cls):
        return "Only Threshold"

    def __init__(self):
        super().__init__()
        self.noise_filtering = None
        self.threshold = 0

    def calculation_run(self, report_fun) -> SegmentationResult:
        self.channel = self.get_channel(self.channel_num)
        image = noise_filtering_dict[self.noise_filtering["name"]].noise_filter(
            self.channel, self.image.spacing, self.noise_filtering["values"]
        )
        res = (image > self.threshold).astype(np.uint8)
        if self.mask is not None:
            res[self.mask == 0] = 0
        self.image = None
        self.channel = None
        return SegmentationResult(
            roi=res,
            parameters=self.get_segmentation_profile(),
            additional_layers={"denoised image": AdditionalLayerDescription(layer_type="image", data=image)},
        )

    def set_parameters(self, channel, threshold, noise_filtering):  # pylint: disable=W0221
        self.channel_num = channel
        self.threshold = threshold
        self.noise_filtering = noise_filtering

    def get_info_text(self):
        return ""

    def get_segmentation_profile(self) -> ROIExtractionProfile:
        return ROIExtractionProfile(
            "",
            self.get_name(),
            {"channel": self.channel_num, "threshold": self.threshold, "noise_filtering": self.noise_filtering},
        )


class BaseThresholdAlgorithm(StackAlgorithm, ABC):
    @classmethod
    def get_fields(cls):
        return [
            AlgorithmProperty("channel", "Channel", 0, value_type=Channel),
            AlgorithmProperty(
                "noise_filtering",
                "Filter",
                noise_filtering_dict.get_default(),
                possible_values=noise_filtering_dict,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "threshold",
                "Threshold",
                threshold_dict.get_default(),
                possible_values=threshold_dict,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty("close_holes", "Fill holes", True, (True, False)),
            AlgorithmProperty("close_holes_size", "Maximum holes size (px)", 200, (0, 10 ** 5), 10),
            AlgorithmProperty(
                "smooth_border",
                "Smooth borders",
                smooth_dict.get_default(),
                possible_values=smooth_dict,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "side_connection",
                "Side by Side connections",
                False,
                (True, False),
                help_text="During calculation of connected components includes" " only side by side connected pixels",
            ),
            AlgorithmProperty("minimum_size", "Minimum size", 8000, (20, 10 ** 6), 1000),
            AlgorithmProperty("use_convex", "Use convex hull", False, (True, False)),
        ]


class BaseSingleThresholdAlgorithm(BaseThresholdAlgorithm, ABC):
    def __init__(self):
        super().__init__()
        self.threshold = None
        self.minimum_size = None
        self.sizes = None
        self.noise_filtering = None
        self.close_holes = False
        self.close_holes_size = 0
        self.smooth_border = {}
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
        image = noise_filtering_dict[self.noise_filtering["name"]].noise_filter(
            self.channel, self.image.spacing, self.noise_filtering["values"]
        )
        mask = self._threshold_and_exclude(image, report_fun)
        if self.close_holes:
            report_fun("Filing holes", 3)
            mask = close_small_holes(mask, self.close_holes_size)
        report_fun("Smooth border", 4)
        self.segmentation = smooth_dict[self.smooth_border["name"]].smooth(mask, self.smooth_border["values"])

        report_fun("Components calculating", 5)
        self.segmentation = sitk.GetArrayFromImage(
            sitk.RelabelComponent(
                sitk.ConnectedComponent(sitk.GetImageFromArray(self.segmentation), self.edge_connection), 20
            )
        )

        self.sizes = np.bincount(self.segmentation.flat)
        ind = bisect(self.sizes[1:], self.minimum_size, lambda x, y: x > y)
        resp = np.copy(self.segmentation)
        resp[resp > ind] = 0
        if self.use_convex:
            report_fun("convex hull", 6)
            resp = convex_fill(resp)
        report_fun("Calculation done", 7)
        return SegmentationResult(
            roi=self.image.fit_array_to_image(resp),
            parameters=self.get_segmentation_profile(),
            additional_layers={
                "denoised image": AdditionalLayerDescription(data=image, layer_type="image"),
                "no size filtering": AdditionalLayerDescription(data=self.segmentation, layer_type="labels"),
            },
        )

    def _set_parameters(
        self,
        channel,
        threshold,
        minimum_size,
        close_holes,
        smooth_border,
        noise_filtering,
        close_holes_size,
        side_connection,
        use_convex,
    ):
        self.channel_num = channel
        self.threshold = threshold
        self.minimum_size = minimum_size
        self.close_holes = close_holes
        self.smooth_border = smooth_border
        self.noise_filtering = noise_filtering
        self.close_holes_size = close_holes_size
        self.edge_connection = not side_connection
        self.use_convex = use_convex

    def get_segmentation_profile(self) -> ROIExtractionProfile:
        return ROIExtractionProfile(
            "",
            self.get_name(),
            {
                "channel": self.channel_num,
                "threshold": self.threshold,
                "minimum_size": self.minimum_size,
                "close_holes": self.close_holes,
                "smooth_border": self.smooth_border,
                "noise_filtering": self.noise_filtering,
                "close_holes_size": self.close_holes_size,
                "side_connection": not self.edge_connection,
                "use_convex": self.use_convex,
            },
        )


class ThresholdAlgorithm(BaseSingleThresholdAlgorithm):
    @classmethod
    def get_name(cls):
        return "Threshold"

    def _threshold_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        return None

    def _threshold_and_exclude(self, image, report_fun):
        report_fun("Threshold calculation", 1)
        threshold_algorithm: BaseThreshold = threshold_dict[self.threshold["name"]]
        mask, _thr_val = threshold_algorithm.calculate_mask(image, self.mask, self.threshold["values"], operator.ge)
        report_fun("Threshold calculated", 2)
        return mask

    def set_parameters(self, **kwargs):
        super()._set_parameters(**kwargs)

    def get_info_text(self):
        return ""


class ThresholdFlowAlgorithm(BaseThresholdAlgorithm):
    def __init__(self):
        super().__init__()
        self.parameters = {}
        self.sizes = [0]

    @classmethod
    def get_name(cls) -> str:
        return "Threshold Flow"

    @classmethod
    def get_fields(cls):
        return [
            AlgorithmProperty("channel", "Channel", 0, value_type=Channel),
            AlgorithmProperty(
                "noise_filtering",
                "Filter",
                noise_filtering_dict.get_default(),
                possible_values=noise_filtering_dict,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "threshold",
                "Threshold",
                double_threshold_dict.get_default(),
                possible_values=double_threshold_dict,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty("close_holes", "Fill holes", True, (True, False)),
            AlgorithmProperty("close_holes_size", "Maximum holes size (px)", 200, (0, 10 ** 5), 10),
            AlgorithmProperty(
                "smooth_border",
                "Smooth borders",
                smooth_dict.get_default(),
                possible_values=smooth_dict,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "side_connection",
                "Side by Side connections",
                False,
                (True, False),
                help_text="During calculation of connected components includes" " only side by side connected pixels",
            ),
            AlgorithmProperty("minimum_size", "Minimum size", 8000, (20, 10 ** 6), 1000),
            AlgorithmProperty(
                "sprawl_type",
                "Flow type",
                sprawl_dict.get_default(),
                possible_values=sprawl_dict,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty("use_convex", "Use convex hull", False, (True, False)),
        ]

    def calculation_run(self, report_fun: Callable[[str, int], None]) -> SegmentationResult:
        report_fun("Noise removal", 0)
        self.channel = self.get_channel(self.channel_num)
        noise_filtered = noise_filtering_dict[self.parameters["noise_filtering"]["name"]].noise_filter(
            self.channel, self.image.spacing, self.parameters["noise_filtering"]["values"]
        )

        report_fun("Threshold apply", 1)
        mask, thr = double_threshold_dict[self.parameters["threshold"]["name"]].calculate_mask(
            noise_filtered, self.mask, self.parameters["threshold"]["values"], operator.ge
        )
        core_objects = np.array(mask == 2).astype(np.uint8)

        report_fun("Core components calculating", 2)
        core_objects = sitk.GetArrayFromImage(
            sitk.RelabelComponent(
                sitk.ConnectedComponent(sitk.GetImageFromArray(core_objects), not self.parameters["side_connection"]),
                20,
            )
        )
        self.sizes = np.bincount(core_objects.flat)
        ind = bisect(self.sizes[1:], self.parameters["minimum_size"], lambda x, y: x > y)
        core_objects[core_objects > ind] = 0

        if self.parameters["close_holes"]:
            report_fun("Filing holes", 3)
            mask = close_small_holes(mask, self.parameters["close_holes_size"])

        report_fun("Smooth border", 4)
        mask = smooth_dict[self.parameters["smooth_border"]["name"]].smooth(
            mask, self.parameters["smooth_border"]["values"]
        )

        report_fun("Sprawl calculation", 5)
        sprawl_algorithm: BaseWatershed = sprawl_dict[self.parameters["sprawl_type"]["name"]]
        segmentation = sprawl_algorithm.sprawl(
            mask,
            core_objects,
            noise_filtered,
            ind,
            self.image.spacing,
            self.parameters["side_connection"],
            operator.gt,
            self.parameters["sprawl_type"]["values"],
            thr[1],
            thr[0],
        )
        if self.parameters["use_convex"]:
            report_fun("convex hull", 6)
            segmentation = convex_fill(segmentation)
        report_fun("Calculation done", 7)
        return SegmentationResult(
            roi=segmentation,
            parameters=self.get_segmentation_profile(),
            additional_layers={
                "denoised image": AdditionalLayerDescription(data=noise_filtered, layer_type="image"),
                "no size filtering": AdditionalLayerDescription(data=mask, layer_type="labels"),
            },
        )

    @staticmethod
    def get_steps_num():
        return 7

    def get_info_text(self):
        return ""

    def set_parameters(self, **kwargs):
        fields = [x.name for x in self.get_fields() if not isinstance(x, str)]
        # TODO Maybe check inclusion
        if set(fields) != set(kwargs.keys()):
            raise ValueError("Not all fields has provided values")
        for name in fields:
            self.parameters[name] = kwargs[name]

    def get_segmentation_profile(self) -> ROIExtractionProfile:
        return ROIExtractionProfile("", self.get_name(), dict(self.parameters))


class AutoThresholdAlgorithm(BaseSingleThresholdAlgorithm):
    @classmethod
    def get_fields(cls):
        res = super().get_fields()
        res.insert(-1, AlgorithmProperty("suggested_size", "Suggested size", 200000, (0, 10 ** 6), 1000))
        return res

    @classmethod
    def get_name(cls):
        return "Auto Threshold"

    def __init__(self):
        super().__init__()
        self.suggested_size = 0

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
        return mask2

    def _threshold_and_exclude(self, image, report_fun):
        if self.mask is not None:
            report_fun("Components exclusion apply", 1)
            image[self.mask == 0] = 0
        report_fun("Threshold calculation", 2)
        return self._threshold_image(image)

    def set_parameters(self, suggested_size, *args, **kwargs):  # pylint: disable=W0221
        self._set_parameters(*args, **kwargs)
        self.suggested_size = suggested_size

    def get_info_text(self):
        return ""

    def get_segmentation_profile(self):
        resp = super().get_segmentation_profile()
        resp.values["suggested_size"] = self.suggested_size
        return resp


final_algorithm_list = [ThresholdAlgorithm, ThresholdFlowAlgorithm, ThresholdPreview, AutoThresholdAlgorithm]


def close_small_holes(image, max_hole_size):
    if image.dtype == bool:
        image = image.astype(np.uint8)
    if len(image.shape) == 2:
        rev_conn = sitk.ConnectedComponent(sitk.BinaryNot(sitk.GetImageFromArray(image)), True)
        return sitk.GetArrayFromImage(sitk.BinaryNot(sitk.RelabelComponent(rev_conn, max_hole_size)))
    for layer in image:
        rev_conn = sitk.ConnectedComponent(sitk.BinaryNot(sitk.GetImageFromArray(layer)), True)
        layer[...] = sitk.GetArrayFromImage(sitk.BinaryNot(sitk.RelabelComponent(rev_conn, max_hole_size)))
    return image
