import dataclasses
import operator
import typing
from abc import ABC
from collections import defaultdict
from copy import deepcopy

import numpy as np
import SimpleITK

from PartSegCore_compiled_backend.multiscale_opening import PyMSO, calculate_mu_mid

from ..algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty, ROIExtractionProfile
from ..channel_class import Channel
from ..mask_partition_utils import BorderRim as BorderRimBase
from ..mask_partition_utils import MaskDistanceSplit as MaskDistanceSplitBase
from ..project_info import AdditionalLayerDescription
from ..universal_const import Units
from ..utils import bisect
from .algorithm_base import SegmentationAlgorithm, SegmentationLimitException, SegmentationResult
from .mu_mid_point import BaseMuMid, mu_mid_dict
from .noise_filtering import noise_filtering_dict
from .threshold import BaseThreshold, double_threshold_dict, threshold_dict
from .watershed import BaseWatershed, calculate_distances_array, get_neigh, sprawl_dict


def blank_operator(_x, _y):
    raise NotImplementedError()


class RestartableAlgorithm(SegmentationAlgorithm, ABC):
    """
    Base class for restartable segmentation algorithm. The idea is to store two copies
    of algorithm parameters and base on difference check from which point restart the calculation.

    :ivar dict ~.parameters: variable for store last run parameters
    :ivar dict ~.new_parameters: variable for store parameters for next run
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.parameters = defaultdict(lambda: None)
        self.new_parameters = {}

    def set_image(self, image):
        self.parameters = defaultdict(lambda: None)
        super().set_image(image)

    def set_mask(self, mask):
        super().set_mask(mask)
        self.parameters["threshold"] = None

    def get_info_text(self):
        return "No info [Report this ass error]"

    def get_segmentation_profile(self) -> ROIExtractionProfile:
        return ROIExtractionProfile("", self.get_name(), deepcopy(self.new_parameters))

    @classmethod
    def support_time(cls):
        return False

    @classmethod
    def support_z(cls):
        return True


class BorderRim(RestartableAlgorithm):
    """
    This class wrap the :py:class:`PartSegCore.mask_partition_utils.BorderRim``
    class in segmentation algorithm interface. It allow user to check how rim look with given set of parameters
    """

    @classmethod
    def get_name(cls):
        return "Border Rim"

    def __init__(self):
        super().__init__()
        self.distance = 0
        self.units = Units.nm

    @classmethod
    def get_fields(cls):
        return ["Need mask"] + BorderRimBase.get_fields()

    def get_info_text(self):
        if self.mask is None:
            return "Need mask"
        return ""

    def calculation_run(self, _report_fun) -> SegmentationResult:
        if self.mask is not None:
            result = BorderRimBase.border_mask(mask=self.mask, voxel_size=self.image.spacing, **self.new_parameters)
            return SegmentationResult(roi=result, parameters=self.get_segmentation_profile())
        raise SegmentationLimitException("Border Rim needs mask")


class MaskDistanceSplit(RestartableAlgorithm):
    """
    This class wrap the :py:class:`PartSegCore.mask_partition_utils.SplitMaskOnPart`
    class in segmentation algorithm interface. It allow user to check how split look with given set of parameters
    """

    def calculation_run(self, report_fun: typing.Callable[[str, int], None]) -> SegmentationResult:
        if self.mask is not None:
            result = MaskDistanceSplitBase.split(
                mask=self.mask, voxel_size=self.image.voxel_size, **self.new_parameters
            )
            return SegmentationResult(roi=result, parameters=self.get_segmentation_profile())
        raise SegmentationLimitException("Mask Distance Split needs mask")

    @classmethod
    def get_name(cls) -> str:
        return "Mask Distance Splitting"

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return ["Need mask"] + MaskDistanceSplitBase.get_fields()


class ThresholdBaseAlgorithm(RestartableAlgorithm, ABC):
    """
    Base class for most threshold Algorithm implemented in PartSeg analysis.
    Created for reduce code repetition.
    """

    threshold_operator = staticmethod(blank_operator)

    @classmethod
    def get_fields(cls):
        return [
            AlgorithmProperty("channel", "Channel", 0, value_type=Channel),
            AlgorithmProperty(
                "noise_filtering",
                "Filter",
                next(iter(noise_filtering_dict.keys())),
                possible_values=noise_filtering_dict,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty("minimum_size", "Minimum size (px)", 8000, (0, 10 ** 6), 1000),
            AlgorithmProperty(
                "side_connection",
                "Connect only sides",
                False,
                (True, False),
                help_text="During calculation of connected components includes" " only side by side connected pixels",
            ),
        ]

    def __init__(self, **kwargs):
        super().__init__()
        self.cleaned_image = None
        self.threshold_image = None
        self._sizes_array = []
        self.components_num = 0
        self.threshold_info = None
        self.old_threshold_info = None

    def get_additional_layers(
        self, full_segmentation: typing.Optional[np.ndarray] = None
    ) -> typing.Dict[str, AdditionalLayerDescription]:
        """
        Create dict with standard additional layers.

        :param full_segmentation: no size filtering if not `self.segmentation`

        :return:
        """
        if full_segmentation is None:
            full_segmentation = self.segmentation
        return {
            "denoised image": AdditionalLayerDescription(data=self.cleaned_image, layer_type="image"),
            "no size filtering": AdditionalLayerDescription(data=full_segmentation, layer_type="labels"),
        }

    def prepare_result(self, roi: np.ndarray) -> SegmentationResult:
        """
        Collect data for result.

        :param roi: array with segmentation
        :return: algorithm result description
        """
        sizes = np.bincount(roi.flat)
        annotation = {i: {"voxels": size} for i, size in enumerate(sizes[1:], 1) if size > 0}
        return SegmentationResult(
            roi=roi,
            parameters=self.get_segmentation_profile(),
            additional_layers=self.get_additional_layers(),
            roi_annotation=annotation,
        )

    def set_image(self, image):
        super().set_image(image)
        self.threshold_info = None

    def get_info_text(self):
        return f"Threshold: {self.threshold_info}\nSizes: " + ", ".join(
            map(str, self._sizes_array[1 : self.components_num + 1])
        )

    def calculation_run(self, report_fun: typing.Callable[[str, int], typing.Any]) -> SegmentationResult:
        """
        main calculation function

        :param report_fun: function used to trace progress
        """
        self.old_threshold_info = self.threshold_info
        restarted = False
        if self.channel is None or self.parameters["channel"] != self.new_parameters["channel"]:
            self.parameters["channel"] = self.new_parameters["channel"]
            self.channel = self.get_channel(self.new_parameters["channel"])
            restarted = True
        if restarted or self.parameters["noise_filtering"] != self.new_parameters["noise_filtering"]:
            self.parameters["noise_filtering"] = deepcopy(self.new_parameters["noise_filtering"])
            noise_filtering_parameters = self.new_parameters["noise_filtering"]
            self.cleaned_image = noise_filtering_dict[noise_filtering_parameters["name"]].noise_filter(
                self.channel, self.image.spacing, noise_filtering_parameters["values"]
            )
            restarted = True
        if restarted or self.new_parameters["threshold"] != self.parameters["threshold"]:
            if self.parameters["threshold"] is None:
                restarted = True
            self.parameters["threshold"] = deepcopy(self.new_parameters["threshold"])
            self.threshold_image = self._threshold(self.cleaned_image)
            if isinstance(self.threshold_info, (list, tuple)) and (
                self.old_threshold_info is None or self.old_threshold_info[0] != self.threshold_info[0]
            ):
                restarted = True
            elif self.old_threshold_info != self.threshold_info:
                restarted = True
            if self.threshold_image.max() == 0:
                res = self.prepare_result(self.threshold_image.astype(np.uint8))
                info_text = (
                    "Something wrong with chosen threshold. Please check it. "
                    "May be to low or to high. The channel bright range is "
                    f"{self.cleaned_image.min()}-{self.cleaned_image.max()} "
                    f"and chosen threshold is {self.threshold_info}"
                )
                return dataclasses.replace(res, info_text=info_text)
        if restarted or self.new_parameters["side_connection"] != self.parameters["side_connection"]:
            self.parameters["side_connection"] = self.new_parameters["side_connection"]
            connect = SimpleITK.ConnectedComponent(
                SimpleITK.GetImageFromArray(self.threshold_image), not self.new_parameters["side_connection"]
            )
            self.segmentation = SimpleITK.GetArrayFromImage(SimpleITK.RelabelComponent(connect))
            self._sizes_array = np.bincount(self.segmentation.flat)
            restarted = True
        if restarted or self.new_parameters["minimum_size"] != self.parameters["minimum_size"]:
            self.parameters["minimum_size"] = self.new_parameters["minimum_size"]
            minimum_size = self.new_parameters["minimum_size"]
            ind = bisect(self._sizes_array[1:], minimum_size, lambda x, y: x > y)
            finally_segment = np.copy(self.segmentation)
            finally_segment[finally_segment > ind] = 0
            self.components_num = ind
            if ind == 0:
                info_text = (
                    "Please check the minimum size parameter. " f"The biggest element has size {self._sizes_array[1]}"
                )
            else:
                info_text = ""
            res = self.prepare_result(finally_segment)
            return dataclasses.replace(res, info_text=info_text)

    def clean(self):
        super().clean()
        self.parameters = defaultdict(lambda: None)
        self.cleaned_image = None
        self.mask = None

    def _threshold(self, image, thr=None):
        if thr is None:
            thr: BaseThreshold = threshold_dict[self.new_parameters["threshold"]["name"]]
        mask, thr_val = thr.calculate_mask(
            image, self.mask, self.new_parameters["threshold"]["values"], self.threshold_operator
        )
        self.threshold_info = thr_val
        return mask


class OneThresholdAlgorithm(ThresholdBaseAlgorithm, ABC):
    """Base class for PartSeg analysis algorithm which apply one threshold. Created for reduce code repetition."""

    @classmethod
    def get_fields(cls):
        fields = super().get_fields()
        fields.insert(
            2,
            AlgorithmProperty(
                "threshold",
                "Threshold",
                next(iter(threshold_dict.keys())),
                possible_values=threshold_dict,
                value_type=AlgorithmDescribeBase,
            ),
        )
        return fields


class LowerThresholdAlgorithm(OneThresholdAlgorithm):
    """
    Implementation of lower threshold algorithm.
    It has same flow like :py:class:`ThresholdBaseAlgorithm`.
    The area of interest are voxels from filtered channel with value above the given threshold
    """

    threshold_operator = staticmethod(operator.gt)

    @classmethod
    def get_name(cls):
        return "Lower threshold"


class UpperThresholdAlgorithm(OneThresholdAlgorithm):
    """
    Implementation of upper threshold algorithm.
    It has same flow like :py:class:`ThresholdBaseAlgorithm`.
    The area of interest are voxels from filtered channel with value below the given threshold
    """

    threshold_operator = staticmethod(operator.lt)

    @classmethod
    def get_name(cls):
        return "Upper threshold"


class RangeThresholdAlgorithm(ThresholdBaseAlgorithm):
    """
    Implementation of upper threshold algorithm.
    It has same flow like :py:class:`ThresholdBaseAlgorithm`.
    The area of interest are voxels from filtered channel with value between the lower and upper threshold
    """

    def set_parameters(self, **kwargs):
        super().set_parameters(**kwargs)
        self.new_parameters["threshold"] = (
            self.new_parameters["lower_threshold"],
            self.new_parameters["upper_threshold"],
        )

    def get_segmentation_profile(self) -> ROIExtractionProfile:
        resp = super().get_segmentation_profile()
        low, upp = resp.values["threshold"]
        del resp.values["threshold"]
        resp.values["lower_threshold"] = low
        resp.values["upper_threshold"] = upp
        return resp

    def _threshold(self, image, thr=None):
        self.threshold_info = self.new_parameters["threshold"]
        return (
            (image > self.new_parameters["threshold"][0]) * np.array(image < self.new_parameters["threshold"][1])
        ).astype(np.uint8)

    @classmethod
    def get_fields(cls):
        fields = super().get_fields()
        fields.insert(2, AlgorithmProperty("lower_threshold", "Lower threshold", 10000, (0, 10 ** 6), 100))
        fields.insert(3, AlgorithmProperty("upper_threshold", "Upper threshold", 10000, (0, 10 ** 6), 100))
        return fields

    @classmethod
    def get_name(cls):
        return "Range threshold"


class TwoLevelThresholdBaseAlgorithm(ThresholdBaseAlgorithm, ABC):
    def __init__(self):
        super().__init__()
        self.sprawl_area = None

    def _threshold(self, image, thr=None):
        if thr is None:
            thr: BaseThreshold = double_threshold_dict[self.new_parameters["threshold"]["name"]]
        mask, thr_val = thr.calculate_mask(
            image, self.mask, self.new_parameters["threshold"]["values"], self.threshold_operator
        )
        self.threshold_info = thr_val
        self.sprawl_area = (mask >= 1).astype(np.uint8)
        return (mask == 2).astype(np.uint8)


class BaseThresholdFlowAlgorithm(TwoLevelThresholdBaseAlgorithm, ABC):
    @classmethod
    def get_fields(cls):
        fields = super().get_fields()
        fields.insert(
            2,
            AlgorithmProperty(
                "threshold",
                "Threshold",
                next(iter(double_threshold_dict.keys())),
                possible_values=double_threshold_dict,
                value_type=AlgorithmDescribeBase,
            ),
        )
        fields.insert(
            3,
            AlgorithmProperty(
                "sprawl_type",
                "Flow type",
                next(iter(sprawl_dict.keys())),
                possible_values=sprawl_dict,
                value_type=AlgorithmDescribeBase,
            ),
        )
        for i, el in enumerate(fields):
            if el.name == "minimum_size":
                index = i
                break
        else:
            raise ValueError("No minimum size field")
        fields[index] = AlgorithmProperty("minimum_size", "Minimum core\nsize (px)", 8000, (0, 10 ** 6), 1000)
        return fields

    def get_info_text(self):
        return (
            "Threshold: "
            + ", ".join(map(str, self.threshold_info))
            + "\nMid sizes: "
            + ", ".join(map(str, self._sizes_array[1 : self.components_num + 1]))
            + "\nFinal sizes: "
            + ", ".join(map(str, self.final_sizes[1:]))
        )

    def __init__(self):
        super().__init__()
        self.finally_segment = None
        self.final_sizes = []
        self.threshold_info = [None, None]

    def clean(self):
        self.sprawl_area = None
        super().clean()

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
            self.finally_segment = segment_data.roi
            finally_segment = segment_data.roi
            restarted = True

        if (
            restarted
            or self.old_threshold_info[1] != self.threshold_info[1]
            or self.new_parameters["sprawl_type"] != self.parameters["sprawl_type"]
        ):
            if self.threshold_operator(self.threshold_info[1], self.threshold_info[0]):
                self.final_sizes = np.bincount(finally_segment.flat)
                return self.prepare_result(self.finally_segment)
            path_sprawl: BaseWatershed = sprawl_dict[self.new_parameters["sprawl_type"]["name"]]
            self.parameters["sprawl_type"] = self.new_parameters["sprawl_type"]
            new_segment = path_sprawl.sprawl(
                self.sprawl_area,
                np.copy(finally_segment),  # TODO add tests for discover this problem
                self.channel,
                self.components_num,
                self.image.spacing,
                self.new_parameters["side_connection"],
                self.threshold_operator,
                self.new_parameters["sprawl_type"]["values"],
                self.threshold_info[1],
                self.threshold_info[0],
            )
            self.final_sizes = np.bincount(new_segment.flat)
            return SegmentationResult(
                roi=new_segment,
                parameters=self.get_segmentation_profile(),
                additional_layers=self.get_additional_layers(full_segmentation=self.sprawl_area),
                roi_annotation={
                    i: {"core voxels": self._sizes_array[i], "voxels": v} for i, v in enumerate(self.final_sizes[1:], 1)
                },
                alternative_representation={"core_objects": finally_segment},
            )


class LowerThresholdFlowAlgorithm(BaseThresholdFlowAlgorithm):
    threshold_operator = staticmethod(operator.gt)

    @classmethod
    def get_name(cls):
        return "Lower threshold with watershed"


class UpperThresholdFlowAlgorithm(BaseThresholdFlowAlgorithm):
    threshold_operator = staticmethod(operator.lt)

    @classmethod
    def get_name(cls):
        return "Upper threshold with watershed"


class OtsuSegment(RestartableAlgorithm):
    @classmethod
    def get_name(cls):
        return "Multiple Otsu"

    @classmethod
    def get_fields(cls):
        return [
            AlgorithmProperty("channel", "Channel", 0, value_type=Channel),
            AlgorithmProperty(
                "noise_filtering",
                "Noise Removal",
                next(iter(noise_filtering_dict.keys())),
                possible_values=noise_filtering_dict,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty("components", "Number of Components", 2, (0, 100)),
            # AlgorithmProperty("mask", "Use mask in calculation", True),
            AlgorithmProperty("valley", "Valley emphasis", True),
            AlgorithmProperty("hist_num", "Number of histogram bins", 128, (8, 2 ** 16)),
        ]

    def __init__(self):
        super().__init__()
        self._sizes_array = []
        self.threshold_info = []

    def calculation_run(self, report_fun):
        channel = self.get_channel(self.new_parameters["channel"])
        noise_filtering_parameters = self.new_parameters["noise_filtering"]
        cleaned_image = noise_filtering_dict[noise_filtering_parameters["name"]].noise_filter(
            channel, self.image.spacing, noise_filtering_parameters["values"]
        )
        cleaned_image_sitk = SimpleITK.GetImageFromArray(cleaned_image)
        res = SimpleITK.OtsuMultipleThresholds(
            cleaned_image_sitk,
            self.new_parameters["components"],
            0,
            self.new_parameters["hist_num"],
            self.new_parameters["valley"],
        )
        res = SimpleITK.GetArrayFromImage(res)
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
        return SegmentationResult(
            roi=res,
            parameters=self.get_segmentation_profile(),
            additional_layers={"denoised_image": AdditionalLayerDescription(data=cleaned_image, layer_type="image")},
        )

    def get_info_text(self):
        return (
            "Threshold: "
            + ", ".join(map(str, self.threshold_info))
            + "\nSizes: "
            + ", ".join(map(str, self._sizes_array))
        )


class BaseMultiScaleOpening(TwoLevelThresholdBaseAlgorithm, ABC):  # pragma: no cover
    @classmethod
    def get_fields(cls):
        return [
            AlgorithmProperty(
                "threshold",
                "Threshold",
                next(iter(double_threshold_dict.keys())),
                possible_values=double_threshold_dict,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "mu_mid",
                "Mu mid value",
                next(iter(mu_mid_dict.keys())),
                possible_values=mu_mid_dict,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty("step_limits", "Limits of Steps", 100, options_range=(1, 1000), value_type=int),
        ] + super().get_fields()

    def get_info_text(self):
        return (
            "Threshold: "
            + ", ".join(map(str, self.threshold_info))
            + "\nMid sizes: "
            + ", ".join(map(str, self._sizes_array[1 : self.components_num + 1]))
            + "\nFinal sizes: "
            + ", ".join(map(str, self.final_sizes[1:]))
            + f"\nsteps: {self.steps}"
        )

    def __init__(self):
        super().__init__()
        self.finally_segment = None
        self.final_sizes = []
        self.threshold_info = [None, None]
        self.steps = 0
        self.mso = PyMSO()
        self.mso.set_use_background(True)

    def clean(self):
        self.sprawl_area = None
        self.mso = PyMSO()
        self.mso.set_use_background(True)
        super().clean()

    def set_image(self, image):
        super().set_image(image)
        self.threshold_info = [None, None]

    def calculation_run(self, report_fun) -> SegmentationResult:
        if self.new_parameters["side_connection"] != self.parameters["side_connection"]:
            neigh, dist = calculate_distances_array(
                self.image.spacing, get_neigh(self.new_parameters["side_connection"])
            )
            self.mso.set_neighbourhood(neigh, dist)
        segment_data = super().calculation_run(report_fun)
        if segment_data is not None and self.components_num == 0:
            self.final_sizes = []
            return segment_data

        if segment_data is None:
            restarted = False
            finally_segment = np.copy(self.finally_segment)
        else:
            self.finally_segment = segment_data.roi
            finally_segment = segment_data.roi
            if np.max(finally_segment) > 250:
                raise SegmentationLimitException(
                    "Current implementation of MSO do not support more than 250 components"
                )
            components = finally_segment.astype(np.uint8)
            components[components > 0] += 1
            components[self.sprawl_area == 0] = 1
            self.mso.set_components(components, self.components_num)
            restarted = True

        if (
            restarted
            or self.old_threshold_info[1] != self.threshold_info[1]
            or self.new_parameters["mu_mid"] != self.parameters["mu_mid"]
        ):
            if self.threshold_operator(self.threshold_info[1], self.threshold_info[0]):
                self.final_sizes = np.bincount(finally_segment.flat)
                return self.prepare_result(self.finally_segment)
            mu_calc: BaseMuMid = mu_mid_dict[self.new_parameters["mu_mid"]["name"]]
            self.parameters["mu_mid"] = self.new_parameters["mu_mid"]
            sprawl_area = (self.sprawl_area > 0).astype(np.uint8)
            sprawl_area[finally_segment > 0] = 0
            mid_val = mu_calc.value(
                sprawl_area,
                self.channel,
                self.threshold_info[0],
                self.threshold_info[1],
                self.new_parameters["mu_mid"]["values"],
            )
            mu_array = calculate_mu_mid(self.channel, self.threshold_info[0], mid_val, self.threshold_info[1])
            self.mso.set_mu_array(mu_array)
            restarted = True

        if restarted or self.new_parameters["step_limits"] != self.parameters["step_limits"]:
            self.parameters["step_limits"] = self.new_parameters["step_limits"]
            count_steps_factor = 20 if self.image.is_2d else 3
            self.mso.run_MSO(self.new_parameters["step_limits"], count_steps_factor)
            self.steps = self.mso.steps_done()
            new_segment = self.mso.get_result_catted()
            new_segment[new_segment > 0] -= 1
            self.final_sizes = np.bincount(new_segment.flat)
            return self.prepare_result(new_segment)


class LowerThresholdMultiScaleOpening(BaseMultiScaleOpening):
    threshold_operator = staticmethod(operator.gt)

    @classmethod
    def get_name(cls):  # pragma: no cover
        return "Lower threshold MultiScale Opening"


class UpperThresholdMultiScaleOpening(BaseMultiScaleOpening):
    threshold_operator = staticmethod(operator.lt)

    @classmethod
    def get_name(cls):  # pragma: no cover
        return "Upper threshold MultiScale Opening"


final_algorithm_list = [
    LowerThresholdAlgorithm,
    UpperThresholdAlgorithm,
    RangeThresholdAlgorithm,
    LowerThresholdFlowAlgorithm,
    UpperThresholdFlowAlgorithm,  # LowerThresholdMultiScaleOpening,
    # UpperThresholdMultiScaleOpening,
    OtsuSegment,
    BorderRim,
    MaskDistanceSplit,
]
