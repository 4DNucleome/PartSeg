import dataclasses
import operator
import typing
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy

import numpy as np
import SimpleITK
from local_migrator import REGISTER, class_to_str, register_class, rename_key
from pydantic import Field, validator

from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.mask_partition_utils import BorderRim as BorderRimBase
from PartSegCore.mask_partition_utils import MaskDistanceSplit as MaskDistanceSplitBase
from PartSegCore.project_info import AdditionalLayerDescription
from PartSegCore.segmentation.algorithm_base import (
    ROIExtractionAlgorithm,
    ROIExtractionResult,
    SegmentationLimitException,
)
from PartSegCore.segmentation.mu_mid_point import BaseMuMid, MuMidSelection
from PartSegCore.segmentation.noise_filtering import NoiseFilterSelection
from PartSegCore.segmentation.threshold import (
    BaseThreshold,
    DoubleThreshold,
    DoubleThresholdParams,
    DoubleThresholdSelection,
    ManualThreshold,
    RangeThresholdSelection,
    SingleThresholdParams,
    ThresholdSelection,
)
from PartSegCore.segmentation.watershed import BaseWatershed, WatershedSelection, calculate_distances_array, get_neigh
from PartSegCore.universal_const import Units
from PartSegCore.utils import BaseModel, bisect
from PartSegCore_compiled_backend.multiscale_opening import PyMSO, calculate_mu_mid
from PartSegImage import Channel

REQUIRE_MASK_STR = "Need mask"


def blank_operator(_x, _y):
    raise NotImplementedError


class RestartableAlgorithm(ROIExtractionAlgorithm, ABC):
    """
    Base class for restartable segmentation algorithm. The idea is to store two copies
    of algorithm parameters and base on difference check from which point restart the calculation.

    :ivar dict ~.parameters: variable for store last run parameters
    :ivar dict ~.new_parameters: variable for store parameters for next run
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.parameters: typing.Dict[str, typing.Optional[typing.Any]] = defaultdict(lambda: None)
        self.new_parameters = self.__argument_class__() if self.__new_style__ else {}  # pylint: disable=not-callable

    def set_image(self, image):
        self.parameters = defaultdict(lambda: None)
        super().set_image(image)

    def set_mask(self, mask):
        super().set_mask(mask)
        self.parameters["threshold"] = None

    def get_info_text(self):
        return "No info [Report this ass error]"

    def get_segmentation_profile(self) -> ROIExtractionProfile:
        return ROIExtractionProfile(name="", algorithm=self.get_name(), values=deepcopy(self.new_parameters))

    @classmethod
    def support_time(cls):
        return False

    @classmethod
    def support_z(cls):
        return True

    @abstractmethod
    def calculation_run(self, report_fun: typing.Callable[[str, int], None]) -> typing.Optional[ROIExtractionResult]:
        """Restartable calculation may return None if there is no need to recalculate"""
        raise NotImplementedError


class BorderRimParameters(BorderRimBase.__argument_class__):
    @staticmethod
    def header():
        return REQUIRE_MASK_STR


class BorderRim(RestartableAlgorithm):
    """
    This class wrap the :py:class:`PartSegCore.mask_partition_utils.BorderRim``
    class in segmentation algorithm interface. It allow user to check how rim look with given set of parameters
    """

    __argument_class__ = BorderRimParameters

    @classmethod
    def get_name(cls):
        return "Border Rim"

    def __init__(self):
        super().__init__()
        self.distance = 0
        self.units = Units.nm

    def get_info_text(self):
        return REQUIRE_MASK_STR if self.mask is None else ""

    def calculation_run(self, _report_fun) -> ROIExtractionResult:
        if self.mask is not None:
            result = BorderRimBase.border_mask(
                mask=self.mask, voxel_size=self.image.spacing, **self.new_parameters.dict()
            )
            return ROIExtractionResult(roi=result, parameters=self.get_segmentation_profile())
        raise SegmentationLimitException("Border Rim needs mask")


class MaskDistanceSplitParameters(MaskDistanceSplitBase.__argument_class__):
    @staticmethod
    def header():
        return REQUIRE_MASK_STR


class MaskDistanceSplit(RestartableAlgorithm):
    """
    This class wrap the :py:class:`PartSegCore.mask_partition_utils.SplitMaskOnPart`
    class in segmentation algorithm interface. It allow user to check how split look with given set of parameters
    """

    __argument_class__ = MaskDistanceSplitParameters

    def calculation_run(self, report_fun: typing.Callable[[str, int], None]) -> ROIExtractionResult:
        if self.mask is not None:
            result = MaskDistanceSplitBase.split(
                mask=self.mask, voxel_size=self.image.voxel_size, **self.new_parameters.dict()
            )
            return ROIExtractionResult(roi=result, parameters=self.get_segmentation_profile())
        raise SegmentationLimitException("Mask Distance Split needs mask")

    @classmethod
    def get_name(cls) -> str:
        return "Mask Distance Splitting"


@register_class(version="0.0.1", migrations=[("0.0.1", rename_key("noise_removal", "noise_filtering", optional=True))])
class ThresholdBaseAlgorithmParameters(BaseModel):
    channel: Channel = Channel(0)
    noise_filtering: NoiseFilterSelection = Field(NoiseFilterSelection.get_default(), title="Filter")
    minimum_size: int = Field(8000, title="Minimum size (px)", ge=0, le=10**6)
    side_connection: bool = Field(
        False,
        title="Connect only sides",
        description="During calculation of connected components includes only side by side connected pixels",
    )

    @validator("noise_filtering")
    def _noise_filter_validate(cls, v):  # pylint: disable=no-self-use
        if not isinstance(v, dict):
            return v
        algorithm = NoiseFilterSelection[v["name"]]
        if not algorithm.__new_style__ or not algorithm.__argument_class__.__fields__:
            return v
        return algorithm.__argument_class__(**REGISTER.migrate_data(class_to_str(algorithm.__argument_class__), {}, v))


class ThresholdBaseAlgorithmParametersAnnot(ThresholdBaseAlgorithmParameters):
    threshold: typing.Any = None


class ThresholdBaseAlgorithm(RestartableAlgorithm, ABC):
    """
    Base class for most threshold Algorithm implemented in PartSeg analysis.
    Created for reduce code repetition.
    """

    __argument_class__ = ThresholdBaseAlgorithmParameters

    new_parameters: ThresholdBaseAlgorithmParametersAnnot

    threshold_operator = staticmethod(blank_operator)

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

    def prepare_result(self, roi: np.ndarray) -> ROIExtractionResult:
        """
        Collect data for result.

        :param roi: array with segmentation
        :return: algorithm result description
        """
        sizes = np.bincount(roi.flat)
        annotation = {i: {"component": i, "voxels": size} for i, size in enumerate(sizes[1:], 1) if size > 0}
        return ROIExtractionResult(
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

    def _lack_of_components(self):
        res = self.prepare_result(self.threshold_image.astype(np.uint8))
        info_text = (
            "Something wrong with chosen threshold. Please check it. "
            "May be too low or too high. The channel brightness range is "
            f"{self.cleaned_image.min()}-{self.cleaned_image.max()} "
            f"and chosen threshold is {self.threshold_info}"
        )
        return dataclasses.replace(res, info_text=info_text)

    def _get_channel(self) -> bool:
        """Get channel from image if number of channel is changed from previous run, or image is changed"""
        if self.channel is None or self.parameters["channel"] != self.new_parameters.channel:
            self.parameters["channel"] = self.new_parameters.channel
            self.channel = self.get_channel(self.new_parameters.channel)
            return True
        return False

    def _update_cleaned_image(self, restarted: bool) -> bool:
        """Update cleaned image if selected channel or or noise filter is changed"""
        if restarted or self.parameters["noise_filtering"] != self.new_parameters.noise_filtering:
            self.parameters["noise_filtering"] = deepcopy(self.new_parameters.noise_filtering)
            noise_filtering_parameters = self.new_parameters.noise_filtering
            self.cleaned_image = NoiseFilterSelection[noise_filtering_parameters.name].noise_filter(
                self.channel, self.image.spacing, noise_filtering_parameters.values
            )
            return True
        return False

    def _calculate_threshold(self, restarted: bool):
        """Calculate threshold if cleaned image is changed"""
        if restarted or self.new_parameters.threshold != self.parameters["threshold"]:
            self.parameters["threshold"] = deepcopy(self.new_parameters.threshold)
            self.threshold_image = self._threshold(self.cleaned_image)
            return True
        return False

    def _calculate_components(self, restarted: bool):
        """Calculate components if threshold image is changed"""
        if restarted or self.new_parameters.side_connection != self.parameters["side_connection"]:
            self.parameters["side_connection"] = self.new_parameters.side_connection
            connect = SimpleITK.ConnectedComponent(
                SimpleITK.GetImageFromArray(self.threshold_image), not self.new_parameters.side_connection
            )
            self.segmentation = SimpleITK.GetArrayFromImage(SimpleITK.RelabelComponent(connect))
            self._sizes_array = np.bincount(self.segmentation.flat)
            return True
        return False

    def _filter_by_size(self, restarted: bool) -> typing.Optional[np.ndarray]:
        """Filter components by size if size filter is changed"""
        if restarted or self.new_parameters.minimum_size != self.parameters["size_filter"]:
            self.parameters["minimum_size"] = self.new_parameters.minimum_size
            minimum_size = self.new_parameters.minimum_size
            ind = bisect(self._sizes_array[1:], minimum_size, operator.gt)
            finally_segment = np.copy(self.segmentation)
            finally_segment[finally_segment > ind] = 0
            self.components_num = ind
            return finally_segment
        return None

    def calculation_run(
        self, report_fun: typing.Callable[[str, int], typing.Any]
    ) -> typing.Optional[ROIExtractionResult]:
        """
        main calculation function

        :param report_fun: function used to trace progress
        """
        # TODO Refactor
        self.old_threshold_info = self.threshold_info
        restarted = self._get_channel()
        restarted = self._update_cleaned_image(restarted)
        restarted = self._calculate_threshold(restarted)
        if self.threshold_image.max() == 0:
            return self._lack_of_components()

        restarted = self._calculate_components(restarted)
        if len(self._sizes_array) < 2:
            return self._lack_of_components()

        finally_segment = self._filter_by_size(restarted)

        if finally_segment is not None:
            if self.components_num == 0:
                info_text = (
                    f"Please check the minimum size parameter. The biggest element has size {self._sizes_array[1]}"
                )
            else:
                info_text = ""
            res = self.prepare_result(finally_segment)
            return dataclasses.replace(res, info_text=info_text)

        return None

    def clean(self):
        super().clean()
        self.parameters: typing.Dict[str, typing.Optional[typing.Any]] = defaultdict(lambda: None)
        self.cleaned_image = None
        self.mask = None

    def _threshold(self, image, thr=None):
        if thr is None:
            thr: BaseThreshold = ThresholdSelection[self.new_parameters.threshold.name]
        mask, thr_val = thr.calculate_mask(
            image, self.mask, self.new_parameters.threshold.values, self.threshold_operator
        )
        self.threshold_info = thr_val
        return mask


class OneThresholdAlgorithmParameters(ThresholdBaseAlgorithmParameters):
    threshold: ThresholdSelection = Field(ThresholdSelection.get_default(), position=2)


class OneThresholdAlgorithm(ThresholdBaseAlgorithm, ABC):
    """Base class for PartSeg analysis algorithm which apply one threshold. Created for reduce code repetition."""

    __argument_class__ = OneThresholdAlgorithmParameters


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


class TwoThreshold(BaseModel):
    # keep for backward compatibility
    lower_threshold: float = Field(1000, ge=0, le=10**6)
    upper_threshold: float = Field(10000, ge=0, le=10**6)


def _to_two_thresholds(dkt):
    dkt["threshold"] = TwoThreshold(
        lower_threshold=dkt.pop("lower_threshold"), upper_threshold=dkt.pop("upper_threshold")
    )
    return dkt


def _to_double_threshold(dkt):
    dkt["threshold"] = DoubleThresholdSelection(
        name=DoubleThreshold.get_name(),
        values=DoubleThresholdParams(
            core_threshold=ThresholdSelection(
                name=ManualThreshold.get_name(),
                values=SingleThresholdParams(threshold=dkt["threshold"].lower_threshold),
            ),
            base_threshold=ThresholdSelection(
                name=ManualThreshold.get_name(),
                values=SingleThresholdParams(threshold=dkt["threshold"].upper_threshold),
            ),
        ),
    )
    return dkt


def _rename_algorithm(dkt):
    values = dkt["threshold"].values
    name = dkt["threshold"].name
    if name == "Base/Core":
        name = "Range"
    dkt["threshold"] = RangeThresholdSelection(name=name, values=values)

    return dkt


@register_class(
    version="0.0.3",
    migrations=[("0.0.1", _to_two_thresholds), ("0.0.2", _to_double_threshold), ("0.0.3", _rename_algorithm)],
)
class RangeThresholdAlgorithmParameters(ThresholdBaseAlgorithmParameters):
    threshold: RangeThresholdSelection = Field(default_factory=RangeThresholdSelection.get_default, position=2)


class RangeThresholdAlgorithm(ThresholdBaseAlgorithm):
    """
    Implementation of upper threshold algorithm.
    It has same flow like :py:class:`ThresholdBaseAlgorithm`.
    The area of interest are voxels from filtered channel with value between the lower and upper threshold
    """

    __argument_class__ = RangeThresholdAlgorithmParameters

    def _threshold(self, image, thr=None):
        if thr is None:
            thr: BaseThreshold = RangeThresholdSelection[self.new_parameters.threshold.name]
        mask, thr_val = thr.calculate_mask(image, self.mask, self.new_parameters.threshold.values, operator.ge)
        mask[mask == 2] = 0
        self.threshold_info = thr_val[::-1]
        return mask

    @classmethod
    def get_name(cls):
        return "Range threshold"


class TwoLevelThresholdBaseAlgorithm(ThresholdBaseAlgorithm, ABC):
    def __init__(self):
        super().__init__()
        self.sprawl_area = None
        self._original_output = None

    def _threshold(self, image, thr=None):
        if thr is None:
            thr: BaseThreshold = DoubleThresholdSelection[self.new_parameters.threshold.name]
        mask, thr_val = thr.calculate_mask(
            image, self.mask, self.new_parameters.threshold.values, self.threshold_operator
        )
        self.threshold_info = thr_val
        self.sprawl_area = (mask >= 1).astype(np.uint8)
        self._original_output = mask
        return (mask == 2).astype(np.uint8)


@register_class(version="0.0.1", migrations=[("0.0.1", rename_key("sprawl_type", "flow_type"))])
class BaseThresholdFlowAlgorithmParameters(ThresholdBaseAlgorithmParameters):
    threshold: DoubleThresholdSelection = Field(DoubleThresholdSelection.get_default(), position=2)
    flow_type: WatershedSelection = Field(WatershedSelection.get_default(), position=3)
    minimum_size: int = Field(8000, title="Minimum core\nsize (px)", ge=0, le=10**6)
    remove_object_touching_border: bool = Field(
        False, title="Remove objects\ntouching border", description="Remove objects touching border"
    )


def remove_object_touching_border(new_segment):
    non_one_dims = np.where(np.array(new_segment.shape) > 1)[0]
    slice_list = [slice(None)] * len(new_segment.shape)
    to_remove = set()
    for dim in non_one_dims:
        slice_copy = slice_list[:]
        slice_copy[dim] = 0
        to_remove.update(np.unique(new_segment[tuple(slice_copy)]))
        slice_copy[dim] = new_segment.shape[dim] - 1
        to_remove.update(np.unique(new_segment[tuple(slice_copy)]))

    res = np.copy(new_segment)
    for i in to_remove:
        if i == 0:
            continue
        res[res == i] = 0
    return res


class BaseThresholdFlowAlgorithm(TwoLevelThresholdBaseAlgorithm, ABC):
    __argument_class__ = BaseThresholdFlowAlgorithmParameters
    new_parameters: BaseThresholdFlowAlgorithmParameters

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

    def calculation_run(self, report_fun) -> typing.Optional[ROIExtractionResult]:
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
            or self.new_parameters.flow_type != self.parameters["flow_type"]
            or self.new_parameters.remove_object_touching_border != self.parameters["remove_object_touching_border"]
        ):
            if self.threshold_operator(self.threshold_info[1], self.threshold_info[0]):
                self.final_sizes = np.bincount(finally_segment.flat)
                return self.prepare_result(self.finally_segment)
            path_sprawl: BaseWatershed = WatershedSelection[self.new_parameters.flow_type.name]
            self.parameters["flow_type"] = self.new_parameters.flow_type
            new_segment = path_sprawl.sprawl(
                self.sprawl_area,
                np.copy(finally_segment),  # TODO add tests for discover this problem
                self.channel,
                self.components_num,
                self.image.spacing,
                self.new_parameters.side_connection,
                self.threshold_operator,
                self.new_parameters.flow_type.values,
                self.threshold_info[1],
                self.threshold_info[0],
            )
            if self.new_parameters.remove_object_touching_border:
                new_segment = remove_object_touching_border(new_segment)

            self.parameters["remove_object_touching_border"] = self.new_parameters.remove_object_touching_border

            self.final_sizes = np.bincount(new_segment.flat)
            return ROIExtractionResult(
                roi=new_segment,
                parameters=self.get_segmentation_profile(),
                additional_layers={
                    "original": AdditionalLayerDescription(data=self._original_output, layer_type="labels"),
                    **self.get_additional_layers(full_segmentation=self.sprawl_area),
                },
                roi_annotation={
                    i: {"component": i, "core voxels": self._sizes_array[i], "voxels": v}
                    for i, v in enumerate(self.final_sizes[1:], 1)
                },
                alternative_representation={"core_objects": finally_segment},
            )
        return None


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


@register_class(version="0.0.1", migrations=[("0.0.1", rename_key("noise_removal", "noise_filtering", optional=True))])
class OtsuSegmentParameters(BaseModel):
    channel: Channel = 0
    noise_filtering: NoiseFilterSelection = Field(NoiseFilterSelection.get_default(), title="Noise Removal")
    components: int = Field(2, title="Number of Components", ge=0, lt=100)
    valley: bool = Field(True, title="Valley emphasis")
    hist_num: int = Field(128, title="Number of histogram bins", ge=8, le=2**16)


class OtsuSegment(RestartableAlgorithm):
    __argument_class__ = OtsuSegmentParameters
    new_parameters: OtsuSegmentParameters

    @classmethod
    def get_name(cls):
        return "Multiple Otsu"

    def __init__(self):
        super().__init__()
        self._sizes_array = []
        self.threshold_info = []

    def calculation_run(self, report_fun):
        channel = self.get_channel(self.new_parameters.channel)
        noise_filtering_parameters = self.new_parameters.noise_filtering
        cleaned_image = NoiseFilterSelection[noise_filtering_parameters.name].noise_filter(
            channel, self.image.spacing, noise_filtering_parameters.values
        )
        cleaned_image_sitk = SimpleITK.GetImageFromArray(cleaned_image)
        res = SimpleITK.OtsuMultipleThresholds(
            cleaned_image_sitk,
            self.new_parameters.components,
            0,
            self.new_parameters.hist_num,
            self.new_parameters.valley,
        )
        res = SimpleITK.GetArrayFromImage(res)
        self._sizes_array = np.bincount(res.flat)[1:]
        self.threshold_info = []
        annotations = {}
        for i in range(1, self.new_parameters.components + 1):
            val = cleaned_image[res == i]
            if val.size:
                self.threshold_info.append(np.min(val))
            elif self.threshold_info:
                self.threshold_info.append(self.threshold_info[-1])
            else:
                self.threshold_info.append(0)
            annotations[i] = {"lower threshold": self.threshold_info[-1]}
            if i > 1:
                annotations[i - 1]["upper threshold"] = self.threshold_info[-1]
        annotations[self.new_parameters.components]["upper threshold"] = np.max(cleaned_image)
        return ROIExtractionResult(
            roi=res,
            parameters=self.get_segmentation_profile(),
            additional_layers={"denoised_image": AdditionalLayerDescription(data=cleaned_image, layer_type="image")},
            roi_annotation=annotations,
        )

    def get_info_text(self):
        return (
            "Threshold: "
            + ", ".join(map(str, self.threshold_info))
            + "\nSizes: "
            + ", ".join(map(str, self._sizes_array))
        )


class BaseMultiScaleOpeningParameters(TwoLevelThresholdBaseAlgorithm.__argument_class__):
    threshold: DoubleThresholdSelection = Field(DoubleThresholdSelection.get_default())
    mu_mid: MuMidSelection = Field(MuMidSelection.get_default(), title="Mu mid value")
    step_limits: int = Field(100, title="Limits of Steps", ge=1, le=1000)


class BaseMultiScaleOpening(TwoLevelThresholdBaseAlgorithm, ABC):  # pragma: no cover
    __argument_class__ = BaseMultiScaleOpeningParameters
    new_parameters: BaseMultiScaleOpeningParameters

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
        self.threshold_info = [float("nan"), float("nan")]
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
        self.threshold_info = [float("nan"), float("nan")]

    def calculation_run(self, report_fun) -> typing.Optional[ROIExtractionResult]:
        if self.new_parameters.side_connection != self.parameters["side_connection"]:
            neigh, dist = calculate_distances_array(self.image.spacing, get_neigh(self.new_parameters.side_connection))
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
            or self.new_parameters.mu_mid != self.parameters["mu_mid"]
        ):
            if self.threshold_operator(self.threshold_info[1], self.threshold_info[0]):
                self.final_sizes = np.bincount(finally_segment.flat)
                return self.prepare_result(self.finally_segment)
            mu_calc: BaseMuMid = MuMidSelection[self.new_parameters.mu_mid.name]
            self.parameters["mu_mid"] = self.new_parameters.mu_mid
            sprawl_area = (self.sprawl_area > 0).astype(np.uint8)
            sprawl_area[finally_segment > 0] = 0
            mid_val = mu_calc.value(
                sprawl_area,
                self.channel,
                self.threshold_info[0],
                self.threshold_info[1],
                self.new_parameters.mu_mid.values,
            )
            mu_array = calculate_mu_mid(self.channel, self.threshold_info[0], mid_val, self.threshold_info[1])
            self.mso.set_mu_array(mu_array)
            restarted = True

        if restarted or self.new_parameters.step_limits != self.parameters["step_limits"]:
            self.parameters["step_limits"] = self.new_parameters.step_limits
            count_steps_factor = 20 if self.image.is_2d else 3
            self.mso.run_MSO(self.new_parameters.step_limits, count_steps_factor)
            self.steps = self.mso.steps_done()
            new_segment = self.mso.get_result_catted()
            new_segment[new_segment > 0] -= 1
            self.final_sizes = np.bincount(new_segment.flat)
            return self.prepare_result(new_segment)
        return None


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
    UpperThresholdFlowAlgorithm,
    OtsuSegment,
    BorderRim,
    MaskDistanceSplit,
]
