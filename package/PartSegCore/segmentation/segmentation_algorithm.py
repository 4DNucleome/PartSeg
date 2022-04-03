import operator
from abc import ABC
from typing import Callable, Optional

import numpy as np
import SimpleITK as sitk
from nme import register_class, rename_key
from pydantic import Field

from PartSegCore.utils import BaseModel
from PartSegImage import Channel

from ..convex_fill import convex_fill
from ..project_info import AdditionalLayerDescription
from ..segmentation.algorithm_base import ROIExtractionAlgorithm, ROIExtractionResult
from ..utils import bisect
from .border_smoothing import NoneSmoothing, OpeningSmoothing, SmoothAlgorithmSelection
from .noise_filtering import NoiseFilterSelection
from .threshold import BaseThreshold, DoubleThresholdSelection, ThresholdSelection
from .watershed import BaseWatershed, FlowMethodSelection


class StackAlgorithm(ROIExtractionAlgorithm, ABC):
    def __init__(self):
        super().__init__()
        self.channel_num = 0

    @classmethod
    def support_time(cls):
        return False

    @classmethod
    def support_z(cls):
        return True

    def get_noise_filtered_channel(self, channel_idx, noise_removal):
        channel = self.get_channel(channel_idx)
        return NoiseFilterSelection[noise_removal.name].noise_filter(channel, self.image.spacing, noise_removal.values)


@register_class(version="0.0.1", migrations=[("0.0.1", rename_key("noise_removal", "noise_filtering", optional=True))])
class ThresholdPreviewParameters(BaseModel):
    channel: Channel = 0
    noise_filtering: NoiseFilterSelection = Field(NoiseFilterSelection.get_default(), title="Filter")
    threshold: int = Field(1000, ge=0, le=10**6)


class ThresholdPreview(StackAlgorithm):
    __argument_class__ = ThresholdPreviewParameters

    new_parameters: ThresholdPreviewParameters

    @classmethod
    def get_name(cls):
        return "Only Threshold"

    def calculation_run(self, report_fun) -> ROIExtractionResult:
        image = self.get_noise_filtered_channel(self.new_parameters.channel, self.new_parameters.noise_filtering)
        report_fun("threshold", 0)
        res = (image > self.new_parameters.threshold).astype(np.uint8)
        report_fun("mask", 1)
        if self.mask is not None:
            res[self.mask == 0] = 0
        report_fun("result", 2)
        val = ROIExtractionResult(
            roi=res,
            parameters=self.get_segmentation_profile(),
            additional_layers={"denoised image": AdditionalLayerDescription(layer_type="image", data=image)},
        )
        report_fun("return", 4)
        return val

    def get_info_text(self):
        return ""

    @staticmethod
    def get_steps_num():
        return 3


def _migrate_smooth_border(dkt: dict):
    if isinstance(dkt["smooth_border"], bool):
        dkt = dkt.copy()
        if dkt["smooth_border"] and "smooth_border_radius" in dkt:
            dkt["smooth_border"] = SmoothAlgorithmSelection(
                name=OpeningSmoothing.get_name(),
                values=OpeningSmoothing.__argument_class__(smooth_border_radius=dkt.pop("smooth_border_radius")),
            )
        else:
            dkt["smooth_border"] = SmoothAlgorithmSelection(
                name=NoneSmoothing.get_name(), values=NoneSmoothing.__argument_class__()
            )
        if "smooth_border_radius" in dkt:
            del dkt["smooth_border_radius"]
    return dkt


@register_class(
    version="0.0.2",
    migrations=[
        ("0.0.1", _migrate_smooth_border),
        ("0.0.2", rename_key("noise_removal", "noise_filtering", optional=True)),
    ],
)
class BaseThresholdAlgorithmParameters(BaseModel):
    channel: Channel = 0
    noise_filtering: NoiseFilterSelection = Field(NoiseFilterSelection.get_default(), title="Filter")
    threshold: ThresholdSelection = Field(ThresholdSelection.get_default(), title="Threshold")
    close_holes: bool = Field(True, title="Fill holes")
    close_holes_size: int = Field(200, title="Maximum holes size (px)", ge=0, le=10**5)
    smooth_border: SmoothAlgorithmSelection = Field(SmoothAlgorithmSelection.get_default(), title="Smooth borders")
    side_connection: bool = Field(
        False,
        title="Side by Side connections",
        description="During calculation of connected components includes only side by side connected pixels",
    )
    minimum_size: int = Field(8000, ge=20, le=10**6)
    use_convex: int = Field(False, title="Use convex hull")


class BaseThresholdAlgorithm(StackAlgorithm, ABC):
    __argument_class__ = BaseThresholdAlgorithmParameters

    new_parameters: BaseThresholdAlgorithmParameters

    def __init__(self):
        super().__init__()
        self.sizes = [0]

    def get_info_text(self):
        if len(self.sizes) > 1:
            return f"ROI sizes: {', '.join(map(str, self.sizes[1:]))}"
        return ""


class MorphologicalWatershed(BaseThresholdAlgorithm):
    def __init__(self):
        super().__init__()
        self.base_sizes = [0]

    @classmethod
    def get_name(cls):
        return "Morphological Watersheed"

    @staticmethod
    def get_steps_num():
        return 7

    def _threshold_and_exclude(self, image, report_fun):
        report_fun("Threshold calculation", 1)
        threshold_algorithm: BaseThreshold = ThresholdSelection[self.new_parameters.threshold.name]
        mask, _thr_val = threshold_algorithm.calculate_mask(
            image, self.mask, self.new_parameters.threshold.values, operator.ge
        )
        report_fun("Threshold calculated", 2)
        return mask

    def calculation_run(self, report_fun):
        report_fun("Noise removal", 0)
        image = self.get_noise_filtered_channel(self.new_parameters.channel, self.new_parameters.noise_filtering)
        mask = self._threshold_and_exclude(image, report_fun)
        if self.new_parameters.close_holes:
            report_fun("Filing holes", 3)
            mask = close_small_holes(mask, self.new_parameters.close_holes_size)
        report_fun("Smooth border", 4)
        self.segmentation = SmoothAlgorithmSelection[self.new_parameters.smooth_border.name].smooth(
            mask, self.new_parameters.smooth_border.values
        )

        report_fun("Components calculating", 5)
        seg_image = sitk.GetImageFromArray(self.segmentation)
        distance_map = sitk.SignedMaurerDistanceMap(
            seg_image, insideIsPositive=False, squaredDistance=False, useImageSpacing=False
        )

        ws = sitk.MorphologicalWatershed(distance_map, markWatershedLine=False, level=1)
        self.segmentation = sitk.GetArrayFromImage(
            sitk.RelabelComponent(sitk.Mask(ws, sitk.Cast(seg_image, ws.GetPixelID())), 20)
        )

        self.base_sizes = np.bincount(self.segmentation.flat)
        ind = bisect(self.base_sizes[1:], self.new_parameters.minimum_size, lambda x, y: x > y)
        resp = np.copy(self.segmentation)
        resp[resp > ind] = 0

        if len(self.base_sizes) == 1:
            info_text = "Please check the threshold parameter. There is no object bigger than 20 voxels."
        elif ind == 0:
            info_text = f"Please check the minimum size parameter. The biggest element has size {self.base_sizes[1]}"
        else:
            info_text = ""
        self.sizes = self.base_sizes[: ind + 1]
        if self.new_parameters.use_convex:
            report_fun("convex hull", 6)
            resp = convex_fill(resp)
            self.sizes = np.bincount(resp.flat)

        report_fun("Calculation done", 7)
        return ROIExtractionResult(
            roi=self.image.fit_array_to_image(resp),
            parameters=self.get_segmentation_profile(),
            additional_layers={
                "denoised image": AdditionalLayerDescription(data=image, layer_type="image"),
                "no size filtering": AdditionalLayerDescription(data=self.segmentation, layer_type="labels"),
            },
            info_text=info_text,
            roi_annotation={i: {"voxels": v} for i, v in enumerate(self.sizes[1:], start=1)},
        )

    def get_info_text(self):
        base_text = super().get_info_text()
        base_sizes = self.base_sizes[: self.sizes.size]
        if np.any(base_sizes != self.sizes):
            base_text += "\nBase ROI sizes " + ", ".join(map(str, base_sizes))
        return base_text


class BaseSingleThresholdAlgorithm(BaseThresholdAlgorithm, ABC):
    def __init__(self):
        super().__init__()
        self.base_sizes = [0]

    @staticmethod
    def get_steps_num():
        return 7

    def _threshold_image(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def _threshold_and_exclude(self, image, report_fun):
        raise NotImplementedError()

    def calculation_run(self, report_fun):
        report_fun("Noise removal", 0)
        image = self.get_noise_filtered_channel(self.new_parameters.channel, self.new_parameters.noise_filtering)
        mask = self._threshold_and_exclude(image, report_fun)
        if self.new_parameters.close_holes:
            report_fun("Filing holes", 3)
            mask = close_small_holes(mask, self.new_parameters.close_holes_size)
        report_fun("Smooth border", 4)
        self.segmentation = SmoothAlgorithmSelection[self.new_parameters.smooth_border.name].smooth(
            mask, self.new_parameters.smooth_border.values
        )

        report_fun("Components calculating", 5)
        self.segmentation = sitk.GetArrayFromImage(
            sitk.RelabelComponent(
                sitk.ConnectedComponent(
                    sitk.GetImageFromArray(self.segmentation), not self.new_parameters.side_connection
                ),
                20,
            )
        )

        self.base_sizes = np.bincount(self.segmentation.flat)
        ind = bisect(self.base_sizes[1:], self.new_parameters.minimum_size, lambda x, y: x > y)
        resp = np.copy(self.segmentation)
        resp[resp > ind] = 0

        if len(self.base_sizes) == 1:
            info_text = "Please check the threshold parameter. There is no object bigger than 20 voxels."
        elif ind == 0:
            info_text = f"Please check the minimum size parameter. The biggest element has size {self.base_sizes[1]}"
        else:
            info_text = ""
        self.sizes = self.base_sizes[: ind + 1]
        if self.new_parameters.use_convex:
            report_fun("convex hull", 6)
            resp = convex_fill(resp)
            self.sizes = np.bincount(resp.flat)

        report_fun("Calculation done", 7)
        return ROIExtractionResult(
            roi=self.image.fit_array_to_image(resp),
            parameters=self.get_segmentation_profile(),
            additional_layers={
                "denoised image": AdditionalLayerDescription(data=image, layer_type="image"),
                "no size filtering": AdditionalLayerDescription(data=self.segmentation, layer_type="labels"),
            },
            info_text=info_text,
            roi_annotation={i: {"voxels": v} for i, v in enumerate(self.sizes[1:], start=1)},
        )

    def get_info_text(self):
        base_text = super().get_info_text()
        base_sizes = self.base_sizes[: self.sizes.size]
        if np.any(base_sizes != self.sizes):
            base_text += "\nBase ROI sizes " + ", ".join(map(str, base_sizes))
        return base_text


class ThresholdAlgorithm(BaseSingleThresholdAlgorithm):
    @classmethod
    def get_name(cls):
        return "Threshold"

    def _threshold_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        return None

    def _threshold_and_exclude(self, image, report_fun):
        report_fun("Threshold calculation", 1)
        threshold_algorithm: BaseThreshold = ThresholdSelection[self.new_parameters.threshold.name]
        mask, _thr_val = threshold_algorithm.calculate_mask(
            image, self.mask, self.new_parameters.threshold.values, operator.ge
        )
        report_fun("Threshold calculated", 2)
        return mask


@register_class(version="0.0.1", migrations=[("0.0.1", rename_key("sprawl_type", "flow_type"))])
class ThresholdFlowAlgorithmParameters(BaseThresholdAlgorithmParameters):
    threshold: DoubleThresholdSelection = Field(DoubleThresholdSelection.get_default())
    flow_type: FlowMethodSelection = Field(FlowMethodSelection.get_default())


class ThresholdFlowAlgorithm(BaseThresholdAlgorithm):
    __argument_class__ = ThresholdFlowAlgorithmParameters

    new_parameters: ThresholdFlowAlgorithmParameters

    @classmethod
    def get_name(cls) -> str:
        return "Threshold Flow"

    def calculation_run(self, report_fun: Callable[[str, int], None]) -> ROIExtractionResult:
        report_fun("Noise removal", 0)
        noise_filtered = self.get_noise_filtered_channel(
            self.new_parameters.channel, self.new_parameters.noise_filtering
        )

        report_fun("Threshold apply", 1)
        mask, thr = DoubleThresholdSelection[self.new_parameters.threshold.name].calculate_mask(
            noise_filtered, self.mask, self.new_parameters.threshold.values, operator.ge
        )
        core_objects = np.array(mask == 2).astype(np.uint8)

        report_fun("Core components calculating", 2)
        core_objects = sitk.GetArrayFromImage(
            sitk.RelabelComponent(
                sitk.ConnectedComponent(sitk.GetImageFromArray(core_objects), not self.new_parameters.side_connection),
                20,
            )
        )
        self.base_sizes = np.bincount(core_objects.flat)
        ind = bisect(self.base_sizes[1:], self.new_parameters.minimum_size, lambda x, y: x > y)
        core_objects[core_objects > ind] = 0

        if self.new_parameters.close_holes:
            report_fun("Filing holes", 3)
            mask = close_small_holes(mask, self.new_parameters.close_holes_size)

        report_fun("Smooth border", 4)
        mask = SmoothAlgorithmSelection[self.new_parameters.smooth_border.name].smooth(
            mask, self.new_parameters.smooth_border.values
        )

        report_fun("Flow calculation", 5)
        sprawl_algorithm: BaseWatershed = FlowMethodSelection[self.new_parameters.flow_type.name]
        segmentation = sprawl_algorithm.sprawl(
            mask,
            core_objects,
            noise_filtered,
            ind,
            self.image.spacing,
            self.new_parameters.side_connection,
            operator.gt,
            self.new_parameters.flow_type.values,
            thr[1],
            thr[0],
        )
        if self.new_parameters.use_convex:
            report_fun("convex hull", 6)
            segmentation = convex_fill(segmentation)
        report_fun("Calculation done", 7)
        return ROIExtractionResult(
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


class AutoThresholdAlgorithmParams(BaseThresholdAlgorithmParameters):
    suggested_size: int = Field(200000, ge=0, le=10**6)


class AutoThresholdAlgorithm(BaseSingleThresholdAlgorithm):
    __argument_class__ = AutoThresholdAlgorithmParams
    new_parameters: AutoThresholdAlgorithmParams

    @classmethod
    def get_name(cls):
        return "Auto Threshold"

    def _threshold_image(self, image: np.ndarray) -> np.ndarray:
        sitk_image = sitk.GetImageFromArray(image)
        sitk_mask = sitk.ThresholdMaximumConnectedComponents(sitk_image, self.new_parameters.suggested_size)
        # TODO what exactly it returns. Maybe it is already segmented.
        mask = sitk.GetArrayFromImage(sitk_mask)
        min_val = np.min(image[mask > 0])
        threshold_algorithm: BaseThreshold = ThresholdSelection[self.new_parameters.threshold.name]
        mask2, thr_val = threshold_algorithm.calculate_mask(
            image, None, self.new_parameters.threshold.values, operator.le
        )
        if thr_val < min_val:
            return mask
        return mask2

    def _threshold_and_exclude(self, image, report_fun):
        if self.mask is not None:
            report_fun("Components exclusion apply", 1)
            image[self.mask == 0] = 0
        report_fun("Threshold calculation", 2)
        return self._threshold_image(image)


class CellFromNucleusFlowParameters(BaseModel):
    nucleus_channel: Channel = Field(0, title="Nucleus Channel")
    nucleus_noise_filtering: NoiseFilterSelection = Field(NoiseFilterSelection.get_default(), title="Filter")
    nucleus_threshold: ThresholdSelection = Field(ThresholdSelection.get_default(), title="Threshold")
    cell_channel: Channel = Field(0, title="Cell Channel")
    cell_noise_filtering: NoiseFilterSelection = Field(NoiseFilterSelection.get_default(), title="Filter")
    cell_threshold: ThresholdSelection = Field(ThresholdSelection.get_default(), title="Threshold")
    flow_type: FlowMethodSelection = Field(FlowMethodSelection.get_default(), title="Flow type")
    close_holes: bool = Field(True, title="Fill holes")
    close_holes_size: int = Field(200, title="Maximum holes size (px)", ge=0, le=10**5)
    smooth_border: SmoothAlgorithmSelection = Field(SmoothAlgorithmSelection.get_default(), title="Smooth borders")
    side_connection: bool = Field(
        False,
        title="Side by Side connections",
        description="During calculation of connected components includes only side by side connected pixels",
    )
    minimum_size: int = Field(8000, ge=20, le=10**6)
    use_convex: int = Field(False, title="Use convex hull")


class CellFromNucleusFlow(StackAlgorithm):
    __argument_class__ = CellFromNucleusFlowParameters
    new_parameters: CellFromNucleusFlowParameters

    def calculation_run(self, report_fun: Callable[[str, int], None]) -> ROIExtractionResult:
        report_fun("Nucleus noise removal", 0)
        nucleus_channel = self.get_noise_filtered_channel(
            self.new_parameters.nucleus_channel, self.new_parameters.nucleus_noise_filtering
        )
        report_fun("Nucleus threshold apply", 1)
        nucleus_mask, _nucleus_thr = ThresholdSelection[self.new_parameters.nucleus_threshold.name].calculate_mask(
            nucleus_channel, self.mask, self.new_parameters.nucleus_threshold.values, operator.ge
        )
        report_fun("Nucleus calculate", 2)
        nucleus_objects = sitk.GetArrayFromImage(
            sitk.RelabelComponent(
                sitk.ConnectedComponent(sitk.GetImageFromArray(nucleus_mask), not self.new_parameters.side_connection),
                20,
            )
        )
        sizes = np.bincount(nucleus_objects.flat)
        ind = bisect(sizes[1:], self.new_parameters.minimum_size, lambda x, y: x > y)
        nucleus_objects[nucleus_objects > ind] = 0
        report_fun("Cell noise removal", 3)
        cell_channel = self.get_noise_filtered_channel(
            self.new_parameters.cell_channel, self.new_parameters.cell_noise_filtering
        )
        report_fun("Cell threshold apply", 4)
        cell_mask, cell_thr = ThresholdSelection[self.new_parameters.cell_threshold.name].calculate_mask(
            cell_channel, self.mask, self.new_parameters.cell_threshold.values, operator.ge
        )

        report_fun("Flow calculation", 5)
        sprawl_algorithm: BaseWatershed = FlowMethodSelection[self.new_parameters.flow_type.name]
        mean_brightness = np.mean(cell_channel[cell_mask > 0])
        if mean_brightness < cell_thr:
            mean_brightness = cell_thr + 10
        segmentation = sprawl_algorithm.sprawl(
            cell_mask,
            nucleus_objects,
            cell_channel,
            ind,
            self.image.spacing,
            self.new_parameters.side_connection,
            operator.gt,
            self.new_parameters.flow_type.values,
            cell_thr,
            mean_brightness,
        )
        report_fun("Smooth border", 6)
        segmentation = SmoothAlgorithmSelection[self.new_parameters.smooth_border.name].smooth(
            segmentation, self.new_parameters.smooth_border.values
        )
        if self.new_parameters.use_convex:
            report_fun("convex hull", 7)
            segmentation = convex_fill(segmentation)
        report_fun("Calculation done", 8)
        return ROIExtractionResult(
            roi=segmentation,
            parameters=self.get_segmentation_profile(),
            additional_layers={
                "no size filtering": AdditionalLayerDescription(data=cell_mask, layer_type="labels"),
            },
        )

    def get_info_text(self):
        return ""

    @staticmethod
    def get_steps_num():
        return 9

    @classmethod
    def get_name(cls) -> str:
        return "Cell from nucleus flow"


final_algorithm_list = [
    ThresholdAlgorithm,
    ThresholdFlowAlgorithm,
    MorphologicalWatershed,
    ThresholdPreview,
    AutoThresholdAlgorithm,
    CellFromNucleusFlow,
]


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
