import typing
import warnings
from abc import ABC

import numpy as np
import SimpleITK as sitk
from pydantic import BaseModel, Field, root_validator

from PartSegCore.class_register import register_class, rename_key

from ..algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty, AlgorithmSelection
from .algorithm_base import SegmentationLimitException


class SingleThresholdParams(BaseModel):
    threshold: float = Field(8000.0, ge=-100000, le=100000, title="Threshold", description="Threshold values")


@register_class(version="0.0.1", migrations=[("0.0.0", rename_key("masked", "apply_mask"))])
class SimpleITKThresholdParams(BaseModel):
    apply_mask: bool = Field(True, title="Apply mask", description="If apply mask before calculate threshold")
    bins: int = Field(128, title="Histogram bins", ge=8, le=2 ** 16)

    @root_validator(pre=True)
    def rename_to_apply_mask(cls, values):
        if "masked" in values:
            return rename_key("masked", "apply_mask")(values)
        return values


class BaseThreshold(AlgorithmDescribeBase, ABC):
    @classmethod
    def calculate_mask(
        cls,
        data: np.ndarray,
        mask: typing.Optional[np.ndarray],
        arguments: dict,
        operator: typing.Callable[[object, object], bool],
    ):
        raise NotImplementedError()


class ManualThreshold(BaseThreshold):
    @classmethod
    def get_name(cls):
        return "Manual"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("threshold", "Threshold", 8000.0, (-100000, 100000))]

    @classmethod
    def calculate_mask(cls, data: np.ndarray, mask: typing.Optional[np.ndarray], arguments: dict, operator):
        result = np.array(operator(data, arguments["threshold"])).astype(np.uint8)
        if mask is not None:
            result[mask == 0] = 0
        return result, arguments["threshold"]


class SitkThreshold(BaseThreshold, ABC):
    bins_num = 128

    @classmethod
    def get_fields(cls):
        return [
            AlgorithmProperty("masked", "Apply mask", True),
            AlgorithmProperty("bins", "histogram bins", cls.bins_num, (8, 2 ** 16)),
        ]

    @classmethod
    def calculate_mask(cls, data: np.ndarray, mask: typing.Optional[np.ndarray], arguments: dict, operator):
        if mask is not None and mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8)
        ob, bg, th_op = (0, 1, np.min) if operator(1, 0) else (1, 0, np.max)
        image_sitk = sitk.GetImageFromArray(data)
        if arguments["masked"] and mask is not None:
            mask_sitk = sitk.GetImageFromArray(mask)
            calculated = cls.calculate_threshold(image_sitk, mask_sitk, ob, bg, arguments["bins"], True, 1)
        else:
            calculated = cls.calculate_threshold(image_sitk, ob, bg, arguments["bins"])
        result = sitk.GetArrayFromImage(calculated)
        if mask is not None:
            result[mask == 0] = 0
        threshold = th_op(data[result > 0]) if np.any(result) else th_op(-data)
        return result, threshold

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        raise NotImplementedError


class OtsuThreshold(SitkThreshold):
    @classmethod
    def get_name(cls):
        return "Otsu"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.OtsuThreshold(*args)


class LiThreshold(SitkThreshold):
    bins_num = 256

    @classmethod
    def get_name(cls):
        return "Li"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.LiThreshold(*args)


class MaximumEntropyThreshold(SitkThreshold):
    bins_num = 256

    @classmethod
    def get_name(cls):
        return "Maximum Entropy"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.MaximumEntropyThreshold(*args)


class RenyiEntropyThreshold(SitkThreshold):
    bins_num = 256

    @classmethod
    def get_name(cls):
        return "Renyi Entropy"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.RenyiEntropyThreshold(*args)


class ShanbhagThreshold(SitkThreshold):
    bins_num = 256

    @classmethod
    def get_name(cls):
        return "Shanbhag"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.ShanbhagThreshold(*args)


class TriangleThreshold(SitkThreshold):
    bins_num = 256

    @classmethod
    def get_name(cls):
        return "Triangle"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.TriangleThreshold(*args)


class YenThreshold(SitkThreshold):
    bins_num = 256

    @classmethod
    def get_name(cls):
        return "Yen"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.YenThreshold(*args)


class HuangThreshold(SitkThreshold):
    bins_num = 128

    @classmethod
    def get_name(cls):
        return "Huang"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.HuangThreshold(*args)


class IntermodesThreshold(SitkThreshold):
    bins_num = 256

    @classmethod
    def get_name(cls):
        return "Intermodes"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        try:
            return sitk.IntermodesThreshold(*args)
        except RuntimeError as e:
            if "Exceeded maximum iterations for histogram smoothing" in e.args[0]:
                raise SegmentationLimitException(*e.args)
            raise


class IsoDataThreshold(SitkThreshold):
    bins_num = 256

    @classmethod
    def get_name(cls):
        return "Iso Data"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.IsoDataThreshold(*args)


class KittlerIllingworthThreshold(SitkThreshold):
    bins_num = 256

    @classmethod
    def get_name(cls):
        return "Kittler Illingworth"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        try:
            return sitk.KittlerIllingworthThreshold(*args)
        except RuntimeError as e:
            if "sigma2 <= 0" in e.args[0]:
                raise SegmentationLimitException(*e.args)
            raise


class MomentsThreshold(SitkThreshold):
    bins_num = 256

    @classmethod
    def get_name(cls):
        return "Moments"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.MomentsThreshold(*args)


class ThresholdSelection(AlgorithmSelection):
    pass


ThresholdSelection.register(ManualThreshold)
ThresholdSelection.register(OtsuThreshold)
ThresholdSelection.register(LiThreshold)
ThresholdSelection.register(RenyiEntropyThreshold)
ThresholdSelection.register(ShanbhagThreshold)
ThresholdSelection.register(TriangleThreshold)
ThresholdSelection.register(YenThreshold)
ThresholdSelection.register(HuangThreshold)
ThresholdSelection.register(IntermodesThreshold)
ThresholdSelection.register(IsoDataThreshold)
ThresholdSelection.register(KittlerIllingworthThreshold)
ThresholdSelection.register(MomentsThreshold)
ThresholdSelection.register(MaximumEntropyThreshold)


class DoubleThreshold(BaseThreshold):
    @classmethod
    def get_name(cls):
        # return "Double Choose"
        return "Base/Core"

    @classmethod
    def get_fields(cls):
        return [
            AlgorithmProperty(
                "core_threshold",
                "Core threshold",
                ThresholdSelection.__register__.get_default(),
                possible_values=ThresholdSelection,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "base_threshold",
                "Base threshold",
                ThresholdSelection.__register__.get_default(),
                possible_values=ThresholdSelection,
                value_type=AlgorithmDescribeBase,
            ),
        ]

    @classmethod
    def calculate_mask(cls, data: np.ndarray, mask: typing.Optional[np.ndarray], arguments: dict, operator):
        thr: BaseThreshold = threshold_dict[arguments["core_threshold"]["name"]]
        mask1, thr_val1 = thr.calculate_mask(data, mask, arguments["core_threshold"]["values"], operator)

        thr: BaseThreshold = threshold_dict[arguments["base_threshold"]["name"]]
        mask2, thr_val2 = thr.calculate_mask(data, mask, arguments["base_threshold"]["values"], operator)
        mask2[mask2 > 0] = 1
        mask2[mask1 > 0] = 2
        return mask2, (thr_val1, thr_val2)


class DoubleOtsu(BaseThreshold):
    @classmethod
    def get_name(cls):
        return "Double Otsu"

    @classmethod
    def get_fields(cls):
        return [  # AlgorithmProperty("mask", "Use mask in calculation", True),
            AlgorithmProperty("valley", "Valley emphasis", True),
            AlgorithmProperty("hist_num", "Histogram bins", 128, (8, 2 ** 16)),
        ]

    @classmethod
    def calculate_mask(
        cls,
        data: np.ndarray,
        mask: typing.Optional[np.ndarray],
        arguments: dict,
        operator: typing.Callable[[object, object], bool],
    ):
        cleaned_image_sitk = sitk.GetImageFromArray(data)
        res = sitk.OtsuMultipleThresholds(cleaned_image_sitk, 2, 0, arguments["hist_num"], arguments["valley"])
        res = sitk.GetArrayFromImage(res)
        thr1 = data[res == 2].min()
        thr2 = data[res == 1].min()
        return res, (thr1, thr2)


class DoubleThresholdSelection(AlgorithmSelection):
    pass


DoubleThresholdSelection.register(DoubleThreshold)
DoubleThresholdSelection.register(DoubleOtsu)

double_threshold_dict = DoubleThresholdSelection.__register__
threshold_dict = ThresholdSelection.__register__


def __getattr__(name):
    if name == "threshold_dict":
        warnings.warn(
            "threshold_dict is deprecated. Please use ThresholdSelection instead", category=FutureWarning, stacklevel=2
        )
        return ThresholdSelection.__register__

    if name == "double_threshold_dict":
        warnings.warn(
            "double_threshold_dict is deprecated. Please use DoubleThresholdSelection instead",
            category=FutureWarning,
            stacklevel=2,
        )
        return DoubleThresholdSelection.__register__

    raise AttributeError(f"module {__name__} has no attribute {name}")
