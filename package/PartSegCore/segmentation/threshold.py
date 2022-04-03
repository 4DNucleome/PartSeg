import typing
import warnings
from abc import ABC

import numpy as np
import SimpleITK as sitk
from nme import register_class, rename_key, update_argument
from pydantic import Field

from PartSegCore.utils import BaseModel

from ..algorithm_describe_base import AlgorithmDescribeBase, AlgorithmSelection
from .algorithm_base import SegmentationLimitException


class SingleThresholdParams(BaseModel):
    threshold: float = Field(8000.0, ge=-100000, le=100000, title="Threshold", description="Threshold values")


@register_class(version="0.0.1", migrations=[("0.0.1", rename_key("masked", "apply_mask"))])
class SimpleITKThresholdParams128(BaseModel):
    apply_mask: bool = Field(True, description="If apply mask before calculate threshold")
    bins: int = Field(128, title="Histogram bins", ge=8, le=2**16)


@register_class(version="0.0.1", migrations=[("0.0.1", rename_key("masked", "apply_mask"))])
class SimpleITKThresholdParams256(BaseModel):
    apply_mask: bool = Field(True, description="If apply mask before calculate threshold")
    bins: int = Field(128, title="Histogram bins", ge=8, le=2**16)


class BaseThreshold(AlgorithmDescribeBase, ABC):
    @classmethod
    def calculate_mask(
        cls,
        data: np.ndarray,
        mask: typing.Optional[np.ndarray],
        arguments: BaseModel,
        operator: typing.Callable[[object, object], bool],
    ):
        raise NotImplementedError()


class ManualThreshold(BaseThreshold):
    __argument_class__ = SingleThresholdParams

    @classmethod
    def get_name(cls):
        return "Manual"

    @classmethod
    @update_argument("arguments")
    def calculate_mask(
        cls, data: np.ndarray, mask: typing.Optional[np.ndarray], arguments: SingleThresholdParams, operator
    ):
        result = np.array(operator(data, arguments.threshold)).astype(np.uint8)
        if mask is not None:
            result[mask == 0] = 0
        return result, arguments.threshold


class SitkThreshold(BaseThreshold, ABC):
    __argument_class__ = SimpleITKThresholdParams128

    @classmethod
    @update_argument("arguments")
    def calculate_mask(
        cls, data: np.ndarray, mask: typing.Optional[np.ndarray], arguments: SimpleITKThresholdParams128, operator
    ):
        if mask is not None and mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8)
        ob, bg, th_op = (0, 1, np.min) if operator(1, 0) else (1, 0, np.max)
        image_sitk = sitk.GetImageFromArray(data)
        if arguments.apply_mask and mask is not None:
            mask_sitk = sitk.GetImageFromArray(mask)
            calculated = cls.calculate_threshold(image_sitk, mask_sitk, ob, bg, arguments.bins, True, 1)
        else:
            calculated = cls.calculate_threshold(image_sitk, ob, bg, arguments.bins)
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
    __argument_class__ = SimpleITKThresholdParams256

    @classmethod
    def get_name(cls):
        return "Li"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.LiThreshold(*args)


class MaximumEntropyThreshold(SitkThreshold):
    __argument_class__ = SimpleITKThresholdParams256

    @classmethod
    def get_name(cls):
        return "Maximum Entropy"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.MaximumEntropyThreshold(*args)


class RenyiEntropyThreshold(SitkThreshold):
    __argument_class__ = SimpleITKThresholdParams256

    @classmethod
    def get_name(cls):
        return "Renyi Entropy"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.RenyiEntropyThreshold(*args)


class ShanbhagThreshold(SitkThreshold):
    __argument_class__ = SimpleITKThresholdParams256

    @classmethod
    def get_name(cls):
        return "Shanbhag"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.ShanbhagThreshold(*args)


class TriangleThreshold(SitkThreshold):
    __argument_class__ = SimpleITKThresholdParams256

    @classmethod
    def get_name(cls):
        return "Triangle"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.TriangleThreshold(*args)


class YenThreshold(SitkThreshold):
    __argument_class__ = SimpleITKThresholdParams256

    @classmethod
    def get_name(cls):
        return "Yen"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.YenThreshold(*args)


class HuangThreshold(SitkThreshold):
    __argument_class__ = SimpleITKThresholdParams256

    @classmethod
    def get_name(cls):
        return "Huang"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.HuangThreshold(*args)


class IntermodesThreshold(SitkThreshold):
    __argument_class__ = SimpleITKThresholdParams256

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
    __argument_class__ = SimpleITKThresholdParams256

    @classmethod
    def get_name(cls):
        return "Iso Data"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.IsoDataThreshold(*args)


class KittlerIllingworthThreshold(SitkThreshold):
    __argument_class__ = SimpleITKThresholdParams256

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
    __argument_class__ = SimpleITKThresholdParams256

    @classmethod
    def get_name(cls):
        return "Moments"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.MomentsThreshold(*args)


class ThresholdSelection(AlgorithmSelection, class_methods=["calculate_mask"], suggested_base_class=BaseThreshold):
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


class DoubleThresholdParams(BaseModel):
    core_threshold: ThresholdSelection = ThresholdSelection.get_default()
    base_threshold: ThresholdSelection = ThresholdSelection.get_default()


class DoubleThreshold(BaseThreshold):
    __argument_class__ = DoubleThresholdParams

    @classmethod
    def get_name(cls):
        # return "Double Choose"
        return "Base/Core"

    @classmethod
    @update_argument("arguments")
    def calculate_mask(
        cls, data: np.ndarray, mask: typing.Optional[np.ndarray], arguments: DoubleThresholdParams, operator
    ):
        thr: BaseThreshold = threshold_dict[arguments.core_threshold.name]
        mask1, thr_val1 = thr.calculate_mask(data, mask, arguments.core_threshold.values, operator)

        thr: BaseThreshold = threshold_dict[arguments.base_threshold.name]
        mask2, thr_val2 = thr.calculate_mask(data, mask, arguments.base_threshold.values, operator)
        mask2[mask2 > 0] = 1
        mask2[mask1 > 0] = 2
        return mask2, (thr_val1, thr_val2)


@register_class(version="0.0.1", migrations=[("0.0.1", rename_key("hist_num", "bins"))])
class DoubleOtsuParams(BaseModel):
    valley: bool = Field(True, title="Valley emphasis")
    bins: int = Field(128, title="Histogram bins", ge=8, le=2**16)


class DoubleOtsu(BaseThreshold):
    __argument_class__ = DoubleOtsuParams

    @classmethod
    def get_name(cls):
        return "Double Otsu"

    @classmethod
    @update_argument("arguments")
    def calculate_mask(
        cls,
        data: np.ndarray,
        mask: typing.Optional[np.ndarray],
        arguments: DoubleOtsuParams,
        operator: typing.Callable[[object, object], bool],
    ):
        cleaned_image_sitk = sitk.GetImageFromArray(data)
        res = sitk.OtsuMultipleThresholds(cleaned_image_sitk, 2, 0, arguments.bins, arguments.valley)
        res = sitk.GetArrayFromImage(res)
        thr1 = data[res == 2].min()
        thr2 = data[res == 1].min()
        return res, (thr1, thr2)


class DoubleThresholdSelection(
    AlgorithmSelection, class_methods=["calculate_mask"], suggested_base_class=BaseThreshold
):
    pass


DoubleThresholdSelection.register(DoubleThreshold)
DoubleThresholdSelection.register(DoubleOtsu)

double_threshold_dict = DoubleThresholdSelection.__register__
threshold_dict = ThresholdSelection.__register__


def __getattr__(name):  # pragma: no cover
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
