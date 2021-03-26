import typing
from abc import ABC

import numpy as np
import SimpleITK as sitk

from ..algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty, Register


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
        return sitk.IntermodesThreshold(*args)


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
        return sitk.KittlerIllingworthThreshold(*args)


class MomentsThreshold(SitkThreshold):
    bins_num = 256

    @classmethod
    def get_name(cls):
        return "Moments"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.MomentsThreshold(*args)


threshold_dict = Register()
threshold_dict.register(ManualThreshold)
threshold_dict.register(OtsuThreshold)
threshold_dict.register(LiThreshold)
threshold_dict.register(RenyiEntropyThreshold)
threshold_dict.register(ShanbhagThreshold)
threshold_dict.register(TriangleThreshold)
threshold_dict.register(YenThreshold)
threshold_dict.register(HuangThreshold)
threshold_dict.register(IntermodesThreshold)
threshold_dict.register(IsoDataThreshold)
threshold_dict.register(KittlerIllingworthThreshold)
threshold_dict.register(MomentsThreshold)
threshold_dict.register(MaximumEntropyThreshold)


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
                threshold_dict.get_default(),
                possible_values=threshold_dict,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "base_threshold",
                "Base threshold",
                threshold_dict.get_default(),
                possible_values=threshold_dict,
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


double_threshold_dict = Register()

double_threshold_dict.register(DoubleThreshold)
double_threshold_dict.register(DoubleOtsu)
