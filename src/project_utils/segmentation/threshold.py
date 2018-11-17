from abc import ABC

from .algorithm_describe_base import Register, AlgorithmDescribeBase, AlgorithmProperty
import numpy as np
import SimpleITK as sitk
import typing


class BaseThreshold(AlgorithmDescribeBase, ABC):
    @classmethod
    def calculate_mask(cls, data: np.ndarray, mask: typing.Optional[np.ndarray], arguments: dict, operator):
        raise NotImplementedError()


class ManualThreshold(BaseThreshold):
    @classmethod
    def get_name(cls):
        return "Manual"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("threshold", "Threshold", 8000, (-100000, 100000))]

    @classmethod
    def calculate_mask(cls, data: np.ndarray, mask: typing.Optional[np.ndarray], arguments: dict, operator):
        result = operator(data, arguments["threshold"])
        return result, arguments["threshold"]


class SitkThreshold(BaseThreshold, ABC):
    bins_num = 128

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("masked", "Apply mask", True),
                AlgorithmProperty("bins", f"{cls.get_name()} bins", cls.bins_num, (8, 2 ** 16))]

    @classmethod
    def calculate_mask(cls, data: np.ndarray, mask: typing.Optional[np.ndarray], arguments: dict, operator):
        if operator(1, 0):
            ob, bg, th_op = 1, 0, np.min
        else:
            ob, bg, th_op = 0, 1, np.max
        image_sitk = sitk.GetImageFromArray(data)
        if arguments["masked"] and mask is not None:
            mask_sitk = sitk.GetImageFromArray(mask)
            calculated = cls.calculate_threshold(image_sitk, mask_sitk, ob, bg, arguments["bins"], True, 1)
        else:
            calculated = cls.calculate_threshold(image_sitk, ob, bg)
        result = sitk.GetArrayFromImage(calculated)
        threshold = th_op(result[result > 0])
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


threshold_dict = Register()
threshold_dict.register(ManualThreshold)
threshold_dict.register(OtsuThreshold)
threshold_dict.register(LiThreshold)

double_threshold_dict = Register()
