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
        result = operator(data, arguments["threshold"]).astype(np.uint8)
        return result, arguments["threshold"]


class SitkThreshold(BaseThreshold, ABC):
    bins_num = 128

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("masked", "Apply mask", True),
                AlgorithmProperty("bins", f"{cls.get_name()} bins", cls.bins_num, (8, 2 ** 16))]

    @classmethod
    def calculate_mask(cls, data: np.ndarray, mask: typing.Optional[np.ndarray], arguments: dict, operator):
        if mask is not None and mask.dtype == np.bool:
            mask = mask.astype(np.uint8)
        if operator(1, 0):
            ob, bg, th_op = 0, 1, np.min
        else:
            ob, bg, th_op = 1, 0, np.max
        print(operator, ob, bg, th_op)
        image_sitk = sitk.GetImageFromArray(data)
        if arguments["masked"] and mask is not None:
            mask_sitk = sitk.GetImageFromArray(mask)
            print("masked")
            calculated = cls.calculate_threshold(image_sitk, mask_sitk, ob, bg, arguments["bins"], True, 1)
        else:
            print("non masked")
            calculated = cls.calculate_threshold(image_sitk, ob, bg, arguments["bins"])
        result = sitk.GetArrayFromImage(calculated)
        threshold = th_op(data[result > 0])
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
        print("otsu:", *args)
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


threshold_dict = Register()
threshold_dict.register(ManualThreshold)
threshold_dict.register(OtsuThreshold)
threshold_dict.register(LiThreshold)

class DoubleThreshold(BaseThreshold):
    @classmethod
    def get_name(cls):
        return "Double Choose"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("core_threshold", "Core threshold", next(iter(threshold_dict.keys())),
                                  possible_values=threshold_dict, property_type=AlgorithmDescribeBase),
                AlgorithmProperty("base_threshold", "Base threshold", next(iter(threshold_dict.keys())),
                                  possible_values=threshold_dict, property_type=AlgorithmDescribeBase)
                ]

    @classmethod
    def calculate_mask(cls, data: np.ndarray, mask: typing.Optional[np.ndarray], arguments: dict, operator):
        thr: BaseThreshold = threshold_dict[arguments["core_threshold"]["name"]]
        mask1, thr_val1 = thr.calculate_mask(data, mask, arguments["core_threshold"]["values"],
                                           operator)

        thr: BaseThreshold = threshold_dict[arguments["base_threshold"]["name"]]
        mask2, thr_val2 = thr.calculate_mask(data, mask, arguments["base_threshold"]["values"],
                                             operator)
        mask2[mask2 > 0] = 2
        mask2[mask1 > 0] = 1
        return mask2, (thr_val1, thr_val2)



double_threshold_dict = Register()

double_threshold_dict.register(DoubleThreshold)