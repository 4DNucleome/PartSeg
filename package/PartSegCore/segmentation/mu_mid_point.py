import warnings
from abc import ABC

import numpy as np

from ..algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty, AlgorithmSelection


class BaseMuMid(AlgorithmDescribeBase, ABC):
    @classmethod
    def get_fields(cls):
        return []

    @classmethod
    def value(cls, sprawl_area: np.ndarray, data: np.ndarray, lower_bound, upper_bound, arguments: dict):
        raise NotImplementedError()


class MeanBound(BaseMuMid):
    @classmethod
    def get_name(cls):
        return "Mean bound"

    @classmethod
    def value(cls, sprawl_area: np.ndarray, data: np.ndarray, lower_bound, upper_bound, arguments: dict):
        return (lower_bound + upper_bound) / 2


class PercentBound(BaseMuMid):
    @classmethod
    def get_name(cls):
        return "Percent from lower bound"

    @classmethod
    def get_fields(cls):
        return [
            AlgorithmProperty(
                "percent",
                "Percent",
                50,
                options_range=(0, 100),
                help_text="Calculate: lower_value + (upper_bound - lower_bound) * percent / 100",
            )
        ]

    @classmethod
    def value(cls, sprawl_area: np.ndarray, data: np.ndarray, lower_bound, upper_bound, arguments: dict):
        return min(lower_bound, upper_bound) + abs(lower_bound - upper_bound) * arguments["percent"] / 100


class MeanPixelValue(BaseMuMid):
    @classmethod
    def get_name(cls):
        return "Mean pixel value"

    @classmethod
    def value(cls, sprawl_area: np.ndarray, data: np.ndarray, lower_bound, upper_bound, arguments: dict):
        return np.mean[data[sprawl_area > 0]]


class MedianPixelValue(BaseMuMid):
    @classmethod
    def get_name(cls):
        return "Median pixel value"

    @classmethod
    def value(cls, sprawl_area: np.ndarray, data: np.ndarray, lower_bound, upper_bound, arguments: dict):
        return np.median[data[sprawl_area > 0]]


class QuantilePixelValue(BaseMuMid):
    @classmethod
    def get_name(cls):
        return "Quantile pixel value"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("quantile", "Quantile", 50, options_range=(0, 100))]

    @classmethod
    def value(cls, sprawl_area: np.ndarray, data: np.ndarray, lower_bound, upper_bound, arguments: dict):
        return np.quantile[data[sprawl_area > 0], arguments["quantile"] / 100]


class MuMidSelection(AlgorithmSelection, class_methods=["value"], suggested_base_class=BaseMuMid):
    pass


MuMidSelection.register(MeanBound)
MuMidSelection.register(PercentBound)
MuMidSelection.register(MeanPixelValue)
MuMidSelection.register(MedianPixelValue)
MuMidSelection.register(QuantilePixelValue)


def __getattr__(name):  # pragma: no cover
    if name == "mu_mid_dict":
        warnings.warn(
            "mu_mid_dict is deprecated. Please use MuMidSelection instead", category=FutureWarning, stacklevel=2
        )
        return MuMidSelection.__register__
    raise AttributeError(f"module {__name__} has no attribute {name}")
