import typing
from abc import ABC
from enum import Enum
import numpy as np

from ..image_operations import gaussian, median
from ..algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty, Register
from ..class_generator import enum_register


class DimensionType(Enum):
    Layer = 1
    Stack = 2

    def __str__(self):
        return self.name.replace("_", " ")


try:
    # noinspection PyUnresolvedReferences,PyUnboundLocalVariable
    reloading
except NameError:
    reloading = False  # means the module is being imported
    enum_register.register_class(DimensionType, old_name="GaussType")


class NoiseFilteringBase(AlgorithmDescribeBase, ABC):
    """Base class for noise filtering operations"""
    @classmethod
    def noise_filter(cls, chanel: np.ndarray, spacing: typing.Iterable[float], arguments: dict) -> np.ndarray:
        """
        This function need be overloaded in implementation

        :param channel: single channel ad 2d or 3d array
        :param spacing: image spacing
        :return: channel array with removed noise
        """
        raise NotImplementedError()


class NoneNoiseFiltering(NoiseFilteringBase):
    @classmethod
    def get_name(cls):
        return "None"

    @classmethod
    def get_fields(cls):
        return []

    @classmethod
    def noise_filter(cls, chanel: np.ndarray, spacing: typing.Iterable[float], arguments: dict):
        return chanel


class GaussNoiseFiltering(NoiseFilteringBase):
    @classmethod
    def get_name(cls):
        return "Gauss"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("dimension_type", "Gauss type", DimensionType.Layer),
                AlgorithmProperty("radius", "Gauss radius", 1.0, property_type=float)]

    @classmethod
    def noise_filter(cls, channel: np.ndarray, spacing: typing.Iterable[float], arguments: dict):
        gauss_radius = calculate_operation_radius(arguments["radius"], spacing, arguments["dimension_type"])
        layer = arguments["dimension_type"] == DimensionType.Layer
        return gaussian(channel, gauss_radius, layer=layer)


def calculate_operation_radius(radius, spacing, gauss_type):
    if gauss_type == DimensionType.Layer:
        if len(spacing) == 3:
            spacing = spacing[1:]
    base = min(spacing)
    if base != max(spacing):
        ratio = [x / base for x in spacing]
        return [radius / r for r in ratio]
    return [radius for _ in spacing]


class MedianNoiseFiltering(NoiseFilteringBase):
    @classmethod
    def get_name(cls):
        return "Median"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("dimension_type", "Median type", DimensionType.Layer),
                AlgorithmProperty("radius", "Median radius", 1, property_type=int)]

    @classmethod
    def noise_filter(cls, channel: np.ndarray, spacing: typing.Iterable[float], arguments: dict):
        gauss_radius = calculate_operation_radius(arguments["radius"], spacing, arguments["dimension_type"])
        layer = arguments["dimension_type"] == DimensionType.Layer
        gauss_radius = [int(x) for x in gauss_radius]
        return median(channel, gauss_radius, layer=layer)


noise_filtering_dict = Register(NoneNoiseFiltering, GaussNoiseFiltering, MedianNoiseFiltering,
                                class_methods=["noise_filter"])
