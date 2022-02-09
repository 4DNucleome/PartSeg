import typing
from abc import ABC
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from ..algorithm_describe_base import AlgorithmDescribeBase, Register
from ..class_generator import enum_register
from ..class_register import update_argument
from ..image_operations import bilateral, gaussian, median
from .algorithm_base import calculate_operation_radius as _calculate_operation_radius


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
    def noise_filter(cls, channel: np.ndarray, spacing: typing.Iterable[float], arguments: dict) -> np.ndarray:
        """
        This function need be overloaded in implementation

        :param channel: single channel ad 2d or 3d array
        :param spacing: image spacing
        :param arguments: additional arguments defined by :py:meth:`get_fields`
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
    def noise_filter(cls, channel: np.ndarray, spacing: typing.Iterable[float], arguments: dict):
        return channel


class GaussNoiseFilteringParams(BaseModel):
    dimension_type: DimensionType = Field(DimensionType.Layer, title="Gauss type")
    radius: float = Field(1.0, title="Gauss radius", ge=0, le=100)


class GaussNoiseFiltering(NoiseFilteringBase):
    __argument_class__ = GaussNoiseFilteringParams

    @classmethod
    def get_name(cls):
        return "Gauss"

    @classmethod
    @update_argument("arguments")
    def noise_filter(cls, channel: np.ndarray, spacing: typing.Iterable[float], arguments: GaussNoiseFilteringParams):
        gauss_radius = calculate_operation_radius(arguments.radius, spacing, arguments.dimension_type)
        layer = arguments.dimension_type == DimensionType.Layer
        return gaussian(channel, gauss_radius, layer=layer)


class BilateralNoiseFilteringParams(BaseModel):
    dimension_type: DimensionType = Field(DimensionType.Layer, title="Bilateral type")
    radius: float = Field(1.0, title="Bilateral radius", ge=0, le=100)


class BilateralNoiseFiltering(NoiseFilteringBase):
    __argument_class__ = BilateralNoiseFilteringParams

    @classmethod
    def get_name(cls):
        return "Bilateral"

    @classmethod
    @update_argument("arguments")
    def noise_filter(
        cls, channel: np.ndarray, spacing: typing.Iterable[float], arguments: BilateralNoiseFilteringParams
    ):
        gauss_radius = calculate_operation_radius(arguments.radius, spacing, arguments.dimension_type)
        layer = arguments.dimension_type == DimensionType.Layer
        return bilateral(channel, max(gauss_radius), layer=layer)


def calculate_operation_radius(radius, spacing, gauss_type):
    res = _calculate_operation_radius(radius, spacing, gauss_type)
    if res == radius:
        return [radius for _ in spacing]
    return res


class MedianNoiseFilteringParams(BaseModel):
    dimension_type: DimensionType = Field(DimensionType.Layer, title="Median type")
    radius: int = Field(1, title="Median radius", ge=0, le=100)


class MedianNoiseFiltering(NoiseFilteringBase):
    __argument_class__ = MedianNoiseFilteringParams

    @classmethod
    def get_name(cls):
        return "Median"

    @classmethod
    @update_argument("arguments")
    def noise_filter(cls, channel: np.ndarray, spacing: typing.Iterable[float], arguments: MedianNoiseFilteringParams):
        gauss_radius = calculate_operation_radius(arguments.radius, spacing, arguments.dimension_type)
        layer = arguments.dimension_type == DimensionType.Layer
        gauss_radius = [int(x) for x in gauss_radius]
        return median(channel, gauss_radius, layer=layer)


noise_filtering_dict = Register(
    NoneNoiseFiltering,
    GaussNoiseFiltering,
    MedianNoiseFiltering,
    BilateralNoiseFiltering,
    class_methods=["noise_filter"],
)
