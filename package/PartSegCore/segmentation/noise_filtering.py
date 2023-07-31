import typing
import warnings
from abc import ABC
from enum import Enum

import numpy as np
from local_migrator import register_class, rename_key, update_argument
from pydantic import Field

from PartSegCore.algorithm_describe_base import AlgorithmDescribeBase, AlgorithmSelection
from PartSegCore.image_operations import bilateral, gaussian, median
from PartSegCore.segmentation.algorithm_base import calculate_operation_radius as _calculate_operation_radius
from PartSegCore.utils import BaseModel


@register_class(old_paths=["PartSeg.utils.segmentation.noise_filtering.GaussType"])
class DimensionType(Enum):
    Layer = 1
    Stack = 2

    def __str__(self):
        return self.name.replace("_", " ")


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
        raise NotImplementedError


class NoneNoiseFiltering(NoiseFilteringBase):
    __argument_class__ = BaseModel

    @classmethod
    def get_name(cls):
        return "None"

    @classmethod
    def noise_filter(cls, channel: np.ndarray, spacing: typing.Iterable[float], arguments: dict):
        return channel


@register_class(version="0.0.1", migrations=[("0.0.1", rename_key("gauss_type", "dimension_type", optional=True))])
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
    return [radius for _ in spacing] if res == radius else res


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


class NoiseFilterSelection(AlgorithmSelection, class_methods=["noise_filter"], suggested_base_class=NoiseFilteringBase):
    pass


NoiseFilterSelection.register(NoneNoiseFiltering)
NoiseFilterSelection.register(GaussNoiseFiltering)
NoiseFilterSelection.register(MedianNoiseFiltering)
NoiseFilterSelection.register(BilateralNoiseFiltering)


def __getattr__(name):  # pragma: no cover
    if name == "noise_filtering_dict":
        warnings.warn(
            "noise_filtering_dict is deprecated. Please use NoiseFilterSelection instead",
            category=FutureWarning,
            stacklevel=2,
        )
        return NoiseFilterSelection.__register__
    raise AttributeError(f"module {__name__} has no attribute {name}")
