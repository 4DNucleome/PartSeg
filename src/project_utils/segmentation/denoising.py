import typing
from abc import ABC
from enum import Enum
from collections import OrderedDict
import numpy as np

from project_utils.image_operations import gaussian
from project_utils.segmentation.algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty
from project_utils.class_generator import enum_register

class GaussType(Enum):
    layer = 1
    stack = 2

enum_register.register_class(GaussType)

class NoiseRemovalBase(AlgorithmDescribeBase, ABC):
    def noise_remove(self, chanel: np.ndarray, spacing: typing.Iterable[float], arguments: dict) -> np.ndarray:
        raise NotImplementedError()


class NoneNoiseRemoval(NoiseRemovalBase):
    @classmethod
    def get_name(cls):
        return "None"

    @classmethod
    def get_fields(cls):
        return []

    @classmethod
    def noise_remove(cls, chanel: np.ndarray, spacing: typing.Iterable[float], arguments: dict):
        return chanel


class GaussNoiseRemoval(NoiseRemovalBase):
    @classmethod
    def get_name(cls):
        return "Gauss"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("gauss_type", "Gauss type", GaussType.layer),
                AlgorithmProperty("radius", "Gauss radius", 1.0, property_type=float)]

    @classmethod
    def noise_remove(cls, channel: np.ndarray, spacing: typing.Iterable[float], arguments: dict):
        gauss_radius = calculate_operation_radius(arguments["radius"], spacing, arguments["gauss_type"])
        layer = arguments["gauss_type"] == GaussType.layer
        return gaussian(channel, gauss_radius, layer=layer)


def calculate_operation_radius(radius, spacing, gauss_type):
    if gauss_type == GaussType.layer:
        if len(spacing) == 3:
            spacing = spacing[1:]
    base = min(spacing)
    if base != max(spacing):
        ratio = [x / base for x in spacing]
        return [radius / r for r in ratio]
    return radius


noise_removal_dict = OrderedDict(((x.get_name(), x) for x in [NoneNoiseRemoval, GaussNoiseRemoval]))
