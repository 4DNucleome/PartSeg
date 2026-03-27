"""
This module contains PartSeg wrappers for function for :py:mod:`..sprawl_utils.find_split`.
"""

import warnings
from abc import ABC
from enum import Enum
from typing import Any, Callable

import numpy as np
from local_migrator import update_argument
from pydantic import Field

from PartSegCore.algorithm_describe_base import AlgorithmDescribeBase, AlgorithmSelection
from PartSegCore.segmentation.algorithm_base import SegmentationLimitException
from PartSegCore.utils import BaseModel
from PartSegCore_compiled_backend.multiscale_opening import MuType, PyMSO, calculate_mu
from PartSegCore_compiled_backend.sprawl_utils.find_split import (
    euclidean_sprawl,
    fdt_sprawl,
    path_maximum_sprawl,
    path_minimum_sprawl,
)


class BaseWatershed(AlgorithmDescribeBase, ABC):
    """base class for all sprawl interface"""

    __argument_class__ = BaseModel

    @classmethod
    def sprawl(
        cls,
        sprawl_area: np.ndarray,
        core_objects: np.ndarray,
        data: np.ndarray,
        components_num: int,
        spacing,
        side_connection: bool,
        operator: Callable[[Any, Any], bool],
        arguments: dict,
        lower_bound,
        upper_bound,
    ):
        """
        This method calculate sprawl

        :param sprawl_area: Mask area to which sprawl should be limited
        :param core_objects: Starting objects for sprawl
        :param data: density information
        :param components_num: number of components in core_objects
        :param spacing: Image spacing. Needed for sprawls which use metrics
        :param side_connection:
        :param operator:
        :param arguments: dict with parameters reported by function :py:meth:`get_fields`
        :param lower_bound: data value lower bound
        :param upper_bound: data value upper bound
        :return:
        """
        raise NotImplementedError


class PathWatershed(BaseWatershed):
    @classmethod
    def get_name(cls):
        return "Path"

    @classmethod
    def sprawl(
        cls,
        sprawl_area: np.ndarray,
        core_objects: np.ndarray,
        data: np.ndarray,
        components_num: int,
        spacing,
        side_connection: bool,
        operator: Callable[[Any, Any], bool],
        arguments: dict,
        lower_bound,
        upper_bound,
    ):
        path_sprawl = path_maximum_sprawl if operator(1, 0) else path_minimum_sprawl
        image = data.astype(np.float64)
        image[sprawl_area == 0] = 0
        neigh = get_neighbourhood(spacing, get_neigh(side_connection))
        mid = path_sprawl(image, core_objects, components_num, neigh)
        return path_sprawl(image, mid, components_num, neigh)


class DistanceWatershed(BaseWatershed):
    """Calculate Euclidean sprawl (watershed) with respect to image spacing"""

    @classmethod
    def get_name(cls):
        return "Euclidean"

    @classmethod
    def sprawl(
        cls,
        sprawl_area: np.ndarray,
        core_objects: np.ndarray,
        data: np.ndarray,
        components_num: int,
        spacing,
        side_connection: bool,
        operator: Callable[[Any, Any], bool],
        arguments: dict,
        lower_bound,
        upper_bound,
    ):
        neigh, dist = calculate_distances_array(spacing, get_neigh(side_connection))
        return euclidean_sprawl(sprawl_area, core_objects, components_num, neigh, dist)


class FDTWatershed(BaseWatershed):
    @classmethod
    def get_name(cls):
        return "Fuzzy distance"

    @classmethod
    def sprawl(
        cls,
        sprawl_area: np.ndarray,
        core_objects: np.ndarray,
        data: np.ndarray,
        components_num: int,
        spacing,
        side_connection: bool,
        operator: Callable[[Any, Any], bool],
        arguments: dict,
        lower_bound,
        upper_bound,
    ):
        image = data.astype(np.float64)
        image[sprawl_area == 0] = 0
        neigh, dist = calculate_distances_array(spacing, get_neigh(side_connection))
        return fdt_sprawl(image, core_objects, components_num, neigh, dist, lower_bound, upper_bound)


class PathDistanceWatershed(BaseWatershed):
    @classmethod
    def get_name(cls):
        return "Path euclidean"

    @classmethod
    def sprawl(
        cls,
        sprawl_area: np.ndarray,
        core_objects: np.ndarray,
        data: np.ndarray,
        components_num: int,
        spacing,
        side_connection: bool,
        operator: Callable[[Any, Any], bool],
        arguments: dict,
        lower_bound,
        upper_bound,
    ):
        mid = PathWatershed.sprawl(
            sprawl_area,
            core_objects,
            data,
            components_num,
            spacing,
            side_connection,
            operator,
            arguments,
            lower_bound,
            upper_bound,
        )
        return DistanceWatershed.sprawl(
            sprawl_area,
            mid,
            data,
            components_num,
            spacing,
            side_connection,
            operator,
            arguments,
            lower_bound,
            upper_bound,
        )


class MSOWatershedParams(BaseModel):
    step_limits: int = Field(100, ge=1, le=1000, title="Steep limits", description="Limits of Steps")
    reflective: bool = False


class MSOWatershed(BaseWatershed):
    __argument_class__ = MSOWatershedParams

    @classmethod
    def get_name(cls):
        return "MultiScale Opening"

    @classmethod
    @update_argument("arguments")
    def sprawl(
        cls,
        sprawl_area: np.ndarray,
        core_objects: np.ndarray,
        data: np.ndarray,
        components_num: int,
        spacing,
        side_connection: bool,
        operator: Callable[[Any, Any], bool],
        arguments: MSOWatershedParams,
        lower_bound,
        upper_bound,
    ):
        if components_num > 250:
            raise SegmentationLimitException("Current implementation of MSO do not support more than 250 components")
        mso = PyMSO()
        neigh, dist = calculate_distances_array(spacing, get_neigh(side_connection))
        components_arr = np.copy(core_objects).astype(np.uint8)
        components_arr[components_arr > 0] += 1
        components_arr[sprawl_area == 0] = 1
        mso.set_neighbourhood(neigh, dist)
        mso.set_components(components_arr, components_num + 1)
        mso.set_use_background(False)
        try:
            mu_array = calculate_mu(data.copy("C"), lower_bound, upper_bound, MuType.base_mu)
        except OverflowError as e:
            raise SegmentationLimitException("Wrong range for ") from e
        if arguments.reflective:
            mu_array[mu_array < 0.5] = 1 - mu_array[mu_array < 0.5]
        mso.set_mu_array(mu_array)
        try:
            mso.run_MSO(arguments.step_limits)
        except RuntimeError as e:
            if e.args[0] == "to many steps: constrained dilation":
                raise SegmentationLimitException(*e.args) from e
            raise

        result = mso.get_result_catted()
        result[result > 0] -= 1
        return result


class WatershedSelection(AlgorithmSelection, class_methods=["sprawl"], suggested_base_class=BaseWatershed):
    """This register contains algorithms for sprawl area from core object."""


WatershedSelection.register(MSOWatershed, old_names=["MultiScale Opening sprawl"])
WatershedSelection.register(PathWatershed, old_names=["Path sprawl"])
WatershedSelection.register(DistanceWatershed, old_names=["Euclidean sprawl"])
WatershedSelection.register(PathDistanceWatershed, old_names=["Path euclidean sprawl"])
WatershedSelection.register(FDTWatershed, old_names=["Fuzzy distance sprawl"])

FlowMethodSelection = WatershedSelection


def __getattr__(name):  # pragma: no cover
    if name == "flow_dict":
        warnings.warn(
            "flow_dict is deprecated. Please use FlowMethodSelection instead", category=FutureWarning, stacklevel=2
        )
        return WatershedSelection.__register__
    if name == "sprawl_dict":
        warnings.warn(
            "sprawl_dict is deprecated. Please use FlowMethodSelection instead", category=FutureWarning, stacklevel=2
        )
        return WatershedSelection.__register__
    raise AttributeError(f"module {__name__} has no attribute {name}")


def get_neigh(sides):
    return NeighType.sides if sides else NeighType.edges


class NeighType(Enum):
    sides = 6
    edges = 18
    vertex = 26

    def __str__(self):
        return self.name


def calculate_distances_array(spacing, neigh_type: NeighType):
    """
    :param spacing: image spacing
    :param neigh_type: neighbourhood type
    :return: neighbourhood array, distance array
    """
    min_dist = min(spacing)
    normalized_spacing = [x / min_dist for x in spacing]
    if len(normalized_spacing) == 2:
        neighbourhood_array = neighbourhood2d
        if neigh_type == NeighType.sides:
            neighbourhood_array = neighbourhood_array[:4]
        normalized_spacing = [0, *normalized_spacing]
    else:
        neighbourhood_array = neighbourhood[: neigh_type.value]
    normalized_spacing = np.array(normalized_spacing)
    return neighbourhood_array, np.sqrt(np.sum((neighbourhood_array * normalized_spacing) ** 2, axis=1))


def get_neighbourhood(spacing, neigh_type: NeighType):
    if len(spacing) == 2:
        if neigh_type == NeighType.sides:
            return neighbourhood2d[:4]
        return neighbourhood2d
    return neighbourhood[: neigh_type.value]


neighbourhood = np.array(
    [
        [0, -1, 0],
        [0, 0, -1],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
        [1, 0, 0],
        [-1, -1, 0],
        [1, -1, 0],
        [-1, 1, 0],
        [1, 1, 0],
        [-1, 0, -1],
        [1, 0, -1],
        [-1, 0, 1],
        [1, 0, 1],
        [0, -1, -1],
        [0, 1, -1],
        [0, -1, 1],
        [0, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1],
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.int8,
)

neighbourhood2d = np.array(
    [[0, -1, 0], [0, 0, -1], [0, 1, 0], [0, 0, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1], [0, 1, 1]], dtype=np.int8
)
