from abc import ABC
from enum import Enum
from typing import Callable, Any

import numpy as np

from ..multiscale_opening import PyMSO, calculate_mu, MuType
from ..distance_in_structure.find_split import path_maximum_sprawl, path_minimum_sprawl, euclidean_sprawl, \
    fdt_sprawl
from ..algorithm_describe_base import Register, AlgorithmDescribeBase, AlgorithmProperty


class BaseSprawl(AlgorithmDescribeBase, ABC):
    @classmethod
    def get_fields(cls):
        return []

    @classmethod
    def sprawl(cls, sprawl_area: np.ndarray, core_objects: np.ndarray, data: np.ndarray, components_num: int, spacing,
               side_connection: bool, operator: Callable[[Any, Any], bool], arguments: dict, lower_bound, upper_bound):
        raise NotImplementedError()


class PathSprawl(BaseSprawl):
    @classmethod
    def get_name(cls):
        return "Path sprawl"

    @classmethod
    def sprawl(cls, sprawl_area: np.ndarray, core_objects: np.ndarray, data: np.ndarray, components_num: int, spacing,
               side_connection: bool, operator: Callable[[Any, Any], bool], arguments: dict, lower_bound, upper_bound):
        if operator(1, 0):
            path_sprawl = path_maximum_sprawl
        else:
            path_sprawl = path_minimum_sprawl
        # print(path_sprawl)
        image = data.astype(np.float64)
        image[sprawl_area == 0] = 0
        neigh = get_neighbourhood(spacing, get_neigh(side_connection))
        mid = path_sprawl(image, core_objects, components_num, neigh)
        return path_sprawl(image, mid, components_num, neigh)


class DistanceSprawl(BaseSprawl):
    @classmethod
    def get_name(cls):
        return "Euclidean sprawl"

    @classmethod
    def sprawl(cls, sprawl_area: np.ndarray, core_objects: np.ndarray, data: np.ndarray, components_num: int, spacing,
               side_connection: bool, operator: Callable[[Any, Any], bool], arguments: dict, lower_bound, upper_bound):
        neigh, dist = calculate_distances_array(spacing, get_neigh(side_connection))
        return euclidean_sprawl(sprawl_area, core_objects, components_num, neigh, dist)


class FDTSprawl(BaseSprawl):
    @classmethod
    def get_name(cls):
        return "Fuzzy distance sprawl"

    @classmethod
    def sprawl(cls, sprawl_area: np.ndarray, core_objects: np.ndarray, data: np.ndarray, components_num: int, spacing,
               side_connection: bool, operator: Callable[[Any, Any], bool], arguments: dict, lower_bound, upper_bound):
        image = data.astype(np.float64)
        image[sprawl_area == 0] = 0
        """if lower_bound > upper_bound:
            image = -image"""
        neigh, dist = calculate_distances_array(spacing, get_neigh(side_connection))
        return fdt_sprawl(image, core_objects, components_num, neigh, dist, lower_bound, upper_bound)


class PathDistanceSprawl(BaseSprawl):
    @classmethod
    def get_name(cls):
        return "Path euclidean sprawl"

    @classmethod
    def sprawl(cls, sprawl_area: np.ndarray, core_objects: np.ndarray, data: np.ndarray, components_num: int, spacing,
               side_connection: bool, operator: Callable[[Any, Any], bool], arguments: dict, lower_bound, upper_bound):
        mid = PathSprawl.sprawl(sprawl_area, core_objects, data, components_num, spacing, side_connection, operator,
                                arguments, lower_bound, upper_bound)
        return DistanceSprawl.sprawl(sprawl_area, mid, data, components_num, spacing, side_connection, operator,
                                     arguments, lower_bound, upper_bound)


class MSOSprawl(BaseSprawl):
    @classmethod
    def get_name(cls):
        return "MultiScale Opening sprawl"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("step_limits", "Limits of Steps", 100, options_range=(1, 1000), property_type=int),
                AlgorithmProperty("reflective", "Reflective", False, property_type=bool)]

    @classmethod
    def sprawl(cls, sprawl_area: np.ndarray, core_objects: np.ndarray, data: np.ndarray, components_num: int, spacing,
               side_connection: bool, operator: Callable[[Any, Any], bool], arguments: dict, lower_bound, upper_bound):
        assert components_num < 255
        mso = PyMSO()
        neigh, dist = calculate_distances_array(spacing, get_neigh(side_connection))
        components_arr = np.copy(core_objects).astype(np.uint8)
        components_arr[components_arr > 0] += 1
        components_arr[sprawl_area == 0] = 1
        mso.set_neighbourhood(neigh, dist)
        mso.set_components(components_arr, components_num+1)
        mso.set_use_background(False)
        mu_array = calculate_mu(data.copy('C'), lower_bound, upper_bound, MuType.base_mu)
        if arguments["reflective"]:
            mu_array[mu_array < 0.5] = 1 - mu_array[mu_array < 0.5]
        mso.set_mu_array(mu_array)
        mso.run_MSO(arguments["step_limits"])
        # print("Steps: ", mso.steps_done(), file=sys.stderr)
        result = mso.get_result_catted()
        result[result > 0] -= 1
        return result


sprawl_dict = Register(MSOSprawl, PathSprawl, DistanceSprawl, PathDistanceSprawl, FDTSprawl)


def get_neigh(sides):
    if sides:
        return NeighType.sides
    else:
        return NeighType.edges


class NeighType(Enum):
    sides = 6
    edges = 18
    vertex = 26


def calculate_distances_array(spacing, neigh_type: NeighType):
    """
    :param spacing:
    :param neigh_type:
    :return: neighbourhood array, distance array
    """
    min_dist = min(spacing)
    normalized_spacing = [x / min_dist for x in spacing]
    if len(normalized_spacing) == 2:
        neighbourhood_array = neighbourhood2d
        if neigh_type == NeighType.sides:
            neighbourhood_array = neighbourhood_array[:4]
        normalized_spacing = [0] + normalized_spacing
    else:
        neighbourhood_array = neighbourhood[:neigh_type.value]
    normalized_spacing = np.array(normalized_spacing)
    return neighbourhood_array, np.sqrt(np.sum((neighbourhood_array * normalized_spacing) ** 2, axis=1))


def get_neighbourhood(spacing, neigh_type: NeighType):
    if len(spacing) == 2:
        if neigh_type == NeighType.sides:
            return neighbourhood2d[:4]
        return neighbourhood2d
    else:
        return neighbourhood[:neigh_type.value]


neighbourhood = \
    np.array([[0, -1, 0], [0, 0, -1],
              [0, 1, 0], [0, 0, 1],
              [-1, 0, 0], [1, 0, 0],

              [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 1, 0],
              [-1, 0, -1], [1, 0, -1], [-1, 0, 1], [1, 0, 1],
              [0, -1, -1], [0, 1, -1], [0, -1, 1], [0, 1, 1],

              [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
              [1, 1, -1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]], dtype=np.int8)

neighbourhood2d = \
    np.array([[0, -1, 0], [0, 0, -1], [0, 1, 0], [0, 0, 1],
              [0, -1, -1], [0, 1, -1], [0, -1, 1], [0, 1, 1]], dtype=np.int8)
