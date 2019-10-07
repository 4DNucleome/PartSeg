"""
This module contains function to perform multiple component sprawl (watershed like)
"""
import logging
import os
import typing
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import numpy as np

from .euclidean_cython import calculate_euclidean
from .fuzzy_distance import fuzzy_distance, calculate_mu_array, MuType
from .path_sprawl_cython import calculate_maximum, calculate_minimum
from .sprawl_utils import get_maximum_component, get_closest_component, get_minimum_component


def path_maximum_sprawl(data_f: np.ndarray, components: np.ndarray, components_count: int,
                        neighbourhood: np.ndarray, distance_cache=None, data_cache=None):
    """
    Calculate sprawl in respect to brightens. Distance between voxels is minimum brightness on
    all paths connecting them.

    :param data_f: array with brightness
    :param components: core components as labeled array
    :param components_count: number of components
    :param neighbourhood: information about neighbourhood, shift array
    :param distance_cache: cache array, for reduce memory allocation. Shape of data_m, dtype is np.float64
    :param data_cache: cache array, for reduce memory allocation. Shape of data_m, dtype is data_m.dtype
        (use np.uint8 instead of np.bool)
    :return: array with updated labels
    """
    if data_cache is None:
        data_cache = np.zeros(data_f.shape, data_f.dtype)
    if components_count == 1:
        np.copyto(data_cache, data_f)
        tmp = calculate_maximum(data_cache, (components == 1).astype(np.uint8), neighbourhood)
        components[tmp > 0] = 1
        return components
    if distance_cache is None:
        distance_cache = np.zeros((components_count,) + data_f.shape, dtype=np.float64)
    for component in range(1, components_count + 1):
        np.copyto(data_cache, data_f)
        data_cache[(components > 0) * (components != component)] = 0
        distance_cache[component - 1] = calculate_maximum(data_cache, (components == component).astype(np.uint8),
                                                          neighbourhood)
    components = get_maximum_component(components, (data_f > 0).astype(np.uint8), distance_cache[:components_count],
                                       components_count)
    return components


def path_minimum_sprawl(data_f, components, components_count, neighbourhood, distance_cache=None, data_cache=None):
    """
    Calculate sprawl in respect to brightens. Distance between voxels is maximum brightness on
    all paths connecting them.

    :param data_f: array with brightness
    :param components: core components as labeled array
    :param components_count: number of components
    :param neighbourhood: information about neighbourhood, shift array
    :param distance_cache: cache array, for reduce memory allocation. Shape of data_m, dtype is np.float64
    :param data_cache: cache array, for reduce memory allocation. Shape of data_m, dtype is data_m.dtype
        (use np.uint8 instead of np.bool)
    :return: a
    """
    maximum = data_f.max()
    if data_cache is None:
        data_cache = np.zeros(data_f.shape, data_f.dtype)
    if components_count == 1:
        np.copyto(data_cache, data_f)
        tmp = calculate_minimum(data_cache, (components == 1).astype(np.uint8), neighbourhood)
        components[tmp < maximum] = 1
        return components
    if distance_cache is None:
        distance_cache = np.zeros((components_count,) + data_f.shape, dtype=np.float64)
    for component in range(1, components_count + 1):
        np.copyto(data_cache, data_f)
        data_cache[(components > 0) * (components != component)] = 0
        distance_cache[component - 1] = calculate_minimum(data_cache, (components == component).astype(np.uint8),
                                                          neighbourhood)
    components = get_minimum_component(components, (data_f > 0).astype(np.uint8), distance_cache[:components_count],
                                       components_count)
    return components


def euclidean_sprawl(data_m: np.ndarray, components: np.ndarray, components_count: int, neigh_arr, dist_arr,
                     distance_cache=None, data_cache=None):
    """
    Calculate euclidean sprawl (watershed)

    :param data_m: area for sprawl
    :param components: core components as labeled array
    :param components_count: number of components
    :param neigh_arr: information about neighbourhood, shift array
    :param dist_arr: information about neighbourhood, distance array for shifted position
    :param distance_cache: cache array, for reduce memory allocation. Shape of data_m, dtype is np.float64
    :param data_cache: cache array, for reduce memory allocation. Shape of data_m, dtype is data_m.dtype
        (use np.uint8 instead of np.bool)
    :return: array with updated labels
    """
    return distance_sprawl(calculate_euclidean, data_m, components, components_count,
                           neigh_arr, dist_arr, distance_cache, data_cache)


def fdt_sprawl(data_m: np.ndarray, components: np.ndarray, components_count: int, neigh_arr, dist_arr, lower_bound,
               upper_bound, distance_cache=None, data_cache=None):
    """
    Function for calculate fdt sprawl

    :param data_m: sprawl area
    :param components: core components as labeled array
    :param components_count: number of components
    :param neigh_arr: information about neighbourhood, shift array
    :param dist_arr: information about neighbourhood, distance array for shifted position
    :param lower_bound: lower bound for calculate mu value
    :param upper_bound: upper bound for calculate mu value
    :param distance_cache: cache array, for reduce memory allocation. Shape of data_m, dtype is np.float64
    :param data_cache: cache array, for reduce memory allocation. Shape of data_m, dtype is data_m.dtype
        (use np.uint8 instead of np.bool)
    :return:
    """
    if lower_bound > upper_bound:
        mu_array = 1 - calculate_mu_array(data_m, upper_bound, lower_bound, MuType.reflection_mu)
    else:
        mu_array = calculate_mu_array(data_m, lower_bound, upper_bound, MuType.reflection_mu)
    return distance_sprawl(partial(fuzzy_distance, mu_array=mu_array),
                           data_m, components, components_count, neigh_arr, dist_arr, distance_cache,
                           data_cache)


def sprawl_component(data_m: np.ndarray, components: np.ndarray, component_number: int,
                     neigh_arr: np.ndarray, dist_arr: np.ndarray, calculate_operator: typing.Callable):
    """
    calculate sprawl for single component

    :param data_m:
    :param components: array of components
    :param component_number: chosen component number
    :param neigh_arr: information about neighbourhood, shift array
    :param dist_arr: information about neighbourhood, distance array for shifted position
    :param calculate_operator: function to be called for calculate sprawl
    :return:
    """
    data_cache = np.copy(data_m)
    data_cache[(components > 0) * (components != component_number)] = 0
    return calculate_operator(data_cache, np.array(components == component_number).astype(np.uint8),
                              neigh_arr, dist_arr)


def distance_sprawl(calculate_operator, data_m: np.ndarray, components: np.ndarray, components_count: int, neigh_arr,
                    dist_array, distance_cache=None, data_cache=None, parallel=False) -> np.ndarray:
    if data_m.dtype == np.bool:
        data_m = data_m.astype(np.uint8)
    if data_cache is None:
        data_cache = np.zeros(data_m.shape, data_m.dtype)
    if components_count == 1:
        np.copyto(data_cache, data_m)
        tmp = calculate_operator(data_cache, (components == 1).astype(np.uint8), neigh_arr, dist_array)
        components[tmp < 2 ** 17] = 1
        return components
    if distance_cache is None:
        distance_cache = np.zeros((components_count,) + data_m.shape, dtype=np.float64)
    if parallel:
        workers_num = os.cpu_count()
        if not workers_num:
            workers_num = 4
        with ThreadPoolExecutor(max_workers=workers_num) as executor:
            result_info = {executor.submit(
                sprawl_component, data_m, components, component_number, neigh_arr, dist_array, calculate_operator):
                               component_number for component_number in range(1, components_count + 1)}
            for result in as_completed(result_info):
                component_number = result_info[result]
                distance_cache[component_number-1] = result.result()

    else:
        for component in range(1, components_count + 1):
            np.copyto(data_cache, data_m)
            data_cache[(components > 0) * (components != component)] = 0
            distance_cache[component - 1] = calculate_operator(data_cache, (components == component).astype(np.uint8),
                                                               neigh_arr, dist_array)
    components = get_closest_component(components, (data_m > 0).astype(np.uint8), distance_cache[:components_count],
                                       components_count)
    return components


def reverse_permutation(perm):
    rev = [0] * len(perm)
    for i, x in enumerate(perm, 1):
        rev[x - 1] = i
    return rev


def relabel_with_perm(labeling, perm):
    logging.debug(f"{labeling}, {perm}")
    perm = reverse_permutation(perm)
    return [perm[x] for x in labeling]


def verify_cohesion(elements: typing.List[int], graph):
    start = elements[0]
    elements_set = set(elements)
    elements_set.remove(start)
    queue = []
    queue.extend(graph[start])
    while len(queue) > 0 and len(elements_set) > 0:
        el = queue.pop()
        if el in elements_set:
            elements_set.remove(el)
            queue.extend(graph[el])
    return len(elements_set) == 0


def relabel_array(data: np.ndarray, perm):
    result = np.zeros(data.shape, dtype=data.dtype)
    for i, val in enumerate(perm):
        result[data == i + 1] = val
    return result
