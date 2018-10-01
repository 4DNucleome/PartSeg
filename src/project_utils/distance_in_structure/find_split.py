import numpy as np
from .euclidean_cython import calculate_euclidean
from .path_sprawl_cython import calculate_maximum, calculate_minimum
import typing

import logging

from .sprawl_utils import get_maximum_component, get_closest_component, get_minimum_component


def compare_maximum(index, matrix_list):
    matrix = matrix_list[index]
    result = np.ones(matrix.shape, dtype=np.bool)
    for i, om in enumerate(matrix_list):
        if i == index:
            continue
        result &= (matrix > om)
    return result


def compare_distance(index, matrix_list):
    matrix = matrix_list[index]
    result = np.ones(matrix.shape, dtype=np.bool)
    for i, om in enumerate(matrix_list):
        if i == index:
            continue
        result &= (matrix < om)
    return result




def path_maximum_sprawl(data_f, components, components_count, distance_cache=None, data_cache=None):
    if data_cache is None:
        data_cache = np.zeros(data_f.shape, data_f.dtype)
    if components_count == 1:
        np.copyto(data_cache, data_f)
        tmp = calculate_maximum(data_cache, (components == 1).astype(np.uint8), 2)
        components[tmp > 0] = 1
        return components
    if distance_cache is None:
        distance_cache = np.zeros((components_count,) + data_f.shape, dtype=np.float64)
    for component in range(1, components_count + 1):
        np.copyto(data_cache, data_f)
        data_cache[(components > 0) * (components != component)] = 0
        distance_cache[component - 1] = calculate_maximum(data_cache, (components == component).astype(np.uint8), 1 if  data_f.shape[0] > 2 else 0)
    components = get_maximum_component(components, (data_f > 0).astype(np.uint8), distance_cache[:components_count], components_count)
    return components

def path_minimum_sprawl(data_f, components, components_count, distance_cache=None, data_cache=None):
    maximum = data_f.max()
    if data_cache is None:
        data_cache = np.zeros(data_f.shape, data_f.dtype)
    if components_count == 1:
        np.copyto(data_cache, data_f)
        tmp = calculate_minimum(data_cache, (components == 1).astype(np.uint8), 0, maximum)
        components[tmp < maximum] = 1
        return components
    if distance_cache is None:
        distance_cache = np.zeros((components_count,) + data_f.shape, dtype=np.float64)
    for component in range(1, components_count + 1):
        np.copyto(data_cache, data_f)
        data_cache[(components > 0) * (components != component)] = 0
        distance_cache[component - 1] = calculate_minimum(data_cache, (components == component).astype(np.uint8), 1 if  data_f.shape[0] > 2 else 0,  maximum)
    components = get_minimum_component(components, (data_f > 0).astype(np.uint8), distance_cache[:components_count], components_count)
    return components


def distance_sprawl(data_m: np.ndarray, components: np.ndarray, components_count: int, distance_cache=None, data_cache=None):

    if data_cache is None:
        data_cache = np.zeros(data_m.shape, data_m.dtype)
    if components_count == 1:
        np.copyto(data_cache, data_m)
        tmp = calculate_euclidean(data_cache, (components == 1).astype(np.uint8), 0)
        components[tmp < 2**17] = 1
        return components
    if distance_cache is None:
        distance_cache = np.zeros((components_count,) + data_m.shape, dtype=np.float64)
    for component in range(1, components_count + 1):
        np.copyto(data_cache, data_m)
        data_cache[(components > 0) * (components != component)] = 0
        distance_cache[component - 1] = calculate_euclidean(data_cache, (components == component).astype(np.uint8), 1 if  data_m.shape[0] > 2 else 0)
    else:
        components = get_closest_component(components, (data_m > 0).astype(np.uint8), distance_cache[:components_count], components_count)
    return components


def sink_components(neighbourhood: dict):
    one_connection = []
    zero_connection = []
    for k, v in neighbourhood.items():
        if len(v) == 0:
            zero_connection.append(k)
        if len(v) == 1:
            one_connection.append(k)
    return one_connection, zero_connection

def reverse_permutation(perm):
    rev = [0] * len(perm)
    for i, x in enumerate(perm, 1):
        rev[x-1] = i
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
        result[data == i+1] = val
    return result


