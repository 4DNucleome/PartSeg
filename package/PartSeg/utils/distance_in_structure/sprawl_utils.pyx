# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, embedsignature=True
# cython: profile=True
# cython: language_level=3

import numpy as np
cimport numpy as np

ctypedef fused image_types:
    np.float32_t
    np.float64_t
    np.uint8_t
    np.uint16_t
    np.uint32_t

def get_maximum_component(components, data_mask, paths, num_of_components=None):
    if num_of_components is None:
        num_of_components = len(paths)
    new_paths = paths.reshape((num_of_components, components.size))
    _get_maximum_component(components.ravel(), data_mask.ravel(), new_paths, num_of_components)
    return components


def _get_maximum_component(np.ndarray[np.uint32_t] components, np.ndarray[np.uint8_t] data_mask, np.ndarray[image_types, ndim=2] paths, int num_of_components):
    cdef Py_ssize_t x, y, size, component_index
    cdef image_types max_val, val
    cdef char flag
    size = components.size
    for x in range(size):
        flag = False
        if components[x] == 0 and data_mask[x]:
            max_val = paths[0, x]
            component_index = 1
            flag = True
            for y in range(1, num_of_components):
                val = paths[y, x]
                if val > max_val:
                    max_val = val
                    component_index = y+1
                    flag = True
                elif val == max_val:
                    flag = False
            if flag:
                components[x] = component_index
    return components

def get_minimum_component(components, data_mask, paths, num_of_components=None):
    if num_of_components is None:
        num_of_components = len(paths)
    new_paths = paths.reshape((num_of_components, components.size))
    _get_minimum_component(components.ravel(), data_mask.ravel(), new_paths, num_of_components)
    return components


def _get_minimum_component(np.ndarray[np.uint32_t] components, np.ndarray[np.uint8_t] data_mask, np.ndarray[image_types, ndim=2] paths, int num_of_components):
    cdef Py_ssize_t x, y, size, component_index
    cdef image_types min_val, val
    cdef char flag
    size = components.size
    for x in range(size):
        flag = False
        if components[x] == 0 and data_mask[x]:
            min_val = paths[0, x]
            component_index = 1
            flag = True
            for y in range(1, num_of_components):
                val = paths[y, x]
                if val < min_val:
                    min_val = val
                    component_index = y+1
                    flag = True
                elif val == min_val:
                    flag = False
            if flag:
                components[x] = component_index
    return components


def get_closest_component(components, data_mask, distances, num_of_components=None):
    if num_of_components is None:
        num_of_components = len(distances)
    new_distances = distances.reshape((num_of_components, components.size))
    _get_closest_component(components.ravel(), data_mask.ravel(), new_distances, num_of_components)
    return components


def _get_closest_component(np.ndarray[np.uint32_t] components, np.ndarray[np.uint8_t] data_mask, np.ndarray[image_types, ndim=2] paths, int num_of_components):
    cdef Py_ssize_t x, y, size, component_index
    cdef image_types min_val, val
    cdef char flag
    size = components.size
    for x in range(size):
        flag = False
        if components[x] == 0 and data_mask[x]:
            min_val = paths[0, x]
            component_index = 1
            flag = True
            for y in range(1, num_of_components):
                val = paths[y, x]
                if val < min_val:
                    min_val = val
                    component_index = y+1
                    flag = True
                elif val == min_val:
                    flag = False
            if flag:
                components[x] = component_index
    return components