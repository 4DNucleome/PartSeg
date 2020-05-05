# distutils: language = c++
# cython: wraparound=False, nonecheck=False, cdivision=True, embedsignature=True, unraisable_tracebacks=True
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

ctypedef fused component_types:
    np.uint32_t
    np.uint16_t
    np.uint8_t


def get_maximum_component(components, data_mask, paths, components_translation, num_of_components=None):
    if num_of_components is None:
        num_of_components = len(paths)
    new_paths = paths.reshape((paths.size//components.size, components.size))
    _get_maximum_component(components.ravel(), data_mask.ravel(), new_paths, num_of_components, components_translation)
    return components


def _get_maximum_component(np.ndarray[component_types] components, np.ndarray[np.uint8_t] data_mask,
                           np.ndarray[image_types, ndim=2] paths, int num_of_components,
                           np.ndarray[np.uint32_t] components_translation):
    cdef Py_ssize_t x, y, size, component_index
    cdef image_types max_val, val
    cdef char flag
    size = components.size
    for x in range(size):
        flag = False
        if components[x] == 0 and data_mask[x]:
            max_val = paths[0, x]
            component_index = components[0]
            flag = True
            for y in range(1, num_of_components+1):
                val = paths[y, x]
                if val > max_val:
                    max_val = val
                    component_index = components_translation[y]
                    flag = True
                elif val == max_val:
                    flag = False
            if flag:
                components[x] = component_index
                paths[0, x] = max_val
    return components

def get_minimum_component(components, data_mask, paths, components_translation, num_of_components=None):
    if num_of_components is None:
        num_of_components = len(paths)
    new_paths = paths.reshape((paths.size//components.size, components.size))
    _get_minimum_component(components.ravel(), data_mask.ravel(), new_paths, num_of_components, components_translation)
    return components


def _get_minimum_component(np.ndarray[component_types] components, np.ndarray[np.uint8_t] data_mask,
                           np.ndarray[image_types, ndim=2] paths, int num_of_components,
                           np.ndarray[np.uint32_t] components_translation):
    cdef Py_ssize_t x, y, size, component_index
    cdef image_types min_val, val
    cdef char flag
    size = components.size
    for x in range(size):
        flag = False
        if components[x] == 0 and data_mask[x]:
            min_val = paths[0, x]
            component_index = components[0]
            flag = True
            for y in range(1, num_of_components+1):
                val = paths[y, x]
                if val < min_val:
                    min_val = val
                    component_index = components_translation[y]
                    flag = True
                elif val == min_val:
                    flag = False
            if flag:
                components[x] = component_index
                paths[0, x] = min_val
    return components


def get_closest_component(components, data_mask, distances, components_translation, num_of_components=None):
    if num_of_components is None:
        num_of_components = len(distances)
    new_distances = distances.reshape((distances.size // components.size, components.size))
    _get_closest_component(components.ravel(), data_mask.ravel(), new_distances, num_of_components,
                           components_translation)
    return components


def _get_closest_component(np.ndarray[component_types] components, np.ndarray[np.uint8_t] data_mask,
                           np.ndarray[image_types, ndim=2] paths, int num_of_components,
                           np.ndarray[np.uint32_t] components_translation):
    cdef Py_ssize_t x, y, size, component_index, count1=0, count2=0, count3=0
    cdef image_types min_val, val
    cdef char flag
    size = components.size
    for x in range(size):
        flag = False
        if data_mask[x]:
            min_val = paths[0, x]
            component_index = components[x]
            flag = True
            for y in range(1, num_of_components+1):
                count3 += 1
                val = paths[y, x]
                if val < min_val:
                    count1 += 1
                    min_val = val
                    component_index = components_translation[y]
                    flag = True
                elif val == min_val:
                    flag = True
                    component_index = 0
            if flag:
                count2 += 1
                components[x] = component_index
                paths[0, x] = min_val
    return components, paths
