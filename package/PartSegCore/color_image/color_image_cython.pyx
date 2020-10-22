# distutils: define_macros=CYTHON_TRACE_NOGIL=1
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, embedsignature=True
import itertools

import numpy as np

cimport numpy as np

from cython.parallel import prange

from libcpp cimport bool


cpdef enum:
    resolution = 1024
"""current accuracy of coloring"""

cdef extern from "<algorithm>" namespace "std" nogil:
    T max[T](const T& v1, const T& v2)
    T min[T](const T& v1, const T& v2)

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

ctypedef fused numpy_types:
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.float32_t
    np.float64_t

ctypedef fused label_types:
    np.uint8_t
    np.uint16_t
    np.uint32_t

def min_max_calc_int(np.ndarray arr):
    cdef Py_ssize_t i
    cdef int min, max, val
    min = arr.flat[0]
    max = arr.flat[0]
    it = np.nditer([arr],
                   flags=['external_loop', 'buffered'],
                   op_flags=[['readwrite']])
    for x in it:
        for i in range(x.size):
            val = x[i]
            if val < min:
                min = val
            if val > max:
                max = val
    return min, max

cdef inline long scale_factor(numpy_types value, double min_val, double factor) nogil:
    return max[long](0, min[long](resolution - 1, <long>((value - min_val) * factor)))


def calculate_borders(np.ndarray[label_types, ndim=4] labels, int border_thick, bool per_layer=True):
    if labels.shape[1] < 3:
        per_layer = True
    if per_layer:
        return calculate_borders2d(labels, border_thick)
    cdef Py_ssize_t x, y, z, t
    cdef int R, Y, Z
    cdef Py_ssize_t x_max = labels.shape[3]
    cdef Py_ssize_t y_max = labels.shape[2]
    cdef Py_ssize_t z_max = labels.shape[1]
    cdef Py_ssize_t t_max = labels.shape[0]
    cdef Py_ssize_t circle_len
    cdef np.ndarray[np.int8_t, ndim=2] circle_shift
    cdef np.ndarray[label_types, ndim=4] local_labels
    cdef label_types label_val

    R = border_thick**2
    values = list(range(-border_thick, border_thick+1))
    circle_list = []
    for x in values:
        for y in values:
            for z in values:
                if (x**2 + y**2 + z **2 <= R):
                    circle_list.append((x,y,z))
    circle_shift = np.array(circle_list).astype(np.int8)
    circle_len = circle_shift.shape[0]
    local_labels = np.zeros((labels.shape[0], labels.shape[1] + 2*border_thick, labels.shape[2] + 2*border_thick, labels.shape[3] + 2*border_thick), dtype=labels.dtype)
    circle_shift = circle_shift + border_thick
    for t in range(0, t_max):
        for z in range(1, z_max-1):
            for y in range(1,y_max-1):
                for x in range(1,x_max-1):
                    label_val = labels[t, z, y, x]
                    if label_val and (labels[t, z, y+1, x] != label_val or labels[t, z, y-1, x] != label_val or
                                      labels[t, z, y, x+1] != label_val or labels[t, z, y, x-1] != label_val or
                                      labels[t, z+1, y, x] != label_val or labels[t, z-1, y, x] != label_val):
                        for circle_pos in range(circle_len):
                            local_labels[t, z + circle_shift[circle_pos, 0], y + circle_shift[circle_pos, 1], x + circle_shift[circle_pos, 2]] = label_val
    if border_thick > 0:
        local_labels = local_labels[:, border_thick:-border_thick, border_thick:-border_thick, border_thick:-border_thick]
    return local_labels


def calculate_borders2d(np.ndarray[label_types, ndim=4] labels, int border_thick):
    cdef Py_ssize_t x, y, z, t
    cdef int R, Y, Z
    cdef Py_ssize_t x_max = labels.shape[3]
    cdef Py_ssize_t y_max = labels.shape[2]
    cdef Py_ssize_t z_max = labels.shape[1]
    cdef Py_ssize_t t_max = labels.shape[0]
    cdef Py_ssize_t circle_len
    cdef np.ndarray[np.int8_t, ndim=2] circle_shift
    cdef np.ndarray[label_types, ndim=4] local_labels
    cdef label_types label_val

    R = border_thick**2
    values = list(range(-border_thick, border_thick+1))
    circle_list = []
    for x in values:
        for y in values:
            if (x**2 + y**2 <= R):
                circle_list.append((x,y))
    circle_shift = np.array(circle_list).astype(np.int8)
    circle_len = circle_shift.shape[0]
    local_labels = np.zeros((labels.shape[0], labels.shape[1], labels.shape[2] + 2*border_thick, labels.shape[3] + 2*border_thick), dtype=labels.dtype)
    circle_shift = circle_shift + border_thick
    for t in range(0, t_max):
        for z in range(0, z_max):
            for y in range(1,y_max-1):
                for x in range(1,x_max-1):
                    label_val = labels[t, z, y, x]
                    if label_val and (labels[t, z, y+1, x] != label_val or labels[t, z, y-1, x] != label_val or
                                      labels[t, z, y, x+1] != label_val or labels[t, z, y, x-1] != label_val):
                        for circle_pos in range(circle_len):
                            local_labels[t, z, y + circle_shift[circle_pos, 0], x + circle_shift[circle_pos, 1]] =\
                                label_val
    if border_thick > 0:
        local_labels = local_labels[:, :, border_thick:-border_thick, border_thick:-border_thick]
    return local_labels


def color_grayscale(np.ndarray[DTYPE_t, ndim=2] cmap, np.ndarray[numpy_types, ndim=2] image, double min_val,
                    double max_val, int single_channel=-1):
    """color image channel in respect to cmap array. Array should be in size (resolution, 3)"""
    # TODO maybe not use empty channels
    cdef Py_ssize_t x_max = image.shape[0]
    cdef Py_ssize_t y_max = image.shape[1]
    cdef int val, x, y
    cdef double factor =  1/ ((max_val - min_val) / (resolution - 1))
    cdef np.ndarray[DTYPE_t, ndim=3] result_array = np.zeros((x_max, y_max, 3), dtype=DTYPE)
    with nogil:
        if 0 <= single_channel <= 2:
            for x in prange(x_max):
                for y in range(y_max):
                    val = scale_factor(image[x, y], min_val, factor)
                    result_array[x, y, single_channel] = cmap[val, single_channel]
        else:
            for x in prange(x_max):
                for y in range(y_max):
                    val = scale_factor(image[x, y], min_val, factor)
                    result_array[x, y, 0] = cmap[val, 0]
                    result_array[x, y, 1] = cmap[val, 1]
                    result_array[x, y, 2] = cmap[val, 2]

    return result_array
