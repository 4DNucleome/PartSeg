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
