# distutils: define_macros=CYTHON_TRACE_NOGIL=1
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, embedsignature=True


import numpy as np
cimport numpy as np
from cython.parallel import prange

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


def calculate_borders(np.ndarray[numpy_types, ndim=3] image, int border_thick, bool per_layer=True):
    cdef Py_ssize_t x, y, z
    cdef int R, Y, Z
    circle_list = []
    if numpy_types.shape[0] == 1:
        per_layer = True
    if border_thick > 0:
        R = border_thick
        if per_layer:
            for x in range(-R,R+1):
                Y = int((R*R-x*x)**0.5) # bound for y given x
                for y in range(-Y,Y+1):
                    circle_list.append([0, x, y])
        else:
             for x in range(-R,R+1):
                Y = int((R*R-x*x)**0.5) # bound for y given x
                for y in range(-Y,Y+1):
                    Z = int((R*R - x*x - y*y)**0.5)
                    for z in range(-Z, Z+1):
                        circle_list.append([z, x, y])
        circle_shift = np.array(circle_list).astype(np.int8)
    else:
        circle_shift = np.array([[0,0,0]]).astype(np.int8)


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

def add_labels(np.ndarray[DTYPE_t, ndim=3] image, np.ndarray[label_types, ndim=2] labels, float overlay, int only_border,
              int border_thick, np.ndarray[np.uint8_t, ndim=1] use_labels, np.ndarray[np.uint8_t, ndim=2] label_colors):
    """
    Add label to given RGB image. Ift modify original image.

    :param image: RGB image on background
    :param labels: labels, different label is different num
    :param overlay: from 0 to 1
    :param only_border: Draw only border of image
    :param border_thick: if draw only border then thick of this border
    :param use_labels: Which labels should be drawer
    :return: changed image
    """
    if labels is None:
        return image
    cdef Py_ssize_t x_max = image.shape[0]
    cdef Py_ssize_t y_max = image.shape[1]
    cdef Py_ssize_t x, y
    cdef Py_ssize_t comp_num = use_labels.size
    cdef int R, Y
    cdef Py_ssize_t circle_len, circle_pos, labels_colors_num
    cdef np.ndarray[np.int8_t, ndim=2] circle_shift
    cdef label_types label_val

    cdef np.ndarray[label_types, ndim=2] local_labels

    # cdef np.uint8_t [:,:] label_colors = label_colors_global
    labels_colors_num = label_colors.shape[0]
    # prevent from usage background
    use_labels[0] = 0

    if only_border:
        circle_list = []
        if border_thick > 0:
            R = border_thick
            for x in range(-R,R+1):
                Y = int((R*R-x*x)**0.5) # bound for y given x
                for y in range(-Y,Y+1):
                    circle_list.append([x,y])
            circle_shift = np.array(circle_list).astype(np.int8)
        else:
            circle_shift = np.array([[0,0]]).astype(np.int8)
        circle_shift += border_thick
        circle_len = circle_shift.shape[0]
        local_labels = np.zeros((labels.shape[0] + 2*border_thick, labels.shape[1] + 2*border_thick), dtype=labels.dtype)
        for x in range(1,x_max-1):
            for y in range(1,y_max-1):
                label_val =labels[x,y]
                if use_labels[label_val] and (labels[x+1,y] != label_val or labels[x-1, y] != label_val or labels[x, y+1] != label_val or labels[x, y-1] != label_val):
                    for circle_pos in range(circle_len):
                        local_labels[x + circle_shift[circle_pos, 0], y + circle_shift[circle_pos, 1]] = labels[x,y]

        if border_thick > 0:
            local_labels = local_labels[border_thick:-border_thick, border_thick:-border_thick]
    else:
        local_labels = np.copy(labels)
        for i in range(1, comp_num):
            if not use_labels[i]:
                local_labels[local_labels == i] = 0
    cdef float part1, part2
    cdef label_types col_num
    for x in prange(x_max, nogil=True):
        for y in range(y_max):
            if local_labels[x,y] > 0:
                col_num = (local_labels[x,y] -1) % labels_colors_num
                image[x, y, 0] = <DTYPE_t> (image[x, y, 0] * (1-overlay) + label_colors[col_num, 0] * overlay)
                image[x, y, 1] = <DTYPE_t> (image[x, y, 1] * (1-overlay) + label_colors[col_num, 1] * overlay)
                image[x, y, 2] = <DTYPE_t> (image[x, y, 2] * (1-overlay) + label_colors[col_num, 2] * overlay)
    # local_labels = labels_color_map[labels]
    return image
