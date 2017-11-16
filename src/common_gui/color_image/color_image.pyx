# cython: profile=True, boundscheck=False, wraparound=False, nonecheck=False
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

import numpy as np
cimport numpy as np
DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

def min_max_calc_int(np.ndarray arr):
    cdef Py_ssize_t i
    cdef int min, max, val
    min = arr.flat[0]
    max = arr.flat[0]
    it = np.nditer([arr],
        flags=['external_loop','buffered'],
        op_flags=[['readwrite']])
    for x in it:
        for i in range(x.size):
            val = x[i]
            if val < min:
                min = val
            if val > max:
                max = val
    return min, max

def color_greyscale(np.ndarray[DTYPE_t, ndim=2] cmap, np.ndarray[DTYPE_t, ndim=2] image):
    cdef x_max = image.shape[0]
    cdef y_max = image.shape[1]
    cdef int val, x, y
    cdef np.ndarray[DTYPE_t, ndim=3] result_array = np.zeros((x_max, y_max, 3), dtype=DTYPE)
    for x in range(x_max):
        for y in range(y_max):
            val = image[x,y]
            result_array[x, y, 0] = cmap[val, 0]
            result_array[x, y, 1] = cmap[val, 1]
            result_array[x, y, 2] = cmap[val, 2]

    return result_array
