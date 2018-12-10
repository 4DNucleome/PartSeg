# cython: profile=True, boundscheck=False, wraparound=False, nonecheck=False
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

import numpy as np
cimport numpy as np

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

ctypedef fused numpy_types:
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.int8_t
    np.int16_t
    np.int32_t
    np.float32_t
    np.float64_t

cdef inline double linear_interpolation(double q1, double q2, double x1, double x2, double x):
    return (q1 * (x1 - x) + q2 * (x - x2)) / (x1 - x2)

cdef inline double bilinear_interpolation(double q11, double q12, double q21, double q22, double x1, double x2,
                                          double y1, double y2, double x, double y):
    cdef double x2x1, y2y1, x2x, y2y, yy1, xx1
    x2x1 = x2 - x1
    y2y1 = y2 - y1
    x2x = x2 - x
    y2y = y2 - y
    yy1 = y - y1
    xx1 = x - x1
    return 1.0 / (x2x1 * y2y1) * (
        q11 * x2x * y2y +
        q21 * xx1 * y2y +
        q12 * x2x * yy1 +
        q22 * xx1 * yy1
    )

def bilinear2d(np.ndarray[numpy_types, ndim=2] array, scale_y, scale_x):
    cdef Py_ssize_t size_x, size_y, x, y, x1, x2
    size_y = array.shape[0] * scale_y
    size_x = array.shape[1] * scale_x
    cdef np.ndarray[np.float64_t, ndim=1] cords_y, cords_x
    cdef np.ndarray[np.uint16_t, ndim=1] floor_y, floor_x
    cords_y = np.linspace(0, array.shape[0], size_y)
    cords_x = np.linspace(0, array.shape[1], size_x)
    floor_x = np.floor(cords_x)
    floor_y = np.floor(cords_y)
    cdef numpy_types q11, q12, q21, q22
    cdef double rx, ry
    cdef np.ndarray[numpy_types, ndim=2] res = np.zeros(shape=(size_y, size_x), dtype=array.dtype)
    for y in range(size_y):
        for x in range(size_x):
            x1 = floor_x[x]
            y1 = floor_y[y]
            q11 = array[y, x]
            q12 = array[y, x + 1]
            q21 = array[y + 1, x]
            q22 = array[y + 1, x + 1]
            rx = cords_x[x]
            ry = cords_y[y]
            res[x, y] = <numpy_types> bilinear_interpolation(q11, q12, q21, q22, x1, x1 + 1, y1, y1 + 1, rx, ry)

    return res

def bilinear2d(np.ndarray[numpy_types, ndim=2] array, double scale_y, double scale_x):
    cdef Py_ssize_t size_x, size_y, x, y, y1, x1
    cdef np.ndarray[np.float64_t, ndim=1] cords_y, cords_x
    cdef np.ndarray[np.intp_t, ndim=1] floor_y, floor_x
    cdef double rx, ry, inter_res
    size_y = <Py_ssize_t> (array.shape[0] * scale_y)
    size_x = <Py_ssize_t> (array.shape[1] * scale_x)
    cords_y = np.linspace(0, array.shape[0] - 1, size_y)
    cords_x = np.linspace(0, array.shape[1] - 1, size_x)
    floor_x = np.floor(cords_x).astype(np.intp)
    floor_y = np.floor(cords_y).astype(np.intp)
    cdef np.ndarray[np.float64_t, ndim=2] res = \
        np.zeros(shape=(size_y, size_x), dtype=np.float64)
    for y in range(size_y - 1):
        y1 = floor_y[y]
        ry = cords_y[y]
        for x in range(size_x - 1):
            x1 = floor_x[x]
            rx = cords_x[x]
            q11 = <double> array[y1, x1]
            q12 = <double> array[y1, x1 + 1]
            q21 = <double> array[y1 + 1, x1]
            q22 = <double> array[y1 + 1, x1 + 1]
            inter_res = bilinear_interpolation(q11, q12, q21, q22, <double> x1, <double> x1 + 1, <double> y1,
                                               <double> y1 + 1, rx, ry)
            res[x, y] = inter_res
        # border case 1
        x = size_x - 1
        x1 = floor_x[x]
        q1 = <double> array[y1, x1]
        q2 = <double> array[y1 + 1, x1]
        rx = cords_x[x]
        inter_res = linear_interpolation(q1, q2, <double> y1, <double> y1 + 1, ry)
        res[x, y] = inter_res
    u = size_y - 1
    y1 = floor_y[y]
    ry = cords_y[y]
    for x in range(size_x - 1):
        x1 = floor_x[x]
        rx = cords_x[x]
        q1 = <double> array[y1, x1]
        q2 = <double> array[y1, x1 + 1]
        inter_res = linear_interpolation(q1, q2, <double> x1, <double> x1 + 1, rx)
        res[x, y] = inter_res

    return res.astype(array.dtype)
