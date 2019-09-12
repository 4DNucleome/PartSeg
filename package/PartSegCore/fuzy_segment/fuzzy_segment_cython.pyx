# cython: boundscheck=False, wraparound=False, nonecheck=False
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

import numpy as np
cimport numpy as np

from libc.math cimport ceil

ctypedef fused numpy_types:
    np.uint8_t
    np.uint16_t



"""np.uint32_t
    np.int8_t
    np.int16_t
    np.int32_t
    np.float32_t
    np.float64_t"""

ctypedef fused numpy_arrays_3:
    np.ndarray[np.uint8_t, ndim=3]
    np.ndarray[np.uint16_t, ndim=3]
    np.ndarray[np.uint32_t, ndim=3]
    np.ndarray[np.int8_t, ndim=3]
    np.ndarray[np.int16_t, ndim=3]
    np.ndarray[np.int32_t, ndim=3]
    np.ndarray[np.float32_t, ndim=3]
    np.ndarray[np.float64_t, ndim=3]


VV = np.array([[-1, -1, -1], [0, -1, -1], [+1, -1, -1],
              [-1, 0, -1], [0, 0, -1], [+1, 0, -1],
              [-1, +1, -1], [0, +1, -1], [+1, +1, -1],

              [-1, -1, 0], [0, -1, 0], [+1, -1, 0],
              [-1, 0, 0], [+1, 0, 0],
              [-1, +1, 0], [0, +1, 0], [+1, +1, 0],

              [-1, -1, +1], [0, -1, +1], [+1, -1, +1],
              [-1, 0, +1], [0, 0, +1], [+1, 0, +1],
              [-1, +1, +1], [0, +1, +1], [+1, +1, +1]], dtype=np.uint8)

R2 = 1.41
R3 = 1.73


DD = np.array([R3, R2, R3,
              R2, 1, R2,
              R3, R2, R3,

              R2, 1, R2,
              1, 1,
              R2, 1, R2,

              R3, R2, R3,
              R2, 1, R2,
              R3, R2, R3], dtype=np.float32)

cdef struct Borders:
    double tap
    double ta
    double tb
    double tbp

ctypedef double (*MU)(double x, Borders b)

ctypedef void (*ADD_STEP)(Py_ssize_t x, Py_ssize_t y, Py_ssize_t z, Py_ssize_t i, void * data)

cdef void none_fun(Py_ssize_t x, Py_ssize_t y, Py_ssize_t z, Py_ssize_t i, void * data):
    return

cdef void move_maximum(Py_ssize_t x, Py_ssize_t y, Py_ssize_t z, Py_ssize_t i, void * data):
    cdef np.ndarray[np.int16_t, ndim=4] array = <np.ndarray[np.int16_t, ndim=4]> data
    cdef np.uint8_t [:, :] V = VV
    array[z,y,x,0] = array[z + V[i, 2], y + V[i, 1], x + V[i, 0],0]
    array[z,y,x,1] = array[z + V[i, 2], y + V[i, 1], x + V[i, 0],1]
    array[z,y,x,2] = array[z + V[i, 2], y + V[i, 1], x + V[i, 0],2]

cdef inline double shrink(double res):
    if res < 0:
        res = 0
    elif res > 1:
        res = 1
    return res

cdef double mu_flag0(double x, Borders b):
    cdef double res
    res = 1 - (x - b.ta) / (b.tb - b.ta)
    return shrink(res)

cdef double mu_flag2(double x, Borders b):
    cdef double res
    res = (x - b.ta) / (b.tb - b.ta)
    if res < 0.5:
        res = 1 - res
    return shrink(res)

cdef double mu_flag5(double x, Borders b):
    if (b.ta - b.tap > 0 and x >= b.tap and x <= b.ta):
        return shrink((x - b.tap) / (b.ta - b.tap))
    elif (b.tb - b.ta > 0 and x > b.ta and x <= b.tb):
        return shrink((b.tb - x) / (b.tb - b.ta))

cdef inline one_step_FDT(numpy_arrays_3 image, np.ndarray[np.uint16_t, ndim=3] lFDT,
                              Borders b, MU fun, Py_ssize_t x, Py_ssize_t y, Py_ssize_t z,
                              Py_ssize_t n_begin, Py_ssize_t n_end, ADD_STEP fun2, void * data):
    if lFDT[z, y, x] == 0:
        return

    cdef np.uint8_t [:, :] V = VV
    cdef np.float32_t [:] D = DD
    cdef Py_ssize_t i

    cdef double min_val, pixel_val, mu_p, mu_q, temp

    mu_p = fun(image[z, x, y], b)
    min_val = lFDT[z, y, x]
    for i in range(n_begin, n_end):
        pixel_val = lFDT[z + V[i, 2], y + V[i, 1], x + V[i, 0]]
        if pixel_val == 0:
            mu_q = 0
        else:
            mu_q = fun(pixel_val, b)
        temp = pixel_val + 50 * (mu_p + mu_q) * D[i]
        temp = ceil(temp - 0.5)
        if temp < min_val:
            lFDT[z, y, x] = <np.uint16_t> temp
            min_val = temp
            fun2(x,y,z,i,data)

cdef compute_FDT_core(numpy_arrays_3 image, np.ndarray[np.uint16_t, ndim=3] lFDT,
                Borders b, MU fun, ADD_STEP fun2, void * data):
    cdef Py_ssize_t x_size, y_size, z_size, x, y, z, i, F_count
    x_size = image.shape[2]
    y_size = image.shape[1]
    z_size = image.shape[0]
    F_count = 0

    for z in range(1, z_size - 1):
        for y in range(1, y_size - 1):
            for x in range(1, x_size - 1):
                one_step_FDT(image, lFDT, b, fun, x, y, z, 0, 13, fun2, data)

    for z in range(z_size-2, 0, -1):
        for y in range(y_size-2, 0 -1):
            for x in range(x_size-2, 0, -1):
                one_step_FDT(image, lFDT, b, fun, x, y, z, 13, 26, fun2, data)

def compute_FDT(numpy_arrays_3 image, np.ndarray[np.uint16_t, ndim=3] lFDT, np.ndarray[np.uint8_t, ndim=3] SS,
                double tap, double ta, double tb, double tbp, int type):
    lFDT[SS == 3] = 0
    cdef Borders b
    b.tap = tap
    b.ta = ta
    b.tb = tb
    b.tbp = tbp
    if type == 0:
        compute_FDT_core(image, lFDT, b, mu_flag0, none_fun, <void *> None)
    elif type == 2:
        compute_FDT_core(image, lFDT, b, mu_flag2, none_fun, <void *> None)
    elif type == 5:
        compute_FDT_core(image, lFDT, b, mu_flag5, none_fun, <void *> None)

def compute_FDT_with_move(numpy_arrays_3 image, np.ndarray[np.uint16_t, ndim=3] lFDT, np.ndarray[np.uint8_t, ndim=3] SS,
                double tap, double ta, double tb, double tbp, int type, data):
    lFDT[SS == 3] = 0
    cdef Borders b
    b.tap = tap
    b.ta = ta
    b.tb = tb
    b.tbp = tbp
    if type == 0:
        compute_FDT_core(image, lFDT, b, mu_flag0, move_maximum, <void *> data)
    elif type == 2:
        compute_FDT_core(image, lFDT, b, mu_flag2, move_maximum, <void *> data)
    elif type == 5:
        compute_FDT_core(image, lFDT, b, mu_flag5, move_maximum, <void *> data)


def (np.ndarray[np.uint16_t, ndim=3] )