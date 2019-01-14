# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# cython: language_level=3

import numpy as np
cimport numpy as np
from numpy cimport float64_t, int8_t, uint8_t

from .distance_utils cimport Point, my_queue, Size

ctypedef fused object_area_types:
    np.float64_t
    np.uint8_t


cdef void put_borders_in_queue(np.ndarray[object_area_types, ndim=3] object_area, my_queue[Point] & current_points,
                                      np.ndarray[uint8_t, ndim=3] base_object,
                                      np.ndarray[int8_t, ndim=2] neighbourhood):
    cdef Size x_size, y_size, z_size, x, y, z, xx, yy, zz
    cdef Point p, p1
    cdef char neigh_length = neighbourhood.shape[0]
    z_size = object_area.shape[0]
    y_size = object_area.shape[1]
    x_size = object_area.shape[2]
    for z in range(1, z_size-1):
        for y in range(1, y_size-1):
            for x in range (1, x_size-1):
                if base_object[z,y,x] > 0:
                    for neigh_it in range(neigh_length):
                        zz = z+neighbourhood[neigh_it, 0]
                        yy = y+neighbourhood[neigh_it, 1]
                        xx = x+neighbourhood[neigh_it, 2]
                        if xx == -1 or xx == x_size or xx == -1 or yy == y_size or zz == -1 or zz == z_size:
                            continue
                        if base_object[zz, yy, zz] == 0:
                            p.z = z
                            p.y = y
                            p.x = x
                            current_points.push(p)
                            break