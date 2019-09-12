# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# cython: language_level=3

import numpy as np
cimport numpy as np
from numpy cimport float64_t, int8_t, uint8_t
from cpython.mem cimport PyMem_Malloc
from .distance_utils cimport Point, my_queue, Size, component_types

ctypedef fused object_area_types:
    float64_t
    uint8_t


cdef void put_borders_in_queue(my_queue[Point] & current_points,
                                      np.ndarray[uint8_t, ndim=3] base_object,
                                      np.ndarray[int8_t, ndim=2] neighbourhood):
    cdef Size x_size, y_size, z_size, x, y, z, xx, yy, zz
    cdef Point p, p1
    cdef char neigh_length = neighbourhood.shape[0]
    z_size = base_object.shape[0]
    y_size = base_object.shape[1]
    x_size = base_object.shape[2]
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
                        if base_object[zz, yy, xx] == 0:
                            p.z = z
                            p.y = y
                            p.x = x
                            current_points.push(p)
                            break


cdef my_queue[Point] * create_borders_queues(np.ndarray[component_types, ndim=3] base_object,
                                      np.ndarray[int8_t, ndim=2] neighbourhood, component_types components_num):
    cdef Size x_size, y_size, z_size, x, y, z, xx, yy, zz
    cdef Point p, p1
    cdef char neigh_length = neighbourhood.shape[0]
    cdef my_queue[Point] * result = <my_queue[Point] *> PyMem_Malloc(components_num * sizeof(my_queue[Point]))
    cdef component_types index
    for index in range(components_num):
        result[index] = my_queue[Point]()
    z_size = base_object.shape[0]
    y_size = base_object.shape[1]
    x_size = base_object.shape[2]
    for z in range(1, z_size-1):
        for y in range(1, y_size-1):
            for x in range (1, x_size-1):
                if base_object[z,y,x] > 0:
                    index = base_object[z,y,x]
                    for neigh_it in range(neigh_length):
                        zz = z+neighbourhood[neigh_it, 0]
                        yy = y+neighbourhood[neigh_it, 1]
                        xx = x+neighbourhood[neigh_it, 2]
                        if xx == -1 or xx == x_size or xx == -1 or yy == y_size or zz == -1 or zz == z_size:
                            continue
                        if base_object[zz, yy, xx] != index:
                            p.z = z
                            p.y = y
                            p.x = x
                            result[index].push(p)
    return result