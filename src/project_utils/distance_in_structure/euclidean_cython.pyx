# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

from libcpp.queue cimport queue
from libcpp cimport bool

import numpy as np
cimport numpy as np

ctypedef np.uint16_t Size

cdef extern from 'my_queue.h':
    cdef cppclass my_queue[T]:
        my_queue() except +
        T front()
        void pop()
        void push(T& v)
        bool empty()

cdef extern from "global_consts.h":
    # const signed char neighbourhood[26][3]
    const char neigh_level[]
    # const float distance[]

cdef struct Point:
    Size x
    Size y
    Size z

def calculate_euclidean(np.ndarray[np.uint8_t, ndim=3] object_area, np.ndarray[np.uint8_t, ndim=3] base_object,
                        np.ndarray[np.uint8_t, ndim=2] neighbourhood, np.ndarray[np.float64_t, ndim=1] distance):
    cdef np.ndarray[np.uint8_t, ndim=3] consumed_area = np.copy(base_object)
    cdef np.ndarray[np.float64_t, ndim=3] result
    cdef Size x_size, y_size, z_size, array_pos, x, y, z
    cdef Py_ssize_t count = 0
    cdef char neigh_length = neighbourhood.shape[0]
    cdef int neigh_it
    cdef my_queue[Point] current_points, new_points
    cdef Point p, p1
    z_size = object_area.shape[0]
    y_size = object_area.shape[1]
    x_size = object_area.shape[2]
    result = np.zeros((z_size, y_size, x_size), dtype=np.float64)
    result[base_object == 0] = 2**17
    for z in range(1, z_size-1):
        for y in range(1, y_size-1):
            for x in range (1, x_size-1):
                if base_object[z,y,x] > 0:
                    for neigh_it in range(neigh_length):
                        if base_object[z+neighbourhood[neigh_it, 0], y+neighbourhood[neigh_it, 1], x+neighbourhood[neigh_it, 2]] == 0:
                            p.z = z
                            p.y = y
                            p.x = x
                            current_points.push(p)
                            break
    while not current_points.empty():
        p = current_points.front()
        current_points.pop()
        """count += 1
        if count > 10000:
            print("step", p)
            count = 0"
        if consumed_area[p.z, p.y, p.x] > 0:
            continue"""
        consumed_area[p.z, p.y, p.x] = 0
        for neigh_it in range(neigh_length):
            z = p.z + neighbourhood[neigh_it, 0]
            y = p.y + neighbourhood[neigh_it, 1]
            x = p.x + neighbourhood[neigh_it, 2]
            if x == 0 or y == 0 or z == 0 or x >= x_size-1 or y >= y_size-1 or z >= z_size-1:
                continue
            if object_area[z, y, x] == 0:
                continue
            if result[z, y, x] > result[p.z, p.y, p.x] + distance[neigh_it]:
                result[z, y, x] = result[p.z, p.y, p.x] + distance[neigh_it]
                if consumed_area[z, y, x] == 0:
                    consumed_area[z, y, x] = 1
                    p1.z = z
                    p1.y = y
                    p1.x = x
                    current_points.push(p1)
    return result


