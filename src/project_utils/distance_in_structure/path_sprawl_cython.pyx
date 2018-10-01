# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# cython: profile=True


from libcpp.queue cimport queue
from libcpp cimport bool
import numpy as np
cimport numpy as np

ctypedef np.int16_t Size

cdef extern from 'my_queue.h':
    cdef cppclass my_queue[T]:
        my_queue() except +
        T front()
        void pop()
        void push(T& v)
        bool empty()
        size_t get_size()

cdef extern from "global_consts.h":
    const signed char neighbourhood[26][3]
    const char neigh_level[]
    const float distance[]

cdef struct Point:
    Size x
    Size y
    Size z

def calculate_maximum(np.ndarray[np.float64_t, ndim=3] object_area, np.ndarray[np.uint8_t, ndim=3] base_object, int level, result=None):
    """
    Calculate maximum path from source
    :param object_area: data with removed other components
    :param base_object: area of component
    :param level: type of neighbourhood (1,2,3)
    :param result:
    :return:
    """
    if result is None:
        result = np.zeros((object_area.shape[0], object_area.shape[1], object_area.shape[2]), dtype=np.float64)
    res, c = _calculate_maximum(object_area, base_object, level, result)
    return res

def _calculate_maximum(np.ndarray[np.float64_t, ndim=3] object_area, np.ndarray[np.uint8_t, ndim=3] base_object,
                       int level, np.ndarray[np.float64_t, ndim=3] result):
    # cdef np.ndarray[np.uint8_t, ndim=3] consumed_area = np.copy(base_object)
    # cdef np.ndarray[np.float64_t, ndim=3] result
    cdef Size x_size, y_size, z_size, array_pos, x, y, z, xx, yy, zz
    cdef Py_ssize_t count = 0
    cdef char neigh_length = neigh_level[level]
    cdef int neigh_it
    cdef my_queue[Point] current_points
    cdef Point p, p1
    cdef np.float64_t object_area_value, result_value
    z_size = object_area.shape[0]
    y_size = object_area.shape[1]
    x_size = object_area.shape[2]


    result[base_object > 0] = object_area[base_object > 0]

    # Find border of component
    for z in range(1, z_size-1):
        for y in range(1, y_size-1):
            for x in range (1, x_size-1):
                if base_object[z,y,x] > 0:
                    for neigh_it in range(neigh_length):
                        zz = z+neighbourhood[neigh_it][0]
                        yy = y+neighbourhood[neigh_it][1]
                        xx = x+neighbourhood[neigh_it][2]
                        if xx == -1 or xx == x_size or xx == -1 or yy == y_size or zz == -1 or zz == z_size:
                            continue
                        if base_object[zz, yy, zz] == 0:
                            p.z = z
                            p.y = y
                            p.x = x
                            current_points.push(p)
                            break
    while not current_points.empty():
        count+=1
        p = current_points.front()
        current_points.pop()
        base_object[p.z, p.y, p.x] = 0
        for neigh_it in range(neigh_length):
            z = p.z + neighbourhood[neigh_it][0]
            y = p.y + neighbourhood[neigh_it][1]
            x = p.x + neighbourhood[neigh_it][2]
            if x < 0 or y < 0 or z < 0 or x >= x_size or y >= y_size or z >= z_size:
                continue
            object_area_value = object_area[z, y, x]
            if object_area_value == 0:
                continue
            if result[z, y, x] == object_area_value:
                continue
            if result[p.z, p.y, p.x] > result[z, y, x]:
                if result[p.z, p.y, p.x] > object_area_value:
                    result[z, y, x] = object_area_value
                else:
                    result[z, y, x] = result[p.z, p.y, p.x]
                if base_object[z, y, x] == 0:
                    p1.z = z
                    p1.y = y
                    p1.x = x
                    current_points.push(p1)
                    base_object[z, y, x] = 1
    return result, count

def calculate_minimum(np.ndarray[np.float64_t, ndim=3] object_area, np.ndarray[np.uint8_t, ndim=3] base_object, int level, float maximum, result=None):
    """
    Calculate maximum path from source
    :param object_area: data with removed other components
    :param base_object: area of component
    :param level: type of neighbourhood (1,2,3)
    :param result:
    :return:
    """
    if result is None:
        result = np.zeros((object_area.shape[0], object_area.shape[1], object_area.shape[2]), dtype=np.float64)
        result[:] = maximum
    res, c = _calculate_minimum(object_area, base_object, level, result)
    return res

def _calculate_minimum(np.ndarray[np.float64_t, ndim=3] object_area, np.ndarray[np.uint8_t, ndim=3] base_object,
                       int level, np.ndarray[np.float64_t, ndim=3] result):
    # cdef np.ndarray[np.uint8_t, ndim=3] consumed_area = np.copy(base_object)
    # cdef np.ndarray[np.float64_t, ndim=3] result
    cdef Size x_size, y_size, z_size, array_pos, x, y, z, xx, yy, zz
    cdef Py_ssize_t count = 0
    cdef char neigh_length = neigh_level[level]
    cdef int neigh_it
    cdef my_queue[Point] current_points
    cdef Point p, p1
    cdef np.float64_t object_area_value, result_value
    z_size = object_area.shape[0]
    y_size = object_area.shape[1]
    x_size = object_area.shape[2]
    print("_calculate_minimum")

    result[base_object > 0] = object_area[base_object > 0]

    # Find border of component
    for z in range(0, z_size):
        for y in range(0, y_size):
            for x in range (0, x_size):
                if base_object[z,y,x] > 0:
                    for neigh_it in range(neigh_length):
                        zz = z+neighbourhood[neigh_it][0]
                        yy = y+neighbourhood[neigh_it][1]
                        xx = x+neighbourhood[neigh_it][2]
                        if xx == -1 or xx == x_size or xx == -1 or yy == y_size or zz == -1 or zz == z_size:
                            continue
                        if base_object[zz, yy, zz] == 0:
                            p.z = z
                            p.y = y
                            p.x = x
                            current_points.push(p)
                            break
    while not current_points.empty():
        count+=1
        p = current_points.front()
        current_points.pop()
        base_object[p.z, p.y, p.x] = 0
        for neigh_it in range(neigh_length):
            z = p.z + neighbourhood[neigh_it][0]
            y = p.y + neighbourhood[neigh_it][1]
            x = p.x + neighbourhood[neigh_it][2]
            if x < 0 or y < 0 or z < 0 or x >= x_size or y >= y_size or z >= z_size:
                continue
            object_area_value = object_area[z, y, x]
            if object_area_value == 0:
                continue
            if result[z, y, x] == object_area_value:
                continue
            if result[p.z, p.y, p.x] < result[z, y, x]:
                if result[p.z, p.y, p.x] < object_area_value:
                    result[z, y, x] = object_area_value
                else:
                    result[z, y, x] = result[p.z, p.y, p.x]
                if base_object[z, y, x] == 0:
                    p1.z = z
                    p1.y = y
                    p1.x = x
                    current_points.push(p1)
                    base_object[z, y, x] = 1
    return result, count