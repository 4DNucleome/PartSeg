# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# cython: language_level=3

from numpy cimport float64_t, int8_t, uint8_t

from .distance_utils cimport Point
include "put_borders_in_queue.pyx"


def calculate_maximum(np.ndarray[float64_t, ndim=3] object_area, np.ndarray[uint8_t, ndim=3] base_object,
                      np.ndarray[int8_t, ndim=2] neighbourhood, result=None):
    """
    Calculate maximum path from source
    :param object_area: data with removed other components
    :param base_object: area of component
    :param neighbourhood: array with relative coordinates of neighbours
    :param result:
    :return:
    """
    if result is None:
        result = np.full((object_area.shape[0], object_area.shape[1], object_area.shape[2]), -np.inf, dtype=np.float64)
    res, c = _calculate_maximum(object_area, base_object, neighbourhood, result)
    return res


cdef _calculate_maximum(np.ndarray[float64_t, ndim=3] object_area, np.ndarray[uint8_t, ndim=3] base_object,
                       np.ndarray[int8_t, ndim=2] neighbourhood, np.ndarray[float64_t, ndim=3] result):
    # cdef np.ndarray[np.uint8_t, ndim=3] consumed_area = np.copy(base_object)
    # cdef np.ndarray[np.float64_t, ndim=3] result
    cdef Size x_size, y_size, z_size, array_pos, x, y, z, xx, yy, zz
    cdef Py_ssize_t count = 0
    cdef char neigh_length = neighbourhood.shape[0]
    cdef int neigh_it
    cdef my_queue[Point] current_points
    cdef Point p, p1
    cdef np.float64_t object_area_value, result_value
    z_size = object_area.shape[0]
    y_size = object_area.shape[1]
    x_size = object_area.shape[2]


    result[base_object > 0] = object_area[base_object > 0]

    # Find border of component
    put_borders_in_queue(current_points, base_object, neighbourhood)
    while not current_points.empty():
        count+=1
        p = current_points.front()
        current_points.pop()
        base_object[p.z, p.y, p.x] = 0
        for neigh_it in range(neigh_length):
            z = p.z + neighbourhood[neigh_it, 0]
            y = p.y + neighbourhood[neigh_it, 1]
            x = p.x + neighbourhood[neigh_it, 2]
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

def calculate_minimum(np.ndarray[float64_t, ndim=3] object_area, np.ndarray[uint8_t, ndim=3] base_object,
                      np.ndarray[int8_t, ndim=2] neighbourhood, result=None):
    """
    Calculate maximum path from source
    :param maximum: maximum possible value on path
    :param object_area: data with removed other components
    :param base_object: area of component
    :param neighbourhood: array with relative coordinates of neighbours
    :param result:
    :return:
    """
    if result is None:
        result = np.full((object_area.shape[0], object_area.shape[1], object_area.shape[2]), np.inf, dtype=np.float64)
    res, c = _calculate_minimum(object_area, base_object, neighbourhood, result)
    return res

cdef _calculate_minimum(np.ndarray[float64_t, ndim=3] object_area, np.ndarray[uint8_t, ndim=3] base_object,
                       np.ndarray[int8_t, ndim=2] neighbourhood, np.ndarray[float64_t, ndim=3] result):
    # cdef np.ndarray[np.uint8_t, ndim=3] consumed_area = np.copy(base_object)
    # cdef np.ndarray[np.float64_t, ndim=3] result
    cdef Size x_size, y_size, z_size, array_pos, x, y, z, xx, yy, zz
    cdef Py_ssize_t count = 0
    cdef char neigh_length = neighbourhood.shape[0]
    cdef int neigh_it
    cdef my_queue[Point] current_points
    cdef Point p, p1
    cdef np.float64_t object_area_value, result_value
    z_size = object_area.shape[0]
    y_size = object_area.shape[1]
    x_size = object_area.shape[2]

    result[base_object > 0] = object_area[base_object > 0]
    put_borders_in_queue(current_points, base_object, neighbourhood)
    while not current_points.empty():
        count+=1
        p = current_points.front()
        current_points.pop()
        base_object[p.z, p.y, p.x] = 0
        for neigh_it in range(neigh_length):
            z = p.z + neighbourhood[neigh_it, 0]
            y = p.y + neighbourhood[neigh_it, 1]
            x = p.x + neighbourhood[neigh_it, 2]
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