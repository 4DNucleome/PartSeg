# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# cython: language_level=3

from  __future__ import print_function

from numpy cimport float64_t, int8_t, uint8_t

from .distance_utils cimport Point

from libc.math cimport abs

include "put_borders_in_queue.pyx"

cdef enum SegmentType:
    reversed_mu = 1
    reflection_mu = 2
    two_object_mu = 3

cdef inline double calculate_two_object_mu(float64_t pixel_val, float64_t lower_bound, float64_t upper_bound,
                                           float64_t lower_mid_bound, float64_t upper_mid_bound):
    cdef double mu = (pixel_val - lower_bound) / (upper_bound - lower_bound);
    if (lower_bound - lower_mid_bound) > 0 and (pixel_val >= lower_mid_bound) and pixel_val <= lower_bound:
        mu = (pixel_val - lower_mid_bound) / (lower_bound - lower_mid_bound)
    elif (upper_bound - lower_bound) > 0 and lower_bound < pixel_val <= upper_bound:
        mu = (upper_bound - pixel_val) / (upper_bound - lower_bound)
    if mu > 1:
        mu = 1
    if mu < 0:
        mu = 0
    return mu

cdef inline double calculate_mu(float64_t pixel_val, float64_t lower_bound, float64_t upper_bound,
                                const SegmentType flag):
    cdef double mu = (pixel_val - lower_bound) / (upper_bound - lower_bound);
    if flag == SegmentType.reversed_mu:
        mu = 1 - mu
    if flag == SegmentType.reflection_mu and mu < 0.5:
        mu = 1 - mu
    if mu > 1:
        mu = 1
    if mu < 0:
        mu = 0
    return mu



def fuzzy_distance(np.ndarray[float64_t, ndim=3] object_area, np.ndarray[uint8_t, ndim=3] base_object,
                   np.ndarray[int8_t, ndim=2] neighbourhood, np.ndarray[float64_t, ndim=1] distance,
                   float64_t lower_bound, float64_t upper_bound):
    cdef np.ndarray[uint8_t, ndim=3] consumed_area = np.copy(base_object)
    cdef np.ndarray[float64_t, ndim=3] result
    cdef Size x_size, y_size, z_size, array_pos, x, y, z, xx, yy, zz
    cdef Py_ssize_t count = 0, count2 = 0
    cdef char neigh_length = neighbourhood.shape[0]
    cdef int neigh_it
    cdef my_queue[Point] current_points, new_points
    cdef Point p, p1
    cdef double point_mu, neigh_mu, mu_diff
    cdef float64_t object_value, object_neighbourhood_value
    z_size = object_area.shape[0]
    y_size = object_area.shape[1]
    x_size = object_area.shape[2]
    result = np.zeros((z_size, y_size, x_size), dtype=np.float64)
    result[base_object == 0] = np.inf
    put_borders_in_queue(object_area, current_points, base_object, neighbourhood)
    while not current_points.empty():
        p = current_points.front()
        current_points.pop()
        count += 1
        """if consumed_area[p.z, p.y, p.x] > 0:
            continue"""
        consumed_area[p.z, p.y, p.x] = 0
        object_value = object_area[p.z, p.y, p.x]
        point_mu = calculate_mu(object_value, lower_bound, upper_bound, SegmentType.reflection_mu)
        for neigh_it in range(neigh_length):
            z = p.z + neighbourhood[neigh_it, 0]
            y = p.y + neighbourhood[neigh_it, 1]
            x = p.x + neighbourhood[neigh_it, 2]
            if x < 0 or y < 0 or z < 0 or x >= x_size or y >= y_size or z >= z_size:
                continue
            if object_area[z, y, x] == 0:
                continue
            object_neighbourhood_value = object_area[z, y, x]
            neigh_mu = calculate_mu(object_neighbourhood_value, lower_bound, upper_bound, SegmentType.reflection_mu)
            mu_diff = (point_mu + neigh_mu) * distance[neigh_it] #  abs(object_value - object_neighbourhood_value)
            if result[z, y, x] > result[p.z, p.y, p.x] + mu_diff:
                result[z, y, x] = result[p.z, p.y, p.x] + mu_diff
                if consumed_area[z, y, x] == 0:
                    consumed_area[z, y, x] = 1
                    p1.z = z
                    p1.y = y
                    p1.x = x
                    current_points.push(p1)
    # print("total_steps: " +str(count), " and ", str(count2))
    return result
