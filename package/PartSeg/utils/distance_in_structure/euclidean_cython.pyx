# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# cython: language_level=3


from numpy cimport float64_t, int8_t, uint8_t
from cpython.mem cimport PyMem_Free
from .distance_utils cimport Point, component_types
include "put_borders_in_queue.pyx"


def calculate_euclidean(np.ndarray[uint8_t, ndim=3] object_area, np.ndarray[uint8_t, ndim=3] base_object,
                        np.ndarray[int8_t, ndim=2] neighbourhood, np.ndarray[float64_t, ndim=1] distance):
    cdef np.ndarray[uint8_t, ndim=3] consumed_area = np.copy(base_object)
    cdef np.ndarray[float64_t, ndim=3] result
    cdef Size x_size, y_size, z_size, array_pos, x, y, z, xx, yy, zz
    cdef Py_ssize_t count = 0
    cdef char neigh_length = neighbourhood.shape[0]
    cdef int neigh_it
    cdef my_queue[Point] current_points
    cdef Point p, p1
    z_size = object_area.shape[0]
    y_size = object_area.shape[1]
    x_size = object_area.shape[2]
    result = np.zeros((z_size, y_size, x_size), dtype=np.float64)
    result[base_object == 0] = np.inf
    put_borders_in_queue(current_points, base_object, neighbourhood)
    while not current_points.empty():
        p = current_points.front()
        current_points.pop()
        count += 1
        """if consumed_area[p.z, p.y, p.x] > 0:
            continue"""
        consumed_area[p.z, p.y, p.x] = 0
        for neigh_it in range(neigh_length):
            z = p.z + neighbourhood[neigh_it, 0]
            y = p.y + neighbourhood[neigh_it, 1]
            x = p.x + neighbourhood[neigh_it, 2]
            if x < 0 or y < 0 or z < 0 or x >= x_size or y >= y_size or z >= z_size:
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
    # print("total_steps: " +str(count))
    return result


def calculate_euclidean_iterative(
        np.ndarray[uint8_t, ndim=3] object_area, np.ndarray[component_types, ndim=3] base_object,
        np.ndarray[int8_t, ndim=2] neighbourhood, np.ndarray[float64_t, ndim=1] distance):
    cdef np.ndarray[component_types, ndim=3] consumed_area = np.copy(base_object)
    cdef np.ndarray[component_types, ndim=3] result = np.copy(base_object)
    cdef np.ndarray[float64_t, ndim=3] distance_cache
    cdef Size x_size, y_size, z_size, array_pos, x, y, z, xx, yy, zz
    cdef component_types i, components_num
    cdef Py_ssize_t count = 0
    cdef char neigh_length = neighbourhood.shape[0]
    cdef int neigh_it
    cdef my_queue[Point] * current_points_array
    cdef Point p, p1
    z_size = object_area.shape[0]
    y_size = object_area.shape[1]
    x_size = object_area.shape[2]
    distance_cache = np.zeros((z_size, y_size, x_size), dtype=np.float64)
    distance_cache[base_object == 0] = np.inf
    components_num = base_object.max()
    current_points_array = create_borders_queues(base_object, neighbourhood, components_num)
    for i in range(components_num):
        while not current_points_array[i].empty():
            p = current_points_array[i].front()
            current_points_array[i].pop()
            count += 1
            """if consumed_area[p.z, p.y, p.x] > 0:
                continue"""
            consumed_area[p.z, p.y, p.x] = 0
            for neigh_it in range(neigh_length):
                z = p.z + neighbourhood[neigh_it, 0]
                y = p.y + neighbourhood[neigh_it, 1]
                x = p.x + neighbourhood[neigh_it, 2]
                if x < 0 or y < 0 or z < 0 or x >= x_size or y >= y_size or z >= z_size:
                    continue
                if object_area[z, y, x] == 0:
                    continue
                if distance_cache[z, y, x] > distance_cache[p.z, p.y, p.x] + distance[neigh_it]:
                    distance_cache[z, y, x] = distance_cache[p.z, p.y, p.x] + distance[neigh_it]
                    if consumed_area[z, y, x] == 0:
                        consumed_area[z, y, x] = 1
                        result[z, y, x] = i
                        p1.z = z
                        p1.y = y
                        p1.x = x
                        current_points_array[i].push(p1)
    PyMem_Free(current_points_array)
    # print("total_steps: " +str(count))
    return result


def show_border(np.ndarray[component_types, ndim=3] base_object, np.ndarray[int8_t, ndim=2] neighbourhood):
    cdef my_queue[Point] current_points
    cdef Point p, p1
    cdef np.ndarray[uint8_t, ndim=3] result
    z_size = base_object.shape[0]
    y_size = base_object.shape[1]
    x_size = base_object.shape[2]
    result = np.zeros((z_size, y_size, x_size), dtype=np.uint8)
    put_borders_in_queue(current_points, base_object, neighbourhood)
    while not current_points.empty():
        p = current_points.front()
        current_points.pop()
        result[p.z, p.y, p.x] = 1
    return result

