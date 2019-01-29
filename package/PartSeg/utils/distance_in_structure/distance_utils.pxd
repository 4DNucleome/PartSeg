# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# cython: language_level=3

cimport numpy as cnp

from libcpp cimport bool


ctypedef cnp.int16_t Size

cdef struct Point:
    Size x
    Size y
    Size z

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

ctypedef fused component_types:
    cnp.uint16_t
    cnp.uint8_t

