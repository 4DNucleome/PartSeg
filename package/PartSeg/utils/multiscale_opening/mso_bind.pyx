# distutils: language = c++
from libcpp.vector cimport vector
from numpy cimport uint8_t, int8_t


cdef extern from 'mso.h' namespace 'MSO':
    cdef cppclass MSO[T]:
        void set_neighbourhood(vector[int8_t] neighbourhood, vector[double] distances) except +
        void erase_data()


cdef class PyMSO:
    cdef MSO[uint8_t] mso

    def
