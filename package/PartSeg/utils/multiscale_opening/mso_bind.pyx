# distutils: language = c++
# cython: language_level=3

from libcpp.vector cimport vector
from numpy cimport uint8_t, uint16_t, uint32_t, float32_t, float64_t, int8_t, bool, int16_t
import numpy
cimport numpy 

ctypedef fused image_types:
    int8_t
    int16_t
    uint8_t
    uint16_t
    uint32_t
    float32_t
    float64_t

ctypedef double mu_type

cpdef enum MuType:
    base_mu = 1
    reflection_mu = 2
    two_object_mu = 3

cdef extern from 'mso.h' namespace 'MSO':
    cdef cppclass MSO[T]:
        void set_neighbourhood(vector[int8_t] neighbourhood, vector[double] distances) except +
        void erase_data()
    
    cdef vector[mu_type] calculate_mu_array[T](T *array, size_t length, T lower_bound, T upper_bound)
    cdef vector[mu_type] calculate_mu_array_masked[T](T *array, size_t length, T lower_bound, T upper_bound, uint8_t *mask)
    cdef vector[mu_type] calculate_reflection_mu_array[T](T *array, size_t length, T lower_bound, T upper_bound)
    cdef vector[mu_type] calculate_reflection_mu_array_masked[T](T *array, size_t length, T lower_bound, T upper_bound, uint8_t *mask)
    cdef vector[mu_type] calculate_two_object_mu[T](T *array, size_t length, T lower_bound, T upper_bound,
                                                    T lower_mid_bound, T upper_mid_bound)
    cdef vector[mu_type] calculate_two_object_mu_masked[T](T *array, size_t length, T lower_bound, T upper_bound,
                                                           T lower_mid_bound, T upper_mid_bound, uint8_t *mask)
    
cdef vector[mu_type] calculate_mu_vector(numpy.ndarray[image_types, ndim=3] image, image_types lower_bound,
                                         image_types upper_bound, MuType type_, mask=None,
                                         image_types lower_mid_bound=0, image_types upper_mid_bound=0):
    cdef vector[mu_type] result;
    cdef numpy.ndarray[uint8_t, ndim=3] mask_array
    cdef size_t length, x
    length = image.size

    if lower_bound > upper_bound:
        reflected = True
        tmp = lower_bound
        lower_bound = upper_bound
        upper_bound = tmp
    else:
        reflected = False
    if mask is not None:
        mask_array = mask.astype(numpy.uint8)
    if type_ == MuType.base_mu:
        if mask is None:
            result = calculate_mu_array(<image_types *> image.data, length, lower_bound, upper_bound)
        else:
            result = calculate_mu_array_masked(<image_types *> image.data, length, lower_bound, upper_bound,
                                               <uint8_t *> mask_array.data)
    elif type_ == MuType.reflection_mu:
        if mask is None:
            result = calculate_reflection_mu_array(<image_types *> image.data, length, lower_bound, upper_bound)
        else:
            result = calculate_reflection_mu_array_masked(<image_types *> image.data, length, lower_bound, upper_bound,
                                                          <uint8_t *> mask_array.data)
        reflected = False
    elif type_ == MuType.two_object_mu:
        if mask is None:
            result = calculate_two_object_mu(<image_types *> image.data, length, lower_bound, upper_bound,
                                             lower_mid_bound, upper_mid_bound)
        else:
            result = calculate_two_object_mu_masked(<image_types *> image.data, length, lower_bound, upper_bound,
                                                    lower_mid_bound, upper_mid_bound,
                                                    <uint8_t *> mask_array.data)
    if reflected:
        if mask is None:
            for x in range(length):
                result[x] = 1 - result[x]
        else:
            for x in range(length):
                if mask_array.data[x]:
                    result[x] = 1 - result[x]
    return result

def calculate_mu(numpy.ndarray[image_types, ndim=3] image, image_types lower_bound,
                 image_types upper_bound, MuType type_, mask=None, image_types lower_mid_bound=0,
                 image_types upper_mid_bound=0):
    cdef vector[mu_type] tmp;
    tmp = calculate_mu_vector(image, lower_bound, upper_bound, type_, mask, lower_bound, upper_mid_bound)
    result = numpy.array(tmp)
    return result.reshape((image.shape[0], image.shape[1], image.shape[2]))


cdef class PyMSO:
    cdef MSO[uint8_t] mso

