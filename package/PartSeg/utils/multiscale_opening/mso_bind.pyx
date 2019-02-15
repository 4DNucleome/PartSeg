# distutils: language = c++
# cython: language_level=3

from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp cimport bool
from numpy cimport uint8_t, uint16_t, uint32_t, float32_t, float64_t, int8_t, int16_t
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
ctypedef uint8_t component_type

cpdef enum MuType:
    base_mu = 1
    reflection_mu = 2
    two_object_mu = 3

cpdef enum SegmentType:
    separate_objects = 1
    separate_from_background = 2


cdef extern from 'mso.h' namespace 'MSO':
    cdef cppclass MSO[T]:
        void set_neighbourhood(vector[int8_t] neighbourhood, vector[double] distances) except +
        void erase_data()
        void set_mu_copy(const vector[mu_type] & mu) except +
        void set_mu_copy(mu_type * mu, size_t length) except +
        void set_mu_swap(vector[mu_type] & mu) except +
        void set_neighbourhood(vector[int8_t] neighbourhood, vector[mu_type] distances)
        void set_neighbourhood(int8_t * neighbourhood, mu_type * distances, size_t neigh_size)
        void compute_FDT(vector[mu_type] & array) except +
        void set_use_background(bool val)
        size_t get_length()
        void set_data[W](T * components, W size, T background_component)
        void set_data[W](T * components, W size)

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
    cdef MSO[component_type] mso
    cdef numpy.ndarray components
    cdef component_type components_num

    def set_image(self, numpy.ndarray[image_types, ndim=3] image, image_types lower_bound,
                  image_types upper_bound, MuType type_, mask=None,
                  image_types lower_mid_bound=0, image_types upper_mid_bound=0):
        cdef vector[mu_type] mu
        mu = calculate_mu_vector(image, lower_bound, upper_bound, type_, mask, lower_mid_bound, upper_mid_bound)
        self.mso.set_mu_swap(mu)

    def set_mu_array(self, numpy.ndarray[mu_type, ndim=3] mu):
        self.mso.set_mu_copy(<mu_type *> mu.data, mu.size)

    def set_components(self, numpy.ndarray[component_type, ndim=3] components):
        cdef vector[uint16_t] shape = vector[uint16_t](3, 0)
        shape[0] = components.shape[0]
        shape[1] = components.shape[1]
        shape[2] = components.shape[2]
        self.mso.set_data(<component_type *> components.data, shape)
        self.components = components

    def set_neighbourhood(self, numpy.ndarray[int8_t, ndim=2] neighbourhood, numpy.ndarray[mu_type] distances):
        self.mso.set_neighbourhood(<int8_t *> neighbourhood.data, <mu_type *> distances.data, distances.size)

    def calculate_FDT(self):
        cdef vector[mu_type] fdt = vector[mu_type](self.mso.get_length(), 0)
        self.mso.compute_FDT(fdt)
        return numpy.array(fdt).reshape(self.components.shape[0], self.components.shape[1], self.components.shape[2])










