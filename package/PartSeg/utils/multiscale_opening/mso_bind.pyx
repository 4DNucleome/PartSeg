# distutils: language = c++
# cython: language_level=3

from libcpp.vector cimport vector
from libcpp cimport bool
from numpy cimport uint8_t, uint16_t, uint32_t, float32_t, float64_t, int8_t, int16_t
import numpy
cimport numpy 
import cython


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
np_component_type = numpy.uint8


cpdef enum MuType:
    base_mu = 1
    reflection_mu = 2
    two_object_mu = 3

cpdef enum SegmentType:
    separate_objects = 1
    separate_from_background = 2


cdef extern from 'mso.h' namespace 'MSO':
    cdef cppclass MSO[T, M]:
        void set_neighbourhood(vector[int8_t] neighbourhood, vector[double] distances) except +
        void erase_data()
        void set_mu_copy(const vector[M] & mu) except +
        void set_mu_copy(M * mu, size_t length) except +
        void set_mu_swap(vector[M] & mu) except +
        void set_neighbourhood(vector[int8_t] neighbourhood, vector[mu_type] distances) except +
        void set_neighbourhood(int8_t * neighbourhood, M * distances, size_t neigh_size) except +
        void set_components_num(T components_num)
        void compute_FDT(vector[M] & array) except +
        void set_use_background(bool val)
        size_t optimum_erosion_calculate(vector[M] &fdt_array, vector[T] &components_arr, vector[bool] & sprawl_area) except +
        size_t constrained_dilation(vector[M] &fdt_array, vector[T] &components_arr, vector[bool] & sprawl_area) except +
        size_t get_length()
        size_t run_MSO() nogil except +
        size_t run_MSO(size_t steps_limits) nogil except +
        void set_data[W](T * components, W size, T background_component)
        void set_data[W](T * components, W size)
        size_t steps_done()
        vector[mu_type] get_fdt()
        vector[T] get_result_catted()

    cdef cppclass MuCalc[R, T]:
        vector[R] calculate_mu_array(T *array, size_t length, T lower_bound, T upper_bound)
        vector[R] calculate_mu_array_masked(T *array, size_t length, T lower_bound, T upper_bound, uint8_t *mask)
        vector[R] calculate_reflection_mu_array(T *array, size_t length, T lower_bound, T upper_bound)
        vector[R] calculate_reflection_mu_array_masked(T *array, size_t length, T lower_bound, T upper_bound, uint8_t *mask)
        vector[R] calculate_two_object_mu(T *array, size_t length, T lower_bound, T upper_bound,
                                                  T lower_mid_bound, T upper_mid_bound)
        vector[R] calculate_two_object_mu_masked(T *array, size_t length, T lower_bound, T upper_bound,
                                                     T lower_mid_bound, T upper_mid_bound, uint8_t *mask)
    
cdef vector[mu_type] calculate_mu_vector(numpy.ndarray[image_types, ndim=3] image, image_types lower_bound,
                                         image_types upper_bound, MuType type_, mask=None,
                                         image_types lower_mid_bound=0, image_types upper_mid_bound=0):
    cdef vector[mu_type] result;
    cdef MuCalc[mu_type, image_types] calc
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
            result = calc.calculate_mu_array(<image_types *> image.data, length, lower_bound, upper_bound)
        else:
            result = calc.calculate_mu_array_masked(<image_types *> image.data, length, lower_bound, upper_bound,
                                               <uint8_t *> mask_array.data)
    elif type_ == MuType.reflection_mu:
        if mask is None:
            result = calc.calculate_reflection_mu_array(<image_types *> image.data, length, lower_bound, upper_bound)
        else:
            result = calc.calculate_reflection_mu_array_masked(<image_types *> image.data, length, lower_bound, upper_bound,
                                                          <uint8_t *> mask_array.data)
        reflected = False
    elif type_ == MuType.two_object_mu:
        if mask is None:
            result = calc.calculate_two_object_mu(<image_types *> image.data, length, lower_bound, upper_bound,
                                             lower_mid_bound, upper_mid_bound)
        else:
            result = calc.calculate_two_object_mu_masked(<image_types *> image.data, length, lower_bound, upper_bound,
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
    cdef MSO[component_type, mu_type] mso
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

    def set_components(self, numpy.ndarray[component_type, ndim=3] components, component_num=None):
        cdef vector[uint16_t] shape = vector[uint16_t](3, 0)
        shape[0] = components.shape[0]
        shape[1] = components.shape[1]
        shape[2] = components.shape[2]
        self.mso.set_data(<component_type *> components.data, shape)
        if component_num is not None:
            self.mso.set_components_num(component_num+2)
        else:
            self.mso.set_components_num(components.max()+2)
        self.components = components

    def set_neighbourhood(self, numpy.ndarray[int8_t, ndim=2] neighbourhood, numpy.ndarray[mu_type] distances):
        self.mso.set_neighbourhood(<int8_t *> neighbourhood.data, <mu_type *> distances.data, distances.size)

    def calculate_FDT(self):
        cdef vector[mu_type] fdt = vector[mu_type](self.mso.get_length(), 0)
        self.mso.compute_FDT(fdt)
        return numpy.array(fdt).reshape(self.components.shape[0], self.components.shape[1], self.components.shape[2])

    def optimum_erosion_calculate(self, numpy.ndarray[mu_type, ndim=3] fdt_array,
                           numpy.ndarray[component_type, ndim=3] components_arr,
                           numpy.ndarray[uint8_t, ndim=3] sprawl_area):
        """ For testing purpose """
        cdef vector[mu_type] fdt
        fdt.assign(<mu_type *> fdt_array.data, (<mu_type *> fdt_array.data) + (<size_t> fdt_array.size))
        cdef vector[component_type] components
        components.assign(<component_type *> components_arr.data,
                              (<component_type *> components_arr.data) + (<size_t> components_arr.size))
        cdef vector[bool] sprawl_vec
        sprawl_vec.resize(sprawl_area.size)
        cdef numpy.ndarray[uint8_t, ndim=1] tmp = sprawl_area.flatten()
        cdef Py_ssize_t x
        for x in range(tmp.size):
            sprawl_vec[x] = <bool> tmp[x]

        self.mso.set_components_num(components_arr.max())

        count = self.mso.optimum_erosion_calculate(fdt, components, sprawl_vec)
        res = numpy.array(components, dtype=np_component_type)
        res = res.reshape([components_arr.shape[i] for i in range(components_arr.ndim)])
        return res

    def constrained_dilation(self, numpy.ndarray[mu_type, ndim=3] fdt_array,
                           numpy.ndarray[component_type, ndim=3] components_arr,
                           numpy.ndarray[uint8_t, ndim=3] sprawl_area):
        """ For testing purpose """
        cdef vector[mu_type] fdt
        fdt.assign(<mu_type *> fdt_array.data, (<mu_type *> fdt_array.data) + (<size_t> fdt_array.size))
        cdef vector[component_type] components
        components.assign(<component_type *> components_arr.data,
                              (<component_type *> components_arr.data) + (<size_t> components_arr.size))
        cdef vector[bool] sprawl_vec
        sprawl_vec.resize(sprawl_area.size)
        cdef numpy.ndarray[uint8_t, ndim=1] tmp = sprawl_area.flatten()
        cdef Py_ssize_t x
        for x in range(tmp.size):
            sprawl_vec[x] = <bool> tmp[x]

        self.mso.set_components_num(components_arr.max())

        count = self.mso.constrained_dilation(fdt, components, sprawl_vec)
        res = numpy.array(components, dtype=np_component_type)
        res = res.reshape([components_arr.shape[i] for i in range(components_arr.ndim)])
        return res

    def run_MSO(self, size_t step_limits=1):
        cdef size_t val
        with nogil:
            val = self.mso.run_MSO(step_limits)
        return val

    def steps_done(self):
        return self.mso.steps_done()

    def set_components_num(self, num):
        self.mso.set_components_num(num+2)

    def get_result_catted(self):
        cdef vector[component_type] res = self.mso.get_result_catted()
        res_arr = numpy.array(res, dtype=np_component_type)
        res_arr = res_arr.reshape([self.components.shape[i] for i in range(self.components.ndim)])
        return res_arr

    def get_fdt(self):
        cdef vector[mu_type] res = self.mso.get_fdt()
        res_arr = numpy.array(res, dtype=numpy.float64)
        res_arr = res_arr.reshape([self.components.shape[i] for i in range(self.components.ndim)])
        return res_arr

    def set_use_background(self, use):
        self.mso.set_use_background(use)


@cython.boundscheck(False)
def calculate_mu_mid(numpy.ndarray[image_types, ndim=3] image, image_types lower_bound, image_types mid_point,
                 image_types upper_bound):
    cdef int16_t x_size, y_size, z_size, x, y, z
    cdef mu_type res_val
    cdef image_types point_val
    x_size = image.shape[2]
    y_size = image.shape[1]
    z_size = image.shape[0]
    cdef numpy.ndarray[mu_type, ndim=3] res = numpy.empty((z_size, y_size, x_size), dtype=numpy.float64)
    if upper_bound < lower_bound:
        tmp = lower_bound
        lower_bound = upper_bound
        upper_bound = tmp

    for z in range(z_size):
        for y in range(y_size):
            for x in range(x_size):
                point_val = image[z, y, x]
                if point_val < mid_point:
                    res_val = (mid_point - point_val)/(mid_point - lower_bound)
                else:
                    res_val = (point_val - mid_point)/(upper_bound - mid_point)
                if res_val > 1:
                    res_val = 1
                elif res_val < 0:
                    res_val = 0
                res[z, y, x] = res_val
    return res











