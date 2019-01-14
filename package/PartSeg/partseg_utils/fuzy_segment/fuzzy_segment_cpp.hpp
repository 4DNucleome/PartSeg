
struct Borders{
    double tap;
    double ta;
    double tb;
    double tbp;
};

inline double shrink(double res){
    if (res < 0)
        res = 0;
    else if (res > 1)
        res = 1;
    return res;
};

double mu_flag0(double x, Borders b){
    double res;
    res = 1 - (x - b.ta) / (b.tb - b.ta);
    return shrink(res);
};

double mu_flag0(double x, Borders b){
    double res;
    res = (x - b.ta) / (b.tb - b.ta);
    if (res < 0.5)
        res = 1 - res;
    return shrink(res)
 };

 double mu_flag5(double x, Borders b){
    if ((b.ta - b.tap > 0) && (x >= b.tap) && (x <= b.ta))
        return shrink((x - b.tap) / (b.ta - b.tap));
    else if ((b.tb - b.ta > 0) && (x > b.ta)  && (x <= b.tb))
        return shrink((b.tb - x) / (b.tb - b.ta));
    else
        return shrink((x - b.ta) / (b.tb - b.ta));
 };

template<T, F>
void one_step_FDT(numpy_arrays_3 image, np.ndarray[np.uint16_t, ndim=3] lFDT,
                              Borders b, MU fun, Py_ssize_t x, Py_ssize_t y, Py_ssize_t z, Py_ssize_t *F_count,
                              Py_ssize_t n_begin, Py_ssize_t n_end):
    if lFDT[z, y, x] == 0:
        return
    mu_p = fun(image[z, x, y], b)
    min_val = lFDT[z, y, x]
    cdef double min_val, pixel_val
    for i in range(n_begin, n_end):
        pixel_val = lFDT[z + V[i, 2], y + V[i, 1], x + V[i, 0]]
        if pixel_val == 0:
            mu_q = 0
        else:
            mu_q = fun(pixel_val, b)
        temp = pixel_val + 50 * (mu_p + mu_q) * D[i]
        temp = ceil(temp - 0.5)
        if temp < min_val:
            lFDT[z, y, x] = temp
            min_val = temp
            F_count += 1