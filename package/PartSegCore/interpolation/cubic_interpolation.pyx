# cython: profile=True, boundscheck=False, wraparound=False, nonecheck=False
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

import numpy as np
cimport numpy as np
DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t
