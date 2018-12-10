from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

extensions = [
    Extension("fuzzy_segment_cython", ["fuzzy_segment_cython.pyx"],
        include_dirs = [np.get_include()])
    ]

setup(
    ext_modules = cythonize(extensions),
    name="fuzzy_segment_cython"
)