from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Build import cythonize

extensions = [Extension("bilinear_interpolation", ["bilinear_interpolation.pyx"], include_dirs=[np.get_include()])]

setup(ext_modules=cythonize(extensions), name="coloring image")
