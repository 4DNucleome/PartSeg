from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

extensions = [
    Extension("project_utils.color_image.color_image", ["project_utils/color_image/color_image.pyx"],
        include_dirs = [np.get_include()])
    ]

setup(
    ext_modules = cythonize(extensions),
    name="coloring image"
)