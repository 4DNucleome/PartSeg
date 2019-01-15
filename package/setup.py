import os
import setuptools
# from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Build import cythonize

current_dir = os.path.dirname(os.path.abspath(__file__))

extensions = [
    Extension('PartSeg.partseg_utils.distance_in_structure.euclidean_cython',
              sources=["PartSeg/partseg_utils/distance_in_structure/euclidean_cython.pyx"],
              include_dirs=[np.get_include()] + [os.path.join(current_dir, "partseg_utils", "distance_in_structure")],
              language='c++', extra_compile_args=["-std=c++11"], extra_link_args=["-std=c++11"]),
    Extension('PartSeg.partseg_utils.distance_in_structure.path_sprawl_cython',
              sources=["PartSeg/partseg_utils/distance_in_structure/path_sprawl_cython.pyx"],
              include_dirs=[np.get_include()] + [os.path.join(current_dir, "partseg_utils", "distance_in_structure")],
              language='c++', extra_compile_args=["-std=c++11"], extra_link_args=["-std=c++11"]),
    Extension('PartSeg.partseg_utils.distance_in_structure.sprawl_utils',
              sources=["PartSeg/partseg_utils/distance_in_structure/sprawl_utils.pyx"],
              include_dirs=[np.get_include()] + [os.path.join(current_dir, "partseg_utils", "distance_in_structure")],
              language='c++', extra_compile_args=["-std=c++11"], extra_link_args=["-std=c++11"]),
    Extension('PartSeg.partseg_utils.distance_in_structure.fuzzy_distance',
              sources=["PartSeg/partseg_utils/distance_in_structure/fuzzy_distance.pyx"],
              include_dirs=[np.get_include()] + [os.path.join(current_dir, "partseg_utils", "distance_in_structure")],
              language='c++', extra_compile_args=["-std=c++11"], extra_link_args=["-std=c++11"]),
    Extension("PartSeg.partseg_utils.color_image.color_image", ["PartSeg/partseg_utils/color_image/color_image.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args=['-std=c++11'],
              language='c++',
              )
]

setuptools.setup(
    ext_modules=cythonize(extensions),
    name="PartSeg",
    version="0.8.2",
    author="Grzegorz Bokota",
    author_email="g.bokota@cent.uw.edu.pl",
    description="PartSeg is python GUI for bio imaging analysis",
    url="https://4dnucleome.cent.uw.edu.pl/PartSeg/",
    packages=setuptools.find_packages(),
    include_package_data=True,
    scripts=["scripts/PartSeg"],
    install_requires=['numpy', 'tifffile>=0.15,<1', 'appdirs', 'SimpleITK', 'PyQt5', 'scipy', 'QtPy', 'sentry_sdk',
                      'deprecation', 'qtawesome', 'six', 'h5py', 'pandas', 'sympy', 'Cython'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha"
    ],
)
