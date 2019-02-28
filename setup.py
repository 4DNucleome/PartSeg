import codecs
import os
import re

import setuptools
# from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Build import cythonize

current_dir = os.path.dirname(os.path.abspath(__file__))

try:
    import tifffile
    import imagecodecs
    import imagecodecs._imagecodecs
    tifffile_string = "tifffile>=0.15"
except ImportError:
    tifffile_string = 'tifffile>=0.15,<1'

extensions = [
    Extension('PartSeg.utils.distance_in_structure.euclidean_cython',
              sources=["package/PartSeg/utils/distance_in_structure/euclidean_cython.pyx"],
              include_dirs=[np.get_include()] + [os.path.join(current_dir, "utils", "distance_in_structure")],
              language='c++', extra_compile_args=["-std=c++11"], extra_link_args=["-std=c++11"]),
    Extension('PartSeg.utils.distance_in_structure.path_sprawl_cython',
              sources=["package/PartSeg/utils/distance_in_structure/path_sprawl_cython.pyx"],
              include_dirs=[np.get_include()] + [os.path.join(current_dir, "utils", "distance_in_structure")],
              language='c++', extra_compile_args=["-std=c++11"], extra_link_args=["-std=c++11"]),
    Extension('PartSeg.utils.distance_in_structure.sprawl_utils',
              sources=["package/PartSeg/utils/distance_in_structure/sprawl_utils.pyx"],
              include_dirs=[np.get_include()] + [os.path.join(current_dir, "utils", "distance_in_structure")],
              language='c++', extra_compile_args=["-std=c++11"], extra_link_args=["-std=c++11"]),
    Extension('PartSeg.utils.distance_in_structure.fuzzy_distance',
              sources=["package/PartSeg/utils/distance_in_structure/fuzzy_distance.pyx"],
              include_dirs=[np.get_include()] + [os.path.join(current_dir, "utils", "distance_in_structure")],
              language='c++', extra_compile_args=["-std=c++11"], extra_link_args=["-std=c++11"]),
    Extension("PartSeg.utils.color_image.color_image", ["package/PartSeg/utils/color_image/color_image.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args=['-std=c++11'],
              language='c++',
              ),
    Extension("PartSeg.utils.multiscale_opening.mso_bind", ["package/PartSeg/utils/multiscale_opening/mso_bind.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args=['-std=c++11', '-Wall'],
              language='c++',
              # undef_macros=["NDEBUG"],
              # define_macros=[("DEBUG", None)]
              )
]


def read(*parts):
    with codecs.open(os.path.join(current_dir, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

def readme():
    this_directory = os.path.abspath(os.path.dirname(__file__))
    reg = re.compile(r'(!\[[^]]*\])\((images/[^)]*)\)')
    with open(os.path.join(this_directory, 'Readme.md')) as f:
        text = f.read()
        text = reg.sub(r'\1(https://raw.githubusercontent.com/4DNucleome/PartSeg/master/\2)', text)
        return text

setuptools.setup(
    ext_modules=cythonize(extensions),
    name="PartSeg",
    version=find_version("package", "PartSeg", "__init__.py"),
    author="Grzegorz Bokota",
    author_email="g.bokota@cent.uw.edu.pl",
    description="PartSeg is python GUI for bio imaging analysis",
    url="https://4dnucleome.cent.uw.edu.pl/PartSeg/",
    packages=setuptools.find_packages('./package'),
    package_dir = {'': 'package'},
    include_package_data=True,
    long_description=readme(),
    long_description_content_type='text/markdown',
    scripts=[os.path.join("package","scripts", "PartSeg")],
    install_requires=['numpy', tifffile_string, 'appdirs', 'SimpleITK', 'PyQt5', 'scipy', 'QtPy', 'sentry_sdk',
                      'deprecation', 'qtawesome', 'six', 'h5py', 'pandas', 'sympy', 'Cython', 'openpyxl', 'xlrd'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha"
    ],
)
