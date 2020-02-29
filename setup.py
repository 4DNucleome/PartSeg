import codecs
import os
import re

import setuptools

# from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Build import cythonize

current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.join(current_dir, "package")
print(current_dir)
try:
    import imagecodecs
    import imagecodecs._imagecodecs

    imagecodecs_string = imagecodecs.__name__
except ImportError:
    imagecodecs_string = "imagecodecs-lite>=2019.4.20"

extensions = [
    Extension(
        "PartSegCore.sprawl_utils.euclidean_cython",
        sources=["package/PartSegCore/sprawl_utils/euclidean_cython.pyx"],
        include_dirs=[np.get_include()] + [os.path.join(package_dir, "PartSegCore", "sprawl_utils")],
        language="c++",
        extra_compile_args=["-std=c++11"],
        extra_link_args=["-std=c++11"],
    ),
    Extension(
        "PartSegCore.sprawl_utils.path_sprawl_cython",
        sources=["package/PartSegCore/sprawl_utils/path_sprawl_cython.pyx"],
        include_dirs=[np.get_include()] + [os.path.join(package_dir, "PartSegCore", "sprawl_utils")],
        language="c++",
        extra_compile_args=["-std=c++11"],
        extra_link_args=["-std=c++11"],
    ),
    Extension(
        "PartSegCore.sprawl_utils.sprawl_utils",
        sources=["package/PartSegCore/sprawl_utils/sprawl_utils.pyx"],
        include_dirs=[np.get_include()] + [os.path.join(package_dir, "PartSegCore", "sprawl_utils")],
        language="c++",
        extra_compile_args=["-std=c++11"],
        extra_link_args=["-std=c++11"],
    ),
    Extension(
        "PartSegCore.sprawl_utils.fuzzy_distance",
        sources=["package/PartSegCore/sprawl_utils/fuzzy_distance.pyx"],
        include_dirs=[np.get_include()] + [os.path.join(package_dir, "PartSegCore", "sprawl_utils")],
        language="c++",
        extra_compile_args=["-std=c++11"],
        extra_link_args=["-std=c++11"],
    ),
    Extension(
        "PartSegCore.color_image.color_image_cython",
        ["package/PartSegCore/color_image/color_image_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11"],
        language="c++",
    ),
    Extension(
        "PartSegCore.multiscale_opening.mso_bind",
        ["package/PartSegCore/multiscale_opening/mso_bind.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11", "-Wall"],
        language="c++",
        # undef_macros=["NDEBUG"],
        # define_macros=[("DEBUG", None)]
    ),
]


def read(*parts):
    with codecs.open(os.path.join(current_dir, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def readme():
    this_directory = os.path.abspath(os.path.dirname(__file__))
    reg = re.compile(r"(!\[[^]]*\])\((images/[^)]*)\)")
    reg2 = re.compile(r"PartSeg-lastest")
    with open(os.path.join(this_directory, "Readme.md")) as f:
        text = f.read()
        text = reg.sub(r"\1(https://raw.githubusercontent.com/4DNucleome/PartSeg/master/\2)", text)
        text = reg2.sub(f"PartSeg-{find_version('package', 'PartSeg', '__init__.py')}", text)
    with open(os.path.join(this_directory, "changelog.md")) as f:
        chg = f.read()
        text += "\n\n" + chg.replace("# ", "## ")
    return text


try:
    import PySide2

    qt_string = PySide2.__name__
except ImportError:
    qt_string = "PyQt5>=5.10.1"


setuptools.setup(
    ext_modules=cythonize(extensions),
    packages=setuptools.find_packages("./package"),
    package_dir={"": "package"},
    include_package_data=True,
    long_description=readme(),
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy>=1.16.0",
        "tifffile>=2019.7.26",
        "czifile>=2019.4.20",
        "oiffile>=2019.1.1",
        imagecodecs_string,
        "appdirs>=1.4.3",
        "SimpleITK>=1.1.0",
        "scipy>=0.19.1",
        "QtPy>=1.3.1",
        "sentry_sdk==0.14.1",
        qt_string,
        "six>=1.11.0",
        "h5py>=2.7.1",
        "packaging>=17.1",
        "pandas>=0.22.0",
        "sympy>=1.1.1",
        "Cython>=0.29.13",
        "openpyxl>=2.4.9",
        "xlrd>=1.1.0",
        "PartSegData==0.9.4",
        "defusedxml>=0.6.0",
    ],
)
