import codecs
import os
import re

import numpy as np
from setuptools import Extension, setup

current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.join(current_dir, "package")


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


def readme():
    this_directory = os.path.abspath(os.path.dirname(__file__))
    reg = re.compile(r"(!\[[^]]*\])\((images/[^)]*)\)")
    reg2 = re.compile(r"PartSeg-lastest")
    with open(os.path.join(this_directory, "Readme.md")) as f:
        text = f.read()
        text = reg.sub(r"\1(https://raw.githubusercontent.com/4DNucleome/PartSeg/master/\2)", text)
        try:
            from setuptools_scm import get_version

            text = reg2.sub(f"PartSeg-{get_version()}", text)
        except ImportError:
            pass
    with open(os.path.join(this_directory, "changelog.md")) as f:
        chg = f.read()
        text += "\n\n" + chg.replace("# ", "## ")
    return text


changelog_path = os.path.join(os.path.dirname(__file__), "changelog.md")
changelog_result_path = os.path.join(os.path.dirname(__file__), "package", "PartSeg", "changelog.py")
if os.path.exists(changelog_path):
    with open(changelog_path) as ff:
        changelog_str = ff.read()
    with open(changelog_result_path, "w") as ff:
        ff.write(f'changelog = """\n{changelog_str}"""\n')


setup(
    ext_modules=extensions,
    include_package_data=True,
    long_description=readme(),
    long_description_content_type="text/markdown",
    use_scm_version=True,
)
