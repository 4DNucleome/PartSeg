import os
import sys

from . import tifffile_fixes  # noqa: F401
from .image import Image
from .image_reader import (
    CziImageReader,
    GenericImageReader,
    ObsepImageReader,
    OifImagReader,
    TiffFileException,
    TiffImageReader,
)
from .image_writer import ImageWriter

__all__ = (
    "Image",
    "TiffImageReader",
    "ImageWriter",
    "TiffFileException",
    "CziImageReader",
    "OifImagReader",
    "ObsepImageReader",
    "GenericImageReader",
)

if os.path.basename(sys.argv[0]) in ["sphinx-build", "sphinx-build.exe"]:  # pragma: no cover
    for el in __all__:
        globals()[el].__module__ = __name__
