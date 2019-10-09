import os
import sys

from .image import Image
from .image_reader import TiffImageReader, TiffFileException, CziImageReader, GenericImageReader
from .image_writer import ImageWriter
from . import tifffile_fixes

__all__ = ("Image", "TiffImageReader", "ImageWriter", "TiffFileException", "CziImageReader", "GenericImageReader")

if os.path.basename(sys.argv[0]) in ["sphinx-build", "sphinx-build.exe"]:
    for el in __all__:
        globals()[el].__module__ = __name__
