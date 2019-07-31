import os
import sys

from .image import Image
from .image_reader import ImageReader, TiffFileException
from .image_writer import ImageWriter

__all__ = ("Image", "ImageReader", "ImageWriter", "TiffFileException")

if os.path.basename(sys.argv[0]) in ["sphinx-build", "sphinx-build.exe"]:
    for el in __all__:
        globals()[el].__module__ = __name__
