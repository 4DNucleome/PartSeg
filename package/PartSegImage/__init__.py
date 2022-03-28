import os
import sys

from . import tifffile_fixes  # noqa: F401
from .channel_class import Channel
from .image import Image
from .image_reader import (
    CziImageReader,
    GenericImageReader,
    ObsepImageReader,
    OifImagReader,
    TiffFileException,
    TiffImageReader,
)
from .image_writer import BaseImageWriter, IMAGEJImageWriter, ImageWriter

__all__ = (
    "BaseImageWriter",
    "Channel",
    "Image",
    "TiffImageReader",
    "IMAGEJImageWriter",
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
