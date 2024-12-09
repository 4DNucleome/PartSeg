import os
import sys

from PartSegImage import tifffile_fixes  # noqa: F401
from PartSegImage.channel_class import Channel
from PartSegImage.image import ChannelInfo, ChannelInfoFull, Image
from PartSegImage.image_reader import (
    CziImageReader,
    GenericImageReader,
    ObsepImageReader,
    OifImagReader,
    TiffFileException,
    TiffImageReader,
)
from PartSegImage.image_writer import BaseImageWriter, IMAGEJImageWriter, ImageWriter

__all__ = (
    "BaseImageWriter",
    "Channel",
    "ChannelInfo",
    "ChannelInfoFull",
    "CziImageReader",
    "GenericImageReader",
    "IMAGEJImageWriter",
    "Image",
    "ImageWriter",
    "ObsepImageReader",
    "OifImagReader",
    "TiffFileException",
    "TiffImageReader",
)

if os.path.basename(sys.argv[0]) in ["sphinx-build", "sphinx-build.exe"]:  # pragma: no cover
    for el in __all__:
        globals()[el].__module__ = __name__
