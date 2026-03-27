import functools
import os.path

from PartSeg.plugins.napari_io.loader import partseg_loader
from PartSegCore.mask.io_functions import LoadStackImageWithMask


def napari_get_reader(path: str):
    return next(
        (
            functools.partial(partseg_loader, LoadStackImageWithMask)
            for extension in LoadStackImageWithMask.get_extensions()
            if path.endswith(extension) and os.path.exists(LoadStackImageWithMask.get_next_file([path]))
        ),
        None,
    )
