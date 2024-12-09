import functools

from PartSeg.plugins.napari_io.loader import partseg_loader
from PartSegCore.mask.io_functions import LoadROI


def napari_get_reader(path: str):
    return next(
        (
            functools.partial(partseg_loader, LoadROI)
            for extension in LoadROI.get_extensions()
            if path.endswith(extension)
        ),
        None,
    )
