import functools

from PartSegCore.mask.io_functions import LoadROI
from PartSegCore.napari_plugins.loader import partseg_loader


def napari_get_reader(path: str):
    return next(
        (
            functools.partial(partseg_loader, LoadROI)
            for extension in LoadROI.get_extensions()
            if path.endswith(extension)
        ),
        None,
    )
