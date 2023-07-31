import functools
import os.path

from PartSegCore.mask.io_functions import LoadStackImageWithMask
from PartSegCore.napari_plugins.loader import partseg_loader


def napari_get_reader(path: str):
    return next(
        (
            functools.partial(partseg_loader, LoadStackImageWithMask)
            for extension in LoadStackImageWithMask.get_extensions()
            if path.endswith(extension) and os.path.exists(LoadStackImageWithMask.get_next_file([path]))
        ),
        None,
    )
