import functools

from PartSeg.plugins.napari_io.loader import partseg_loader
from PartSegCore.analysis.load_functions import LoadStackImage


def napari_get_reader(path: str):
    return next(
        (
            functools.partial(partseg_loader, LoadStackImage)
            for extension in LoadStackImage.get_extensions()
            if path.lower().endswith(extension.lower())
        ),
        None,
    )
