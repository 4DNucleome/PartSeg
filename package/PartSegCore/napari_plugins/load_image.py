import functools

from PartSegCore.analysis.load_functions import LoadStackImage
from PartSegCore.napari_plugins.loader import partseg_loader


def napari_get_reader(path: str):
    return next(
        (
            functools.partial(partseg_loader, LoadStackImage)
            for extension in LoadStackImage.get_extensions()
            if path.lower().endswith(extension.lower())
        ),
        None,
    )
