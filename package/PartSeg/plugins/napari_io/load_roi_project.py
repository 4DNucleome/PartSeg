import functools
import os.path

from PartSeg.plugins.napari_io.loader import partseg_loader
from PartSegCore.analysis.load_functions import LoadProject


def napari_get_reader(path: str):
    return next(
        (
            functools.partial(partseg_loader, LoadProject)
            for extension in LoadProject.get_extensions()
            if path.endswith(extension) and os.path.exists(LoadProject.get_next_file([path]))
        ),
        None,
    )
