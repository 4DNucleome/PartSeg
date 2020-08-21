import functools
import os.path

from napari_plugin_engine import napari_hook_implementation

from PartSegCore.mask.io_functions import LoadStackImageWithMask
from PartSegCore.napari_plugins.loader import partseg_loader


@napari_hook_implementation
def napari_get_reader(path: str):
    for extension in LoadStackImageWithMask.get_extensions():
        if path.endswith(extension) and os.path.exists(LoadStackImageWithMask.get_next_file([path])):
            return functools.partial(partseg_loader, LoadStackImageWithMask)
