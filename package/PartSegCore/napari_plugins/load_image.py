import functools

from napari_plugin_engine import napari_hook_implementation

from PartSegCore.analysis.load_functions import LoadStackImage
from PartSegCore.napari_plugins.loader import partseg_loader


@napari_hook_implementation
def napari_get_reader(path: str):
    for extension in LoadStackImage.get_extensions():
        if path.endswith(extension):
            return functools.partial(partseg_loader, LoadStackImage)
