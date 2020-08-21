import functools
import os.path

from napari_plugin_engine import napari_hook_implementation

from PartSegCore.analysis.load_functions import LoadProject
from PartSegCore.napari_plugins.loader import partseg_loader


@napari_hook_implementation
def napari_get_reader(path: str):
    for extension in LoadProject.get_extensions():
        if path.endswith(extension) and os.path.exists(LoadProject.get_next_file([path])):
            return functools.partial(partseg_loader, LoadProject)
