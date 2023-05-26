import contextlib
import importlib
import itertools
import os
import pkgutil
import sys
import typing
from importlib.metadata import entry_points

import PartSegCore.plugins


def register_napari_plugins():
    import napari
    import napari_plugin_engine
    import napari_svg

    from PartSeg.plugins import napari_widgets
    from PartSegCore.napari_plugins import (
        load_image,
        load_mask_project,
        load_masked_image,
        load_roi_project,
        save_mask_roi,
    )

    for module in [
        napari_svg,
        napari_plugin_engine,
        load_image,
        load_mask_project,
        load_masked_image,
        load_roi_project,
        save_mask_roi,
        napari_widgets,
    ]:
        with contextlib.suppress(ValueError):
            napari.plugins.plugin_manager.register(module)


def get_plugins():
    if getattr(sys, "frozen", False):
        new_path = [os.path.join(os.path.dirname(os.path.dirname(__path__[0])), "plugins")]
        packages = pkgutil.iter_modules(new_path, "plugins.")
        register_napari_plugins()
    else:
        packages = pkgutil.iter_modules(__path__, f"{__name__}.")
    packages2 = itertools.chain(
        entry_points().get("PartSeg.plugins", []),
        entry_points().get("partseg.plugins", []),
    )
    return [importlib.import_module(el.name) for el in packages] + [el.load() for el in packages2]


plugins_loaded = set()


def register():
    PartSegCore.plugins.register()
    for el in get_plugins():
        if hasattr(el, "register") and el.__name__ not in plugins_loaded:
            if not isinstance(el.register, typing.Callable):  # pragma: no cover
                raise TypeError(f"Plugin {el.__name__} has no register method")
            el.register()
            plugins_loaded.add(el.__name__)


def register_if_need():
    if len(plugins_loaded) == 0:
        register()
