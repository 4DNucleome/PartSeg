import contextlib
import importlib
import logging
import os
import pkgutil
import sys
import typing
from pathlib import Path

import PartSegCore.plugins


def register_napari_plugins():  # pragma: no cover
    import npe2

    import PartSeg

    with contextlib.suppress(ValueError):
        npe2.PluginManager.instance().register(
            os.path.join(os.path.dirname(os.path.dirname(PartSeg.__file__)), "napari.yaml")
        )
        base_path = Path(__file__).parent.parent.parent / "plugins"
        for el in base_path.glob("*/napari.yaml"):
            try:
                npe2.PluginManager.instance().register(str(el))
            except Exception:  # noqa: PERF203
                logging.exception("Failed to register %s", el)


def get_plugins():
    if getattr(sys, "frozen", False):  # pragma: no cover
        new_path = [os.path.join(os.path.dirname(os.path.dirname(__path__[0])), "plugins")]
        sys.path.append(new_path[0])
        packages = pkgutil.iter_modules(new_path)
        register_napari_plugins()
    else:
        sys.path.append(os.path.dirname(__file__))
        packages = list(pkgutil.iter_modules(__path__))
    packages2 = PartSegCore.plugins.iter_entrypoints("PartSeg.plugins")
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
