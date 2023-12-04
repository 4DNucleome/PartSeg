import importlib
import itertools
import pkgutil
import typing
from importlib.metadata import entry_points


def get_plugins():
    packages = pkgutil.iter_modules(__path__, f"{__name__}.")
    packages2 = itertools.chain(
        entry_points().get("PartSegCore.plugins", []),
        entry_points().get("partsegcore.plugins", []),
    )
    return [importlib.import_module(el.name) for el in packages] + [el.load() for el in packages2]


plugins_loaded = set()


def register():
    for el in get_plugins():
        if hasattr(el, "register") and el.__name__ not in plugins_loaded:
            if not isinstance(el.register, typing.Callable):  # pragma: no cover
                raise TypeError(f"Plugin {el.__name__} has no register method")
            el.register()
            plugins_loaded.add(el.__name__)


def register_if_need():
    if len(plugins_loaded) == 0:
        register()
