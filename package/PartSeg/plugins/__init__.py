import importlib
import os
import pkgutil
import pkg_resources
import sys


def get_plugins():
    if getattr(sys, "frozen", False):
        new_path = [os.path.join(os.path.dirname(os.path.dirname(__path__[0])), "plugins")]
        packages = pkgutil.iter_modules(new_path, "plugins" + ".")
    else:
        packages = pkgutil.iter_modules(__path__, __name__ + ".")
    packages2 = pkg_resources.iter_entry_points("PartSeg.plugins")
    return [importlib.import_module(el.name) for el in packages] + [el.load() for el in packages2]


plugins_loaded = set()


def register():
    for el in get_plugins():
        if hasattr(el, "register") and el.__name__ not in plugins_loaded:
            el.register()
            plugins_loaded.add(el.__name__)
