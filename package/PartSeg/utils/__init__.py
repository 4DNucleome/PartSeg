import importlib
import os
import sys
from importlib.abc import Loader, MetaPathFinder


class MyLoader(Loader):
    def module_repr(self, module):
        return repr(module)

    def load_module(self, fullname):
        old_name = fullname
        names = fullname.split(".")
        names[1] = "PartSegCore"
        fullname = ".".join(names[1:])
        module = importlib.import_module(fullname)
        sys.modules[old_name] = module
        return module


class MyImport(MetaPathFinder):
    def find_module(self, fullname, path=None):
        names = fullname.split(".")
        if len(names) >= 2 and names[0] == "PartSeg" and names[1] == "utils":
            return MyLoader()


sys.meta_path.append(MyImport())

print(
    "[Warning] PartSeg.utils module name is deprecated. It is renamed to PartSegCore "
    "To fail this import set environment variable 'NO_DEPRECATED' to 1",
    file=sys.stderr,
)

if "NO_DEPRECATED" in os.environ and os.environ["NO_DEPRECATED"] == "1":
    raise ImportError("PartSeg.utils is deprecated and 'NO_DEPRECATED' is set to 1")
