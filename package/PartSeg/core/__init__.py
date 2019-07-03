import sys
from importlib.abc import MetaPathFinder, Loader
import importlib
# noinspection PyUnresolvedReferences
from PartSeg.utils import *


class MyLoader(Loader):
    def module_repr(self, module):
        return repr(module)

    def load_module(self, fullname):
        old_name = fullname
        names = fullname.split(".")
        names[1] = "utils"
        fullname = ".".join(names)
        module = importlib.import_module(fullname)
        sys.modules[old_name] = module
        return module
        

class MyImport(MetaPathFinder):
    def find_module(self, fullname, path=None):
        names = fullname.split(".")
        if len(names) >= 2 and names[0] == "PartSeg" and names[1] == "core":
            return MyLoader()


sys.meta_path.append(MyImport())
