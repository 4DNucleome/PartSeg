import os
import importlib

def register():
    self_dir = os.path.dirname(__file__)
    packages = [x for x in os.listdir(self_dir) if os.path.isdir(os.path.join(self_dir, x)) and
                os.path.exists(os.path.join(self_dir, x, "__init__.py"))]

    for el in packages:
        importlib.import_module("."+el, __package__).register()

