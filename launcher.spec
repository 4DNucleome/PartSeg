# -*- mode: python -*-
from PyInstaller.building.build_main import Analysis, PYZ, EXE, BUNDLE, COLLECT
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

block_cipher = None
import sys
import os
from packaging.version import parse as parse_version
import platform
import zmq
import itertools
import debugpy._vendored
import importlib.metadata

sys.setrecursionlimit(5000)
sys.path.append(os.path.abspath("__file__"))


# import plugins
import PartSeg.__main__
import PartSegData.__init__

base_path = os.path.dirname(PartSeg.__main__.__file__)
data_path = os.path.dirname(PartSegData.__init__.__file__)

import napari

napari_version = parse_version(napari.__version__)

if napari_version <= parse_version("0.4.16"):
    from pathlib import Path

    import qtpy
    from napari._qt import qt_resources
    from napari.resources import ICON_PATH
    from napari.utils.misc import dir_hash

    def import_resources():
        qt_resources._register_napari_resources()
        icon_hash = dir_hash(ICON_PATH)  # get hash of icons folder contents
        key = f"_qt_resources_{qtpy.API_NAME}_{qtpy.QT_VERSION}_{icon_hash}"
        key = key.replace(".", "_")
        return Path(qt_resources.__file__).parent / f"{key}.py"
else:
    def import_resources():
        from napari import resources

        return os.path.join(os.path.dirname(resources.__file__), "icons")

import contextlib
from dask import config

import imagecodecs


plugins = []
if napari_version <= parse_version("0.4.16"):
    napari.plugins.plugin_manager.discover()
    plugins = [x.__name__ for x in napari.plugins.plugin_manager.plugins.values()]

from imageio.config.plugins import known_plugins as imageio_known_plugins

hiddenimports = (
    [f"imagecodecs.{y}" for y in (x if x[0] == "_" else f"_{x}" for x in imagecodecs._extensions())]
    + ["imagecodecs._shared"]
    + plugins
    + ["pkg_resources.py2_warn", "ipykernel.datapub"]
    + [
        "numpy.core._dtype_ctypes",
        "sentry_sdk.integrations.logging",
        "sentry_sdk.integrations.stdlib",
        "sentry_sdk.integrations.excepthook",
        "sentry_sdk.integrations.dedupe",
        "sentry_sdk.integrations.atexit",
        "sentry_sdk.integrations.modules",
        "sentry_sdk.integrations.argv",
        "sentry_sdk.integrations.threading",
        "numpy.random.common",
        "numpy.random.bounded_integers",
        "numpy.random.entropy",
        "PartSegCore.register",
        "PartSegCore.channel_class",
        "nme",
        "defusedxml.cElementTree",
        "vispy.app.backends._pyqt5",
        "magicgui.backends._qtpy",
        "freetype",
        "psygnal._signal",
        "psygnal._dataclass_utils",
        "psygnal._weak_callback",
        "imagecodecs._imagecodecs",
        "PartSeg.plugins.napari_widgets",
        "PartSegCore.napari_plugins",
    ]
    + [x.module_name for x in imageio_known_plugins.values()]
    + [x for x in collect_submodules("skimage") if "tests" not in x]
    + collect_submodules("scipy")
)


# psygnal handle
for package_path in importlib.metadata.files("psygnal"):
    if package_path.suffix in {".so", ".pyd"} and "psygnal" not in package_path.name:
        hiddenimports.append(package_path.name.split(".")[0])

hiddenimports.append("mypy_extensions")


with contextlib.suppress(ImportError):
    from sentry_sdk.integrations import _AUTO_ENABLING_INTEGRATIONS

    for el in _AUTO_ENABLING_INTEGRATIONS:
        hiddenimports.append(os.path.splitext(el)[0])
qt_data = []

# print(["plugins." + x.name for x in plugins.get_plugins()])

pyzmq_libs = os.path.abspath(os.path.join(os.path.dirname(zmq.__file__), os.pardir, "pyzmq.libs"))
pyzmq_data = []

if os.path.exists(pyzmq_libs):
    pyzmq_data = [(os.path.join(pyzmq_libs, x), "pyzmq.libs") for x in os.listdir(pyzmq_libs)]


napari_resource_path = import_resources()
napari_base_path = os.path.dirname(os.path.dirname(napari.__file__))
napari_resource_dest_path = os.path.relpath(os.path.dirname(napari_resource_path), napari_base_path)

packages = itertools.chain(
    importlib.metadata.entry_points(group="PartSeg.plugins"),
    importlib.metadata.entry_points(group="partseg.plugins"),
    importlib.metadata.entry_points(group="PartSegCore.plugins"),
    importlib.metadata.entry_points(group="partsegcore.plugins"),
)

plugins_data = [(os.path.join(base_path, "napari.yaml"), ".")]

for package in packages:
    module = package.load()
    if hasattr(module, "_hiddentimports"):
        hiddenimports += module._hiddentimports
    path_to_module = os.path.dirname(module.__file__)
    plugins_data.append(
        (os.path.join(path_to_module, "*.py"), os.path.join("plugins", os.path.basename(path_to_module)))
    )

if napari_version > parse_version("0.4.16"):
    import napari_builtins

    yaml_file = os.path.join(os.path.dirname(napari_builtins.__file__), "builtins.yaml")
    plugins_data.append((yaml_file, "napari_builtins"))


a = Analysis(
    [os.path.join(base_path, "launcher_main.py")],
    # pathex=['C:\\Users\\Grzegorz\\Documents\\segmentation-gui\\PartSeg'],
    binaries=[],
    datas=[
        (os.path.join(data_path, x), y)
        for x, y in [
            ("static_files/icons/*", "PartSegData/static_files/icons"),
            ("static_files/initial_images/*", "PartSegData/static_files/initial_images"),
            ("static_files/colors.npz", "PartSegData/static_files/"),
            ("fonts/*", "PartSegData/fonts/"),
        ]
    ]
    + qt_data
    + [(os.path.join(base_path, "napari.yaml"), "PartSeg")]
    + [(os.path.join(base_path, "plugins/napari_widgets/__init__.py"), "plugins/napari")]
    + [(os.path.join(base_path, "plugins/napari_widgets/simple_measurement_widget.py"), "plugins/napari")]
    # + [ ("Readme.md", "/"), ("changelog.md", "/")]
    + [(napari_resource_path, napari_resource_dest_path)]
    + [(os.path.join(os.path.dirname(config.__file__), "dask.yaml"), "dask")]
    + collect_data_files("dask")
    + collect_data_files("vispy")
    + collect_data_files("napari")
    + collect_data_files("napari_svg")
    + collect_data_files("napari_console")
    + collect_data_files("freetype")
    + collect_data_files("skimage")
    + collect_data_files("fonticon_fa6")
    + collect_data_files("jsonschema_specifications")
    + collect_data_files("PartSegCore-compiled-backend")
    + pyzmq_data
    + plugins_data
    + [(os.path.dirname(debugpy._vendored.__file__), "debugpy/_vendored")]
    + copy_metadata("PartSeg", recursive=True),
    hiddenimports=hiddenimports,
    # + ["plugins." + x.name for x in plugins.get_plugins()],
    hookspath=[],
    runtime_hooks=[],
    excludes=["tcl", "Tkconstants", "Tkinter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe_args = {
    "exclude_binaries": True,
    "name": "PartSeg",
    "debug": False,
    "bootloader_ignore_signals": False,
    "strip": False,
    "upx": True,
    "console": True,
}

if platform.system() == "Darwin":
    exe_args["icon"] = os.path.join(PartSegData.__init__.icons_dir, "icon.icns")
elif platform.system() == "Windows":
    exe_args["icon"] = os.path.join(PartSegData.__init__.icons_dir, "icon.ico")

exe = EXE(pyz, a.scripts, [], **exe_args)

if platform.system() == "Darwin2":
    app = BUNDLE(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        name="PartSeg.app",
        icon=os.path.join(PartSegData.__init__.icons_dir, "icon.icns"),
        bundle_identifier=None,
        info_plist={"NSHighResolutionCapable": "True"},
    )
else:
    coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=True, name="PartSeg")
