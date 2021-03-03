# -*- mode: python -*-
from PyInstaller.building.build_main import Analysis, PYZ, EXE, BUNDLE, COLLECT
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None
import sys
import os
from packaging.version import parse as parse_version
import platform
import zmq

sys.setrecursionlimit(5000)
sys.path.append(os.path.dirname("__file__"))


# import plugins
import PartSeg.__main__
import PartSegData.__init__

base_path = os.path.dirname(PartSeg.__main__.__file__)
data_path = os.path.dirname(PartSegData.__init__.__file__)

import napari

napari_version = parse_version(napari.__version__)

if napari_version < parse_version("0.4.3"):
    from napari.resources import import_resources
elif napari_version < parse_version("0.4.6"):
    from napari._qt.qt_resources import import_resources as napari_import_resources

    def import_resources():
        return napari_import_resources()[0]
else:
    from pathlib import Path

    import qtpy
    from napari._qt import qt_resources
    from napari.resources import ICON_PATH
    from napari.utils.misc import dir_hash

    def import_resources():
        qt_resources._register_napari_resources()
        icon_hash = dir_hash(ICON_PATH)  # get hash of icons folder contents
        key = f'_qt_resources_{qtpy.API_NAME}_{qtpy.QT_VERSION}_{icon_hash}'
        key = key.replace(".", "_")
        save_path = Path(qt_resources.__file__).parent / f"{key}.py"
        return save_path


from dask import config

import imagecodecs

napari.plugins.plugin_manager.discover()

hiddenimports = ["imagecodecs._" + x for x in imagecodecs._extensions()] +\
                ["imagecodecs._shared"] + [x.__name__ for x in napari.plugins.plugin_manager.plugins.values()] + \
                ["pkg_resources.py2_warn", "scipy.special.cython_special", "ipykernel.datapub"] + [
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
                    "defusedxml.cElementTree",
                    "vispy.app.backends._pyqt5",
                    "scipy.spatial.transform._rotation_groups",
                    "magicgui.backends._qtpy",
                ]

try:
    from sentry_sdk.integrations import _AUTO_ENABLING_INTEGRATIONS
    for el in _AUTO_ENABLING_INTEGRATIONS:
        hiddenimports.append(os.path.splitext(el)[0])
except ImportError:
    pass

if platform.system() == "Windows":
    import PyQt5

    qt_path = os.path.dirname(PyQt5.__file__)
    qt_data = [(os.path.join(qt_path, "Qt", "bin", "Qt5Core.dll"), os.path.join("PyQt5", "Qt", "bin"))]
else:
    qt_data = []

# print(["plugins." + x.name for x in plugins.get_plugins()])

pyzmq_libs = os.path.abspath(os.path.join(os.path.dirname(zmq.__file__), os.pardir, "pyzmq.libs"))
pyzmq_data = []

if os.path.exists(pyzmq_libs):
    pyzmq_data = [(os.path.join(pyzmq_libs, x), "pyzmq.libs") for x in os.listdir(pyzmq_libs)]

napari_resource_path = import_resources()
napari_base_path = os.path.dirname(os.path.dirname(napari.__file__))
napari_resource_dest_path = os.path.relpath(os.path.dirname(napari_resource_path), napari_base_path)

a = Analysis(
    ["package/PartSeg/launcher_main.py"],
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
    + [(os.path.join(base_path, "plugins/itk_snap_save/__init__.py"), "plugins/itk_snap_save")]
    + [(os.path.join(base_path, "plugins/smFish_project/*"), "plugins/napari_copy_labels")]
    + [(napari_resource_path, napari_resource_dest_path)]
    + [(os.path.join(os.path.dirname(config.__file__), "dask.yaml"), "dask")]
    + collect_data_files("dask")
    + collect_data_files("vispy")
    + collect_data_files("napari")
    + pyzmq_data,
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
