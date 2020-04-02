# -*- mode: python -*-
from PyInstaller.building.build_main import Analysis, PYZ, EXE, BUNDLE, COLLECT

block_cipher = None
import sys
import os
import platform
sys.setrecursionlimit(5000)
sys.path.append(os.path.dirname('__file__'))


# import plugins
import PartSeg.__main__
import PartSegData.__init__
base_path = os.path.dirname(PartSeg.__main__.__file__)
data_path = os.path.dirname(PartSegData.__init__.__file__)


import imagecodecs
hiddenimports = ["imagecodecs._" + x for x in imagecodecs._extensions()] + ["imagecodecs._shared"]


if platform.system() == "Windows":
    import PyQt5
    qt_path = os.path.dirname(PyQt5.__file__)
    qt_data = [(os.path.join(qt_path, "Qt", "bin", "Qt5Core.dll"), os.path.join("PyQt5", "Qt", "bin"))]
else:
    qt_data = []

# print(["plugins." + x.name for x in plugins.get_plugins()])

a = Analysis(['package/PartSeg/launcher_main.py'],
             # pathex=['C:\\Users\\Grzegorz\\Documents\\segmentation-gui\\PartSeg'],
             binaries=[],
             datas = [(os.path.join(data_path, x), y) for x, y in [
                 ("static_files/icons/*", "PartSegData/static_files/icons"),
                 ("static_files/initial_images/*", "PartSegData/static_files/initial_images"),
                 ("static_files/colors.npz", "PartSegData/static_files/"),
                 ("fonts/*", "PartSegData/fonts/")]] + qt_data +
                     [(os.path.join(base_path, "plugins/itk_snap_save/__init__.py"), "plugins/itk_snap_save")],
             hiddenimports=hiddenimports + [
                 'numpy.core._dtype_ctypes', 'sentry_sdk.integrations.logging', 'sentry_sdk.integrations.stdlib',
                 'sentry_sdk.integrations.excepthook','sentry_sdk.integrations.dedupe', 'sentry_sdk.integrations.atexit'
                 , 'sentry_sdk.integrations.modules', 'sentry_sdk.integrations.argv',
                 'sentry_sdk.integrations.threading', 'numpy.random.common', 'numpy.random.bounded_integers',
                 'numpy.random.entropy', "PartSegCore.register", "defusedxml.cElementTree"],
             # + ["plugins." + x.name for x in plugins.get_plugins()],
             hookspath=[],
             runtime_hooks=[],
             excludes=['tcl', 'Tkconstants', 'Tkinter'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe_args = {
    "exclude_binaries": True,
    "name": 'PartSeg',
    "debug": False,
    "bootloader_ignore_signals": False,
    "strip": False,
    "upx": True,
    "console": True
}

if platform.system() == "Darwin":
    exe_args["icon"] = os.path.join(PartSegData.__init__.icons_dir, "icon.icns")
elif platform.system() == "Windows":
    exe_args["icon"] = os.path.join(PartSegData.__init__.icons_dir, "icon.ico")

exe = EXE(pyz,
          a.scripts,
          [],
          **exe_args
          )

if platform.system() == "Darwin":
    app = BUNDLE(exe,
                 a.binaries,
                 a.zipfiles,
                 a.datas,
                 name="PartSeg.app",
                 icon=os.path.join(PartSegData.__init__.icons_dir, "icon.icns"),
                 bundle_identifier=None,
                 info_plist={
                     'NSHighResolutionCapable': 'True'
                 })
else:
    coll = COLLECT(exe,
                   a.binaries,
                   a.zipfiles,
                   a.datas,
                   strip=False,
                   upx=True,
                   name='PartSeg')
