# -*- mode: python -*-

block_cipher = None
import sys
import os
sys.setrecursionlimit(5000)
sys.path.append(os.path.dirname('__file__'))

import tifffile
import plugins

num = tifffile.__version__.split(".")[0]

if num == '0':
    hiddenimports = ["tifffile._tifffile"]
else:
    hiddenimports = ["imagecodecs._imagecodecs"]

print(["plugins." + x.name for x in plugins.get_plugins()])

a = Analysis(['launcher_main.py'],
             # pathex=['C:\\Users\\Grzegorz\\Documents\\segmentation-gui\\src'],
             binaries=[],
             datas=[("static_files/icons/*", "static_files/icons"), ("static_files/initial_images/*", "static_files/initial_images"), ("static_files/colors.npz", "static_files/")],
             hiddenimports=["tifffile._tifffile", "plugins"], # + ["plugins." + x.name for x in plugins.get_plugins()],
             hookspath=[os.path.join(os.path.dirname('__file__'), "hooks")],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='PartSeg',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='PartSeg')
