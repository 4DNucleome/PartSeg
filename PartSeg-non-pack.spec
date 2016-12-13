# -*- mode: python -*-

block_cipher = None


a = Analysis(['main.py'],
             pathex=['C:\\Users\\Grzegorz\\Documents\\segmentation-gui'],
             binaries=None,
             datas=[("clean_segment.tiff", "."), ("icon.png", "."), ("icons/*", "icons"),("icons/bigicons/*", "icons/bigicons")],
             hiddenimports=["packaging", "packaging.version", "packaging.specifiers", "packaging.requirements"],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='PartSeg',
          debug=False,
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
