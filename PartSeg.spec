# -*- mode: python -*-

block_cipher = None


a = Analysis(['main.py'],
             pathex=['C:\\Users\\Grzegorz\\Documents\\segmentation-gui'],
             binaries=None,
             datas=[("clean_segment.tiff", "."), ("icon.png", ".")],
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
          a.binaries,
          a.zipfiles,
          a.datas,
          name='PartSeg',
          debug=False,
          strip=False,
          upx=True,
          console=False , icon='icon.ico')