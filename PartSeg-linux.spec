# -*- mode: python -*-

block_cipher = None


a = Analysis(['main.py'],
             pathex=['/home/czaki/Dokumenty/projekty/PartSeg'],
             binaries=None,
             datas=[("clean_segment.tiff", "."), ("stack.tif", "."), ("icon.png", "."), ("icons/*", "icons"),("icons/bigicons/*", "icons/bigicons")],
             hiddenimports=["packaging", "packaging.version", "packaging.specifiers", "packaging.requirements", "h5py.defs", 'h5py.utils', 'h5py.h5ac', "h5py._proxy"],
             hookspath=["./hooks/linux/"],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

b = Analysis(['stack_main.py'],
             pathex=['/home/czaki/Dokumenty/projekty/PartSeg'],
             binaries=None,
             datas=[("stack.tif", "."), ("icon.png", "."), ("icons/*", "icons"),("icons/bigicons/*", "icons/bigicons")],
             hiddenimports=["packaging", "packaging.version", "packaging.specifiers", "packaging.requirements", "h5py.defs", 'h5py.utils', 'h5py.h5ac', "h5py._proxy"],
             hookspath=["./hooks/linux/"],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

MERGE((a, 'main', 'PartSeg'), (b, 'stack_main', 'StackSeg'))

a_pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
a_exe = EXE(a_pyz,
          a.scripts,
          exclude_binaries=True,
          name='PartSeg',
          debug=False,
          strip=False,
          upx=True,
          console=False,
          icon='icon.icns')

a_coll = COLLECT(a_exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='PartSeg')

b_pyz = PYZ(b.pure, b.zipped_data,
             cipher=block_cipher)
b_exe = EXE(b_pyz,
          b.scripts,
          exclude_binaries=True,
          name='StackSeg',
          debug=False,
          strip=False,
          upx=True,
          console=False,
          icon='icon.icns')

b_coll = COLLECT(b_exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='StackSeg')