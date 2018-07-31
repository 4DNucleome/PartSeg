# -*- mode: python -*-

block_cipher = None


a = Analysis(['main.py'],
             pathex=['/Users/grzegorzbokota/Documents/projekty/segmentacja-gui'],
             binaries=None,
             datas=[("clean_segment.tiff", "."), ("icon.png", "."), ("icons/*", "icons"),("icons/bigicons/*", "icons/bigicons")],
             hiddenimports=["packaging", "packaging.version", "packaging.specifiers", "packaging.requirements"],
             hookspath=["./hooks/mac/"],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

b = Analysis(['stack_main.py'],
             pathex=['/Users/grzegorzbokota/Documents/projekty/segmentacja-gui'],
             binaries=None,
             datas=[("clean_segment.tiff", "."), ("icon.png", "."), ("icons/*", "icons"),("icons/bigicons/*", "icons/bigicons")],
             hiddenimports=["packaging", "packaging.version", "packaging.specifiers", "packaging.requirements"],
             hookspath=["./hooks/mac/"],
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
          name='PartSeg',
          debug=False,
          strip=False,
          upx=True,
          console=False,
          icon='icon.icns')

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='PartSeg')

"""app = BUNDLE(exe,
             a.binaries,
             a.zipfiles,
             a.datas,
             name='PartSeg.app',
             icon='icon.icns',
             bundle_identifier=None,
             info_plist={
             'NSHighResolutionCapable': 'True'
             },)"""
