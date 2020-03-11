# -*- mode: python -*-

block_cipher = None

a = Analysis(['scripts/nlsam_denoising'],
             pathex=['./'],
             datas=None,
             hiddenimports=['nlsam',
                            'spams',
                            'scipy.special',
                            'scipy.special.cython_special'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure,
          a.zipped_data,
          cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='nlsam_denoising',
          debug=False,
          strip=False,
          upx=False,
          console=True)
