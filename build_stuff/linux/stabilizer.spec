# -*- mode: python -*-

block_cipher = None

<<<<<<< 20d4e6a9b256572fdf4aae7e0b95083285a5563c
a = Analysis(['../../scripts/stabilizer'],
=======

a = Analysis(['scripts/stabilizer'],
>>>>>>> added linux build stuff
             pathex=['/home/samuel/nlsam'],
             binaries=None,
             datas=None,
             hiddenimports=['scipy.special._ufuncs_cxx',
                            'scipy.linalg.cython_blas',
                            'scipy.linalg.cython_lapack',
                            'scipy.integrate',
                            'cython_gsl',
                            'scipy.special',
                            'scipy.integrate.quadrature',
                            'scipy.integrate.odepack',
                            'scipy.integrate._odepack',
                            'scipy.integrate.quadpack',
                            'scipy.integrate._quadpack',
                            'scipy.integrate._ode',
                            'scipy.integrate.vode',
                            'scipy.integrate._dop',
                            'scipy.integrate.lsoda'],
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
          name='stabilizer',
          debug=False,
          strip=False,
          upx=True,
          console=True )
