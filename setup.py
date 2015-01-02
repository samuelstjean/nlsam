#!/usr/bin/env python
from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext

import cython_gsl

ext = [Extension('nlsam.stabilizer',
                ['nlsam/stabilizer.pyx'],
                libraries=cython_gsl.get_libraries(),
                library_dirs=[cython_gsl.get_library_dir()],
                cython_include_dirs=[cython_gsl.get_cython_include_dir()])]

setup(
    name = 'gsl hyp1f1',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext
)
