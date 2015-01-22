#!/usr/bin/env python
from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext

import numpy
from dipy.utils.optpkg import optional_package

cython_gsl, have_cython_gsl, _ = optional_package("cython_gsl")

if not have_cython_gsl:
    raise ValueError('cannot find gsl package (required for hyp1f1), \
        try pip install cythongsl and sudo apt-get install libgsl0-dev libgsl0ldbl')

ext = [Extension('scilpy.denoising.stabilizer',
                ['scilpy/denoising/stabilizer.pyx'],
                libraries=cython_gsl.get_libraries(),
                library_dirs=[cython_gsl.get_library_dir()],
                cython_include_dirs=[cython_gsl.get_cython_include_dir()],
                include_dirs=[numpy.get_include()])]

setup(
    name = 'gsl hyp1f1',
    include_dirs = [cython_gsl.get_include()],
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext
)
