#!/usr/bin/env python

# Download setuptools if not present
try:
    import setuptools
except ImportError:
    import ez_setup
    ez_setup.use_setuptools()

from setuptools import setup, find_packages
from Cython.Distutils import Extension
from Cython.Distutils import build_ext

from glob import glob
from os.path import splitext, join

import numpy
from dipy.utils.optpkg import optional_package

cython_gsl, have_cython_gsl, _ = optional_package("cython_gsl")

if not have_cython_gsl:
    raise ValueError('cannot find gsl package (required for hyp1f1), \
        try pip install cythongsl and sudo apt-get install libgsl0-dev libgsl0ldbl')

params = {}
params['name'] = 'nlsam'
params['version'] = '0.1'
params['requires'] = ['cythongsl', 'spams', 'numpy>=1.7.1', 'cython>=0.21']
params['deps'] = ['dipy>=0.11',
                  'scipy>=0.12',
                  'nibabel>=1.3']
# params['links'] = ['https://github.com/samuelstjean/spams-python/releases/tag/0.1#egg=spams-2.5']
# params['links'] = ['https://github.com/samuelstjean/spams-python/releases/download/0.1/spams-python-v2.5-svn2014-07-04.tar.gz#egg=spams-2.5']
params['links'] = ['git+https://github.com/samuelstjean/spams-python/releases/download/0.1/spams-python-v2.5-svn2014-07-04.tar.gz#egg=spams-2.5']
# https://github.com/samuelstjean/spams-python/releases/tag/0.1
#/0.1/spams-python-v2.5-svn2014-07-04.tar.gz']

ext_modules = []
for pyxfile in glob(join('nlsam', '*.pyx')):

    ext_name = splitext(pyxfile)[0].replace('/', '.')
    ext = Extension(ext_name,
                    [pyxfile],
                    libraries=cython_gsl.get_libraries(),
                    library_dirs=[cython_gsl.get_library_dir()],
                    cython_include_dirs=[cython_gsl.get_cython_include_dir()],
                    include_dirs=[numpy.get_include()])

    ext_modules.append(ext)

setup(
    name=params['name'],
    version=params['version'],
    include_dirs=[cython_gsl.get_include()],
    packages=find_packages(),
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    dependency_links=params['links'],
    setup_requires=params['requires'],
    install_requires=params['deps'] + params['requires'],
    scripts=glob(join('scripts', '*')),
)
