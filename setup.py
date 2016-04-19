#!/usr/bin/env python

modlist = ['nlsam.utils',
           'nlsam.stabilizer']

params = {}
params['name'] = 'nlsam'
params['version'] = '0.2'
params['requires'] = ['cythongsl>=0.2.1',
                      'numpy>=1.10.4',
                      'cython>=0.21']
params['deps'] = ['scipy>=0.12',
                  'nibabel>=1.3',
                  'spams>=2.4']
params['links'] = ['https://github.com/samuelstjean/spams-python/archive/master.zip#egg=spams-2.5']

###############################################
# Build stuff is below this line
###############################################

import os
from os.path import join, exists, splitext

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if exists('MANIFEST'): os.remove('MANIFEST')

# Download setuptools if not present
try:
    import setuptools
except ImportError:
    import ez_setup
    ez_setup.use_setuptools()

try:
    import cython
except ImportError:
    raise ImportError('Could not find cython, which is required for building. \nTry running pip install cython')

from setuptools import setup, find_packages
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
from distutils.version import LooseVersion
from glob import glob

try:
    import numpy
except ImportError:
    raise ImportError('Could not find numpy, which is required for building. \nTry running pip install numpy')

try:
    from nibabel.optpkg import optional_package
except ImportError:
    raise ImportError('Could not find nibabel, which is required for building. \nTry running pip install nibabel')

cython_gsl, have_cython_gsl, _ = optional_package("cython_gsl")

if not have_cython_gsl:
    raise ImportError('cannot find gsl package (required for hyp1f1), \n\
        try pip install cythongsl and sudo apt-get install libgsl0-dev libgsl0ldbl')

# Check for local version of dipy if it exists, since it would replace a locally built
# but not installed version.
dipy, have_dipy, _ = optional_package("dipy")

if have_dipy:
    print('Found local version of dipy in ' + dipy.__file__)
    if LooseVersion(dipy.__version__) <= '0.11':
        raise ValueError('Local dipy version is {}, but you need at least 0.11!'.format(dipy.__version__))
else:
    print('Cannot find dipy, it will be installed using pip.')
    params['deps'].append('dipy>=0.11')

ext_modules = []

for pyxfile in modlist:

    ext_name = splitext(pyxfile)[0].replace('/', '.')
    source = join(*pyxfile.split('.')) + '.pyx'

    ext = Extension(pyxfile,
                    [source],
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
    setup_requires=params['requires'],
    install_requires=params['deps'] + params['requires'],
    dependency_links=params['links'],
    scripts=glob(join('scripts', '*'))
)
