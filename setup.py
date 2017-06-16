#!/usr/bin/env python

import os
from os.path import join, exists, splitext

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if exists('MANIFEST'):
    os.remove('MANIFEST')

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

try:
    import numpy
except ImportError:
    raise ImportError('Could not find numpy, which is required for building. \nTry running pip install numpy')

try:
    import cython_gsl
except ImportError:
    error = 'Cannot find gsl package (required for hyp1f1), \n' + \
            'try pip install cythongsl and \nsudo apt-get install libgsl0-dev libgsl0ldbl on Ubuntu and friends' + \
            '\nor\n brew install gsl on mac'
    raise ImportError(error)

from nlsam import get_setup_params
params = get_setup_params()

# Check for local version of dipy if it exists, since it would replace a locally built
# but not installed version.
try:
    import dipy
    print('Found local version of dipy in ' + dipy.__file__)
    if LooseVersion(dipy.__version__) < LooseVersion('0.11'):
        raise ValueError('Local dipy version is {}, but you need at least 0.11!'.format(dipy.__version__))
except ImportError:
    print('Cannot find dipy, it will be installed using pip.')
    params['dependencies'].append('dipy>=0.11')

ext_modules = []

for pyxfile in params['modules']:

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
    author=params['author'],
    author_email=params['author_email'],
    url=params['url'],
    include_dirs=[cython_gsl.get_include()],
    packages=find_packages(),
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    install_requires=params['dependencies'],
    dependency_links=params['links'],
    scripts=params['scripts']
)
