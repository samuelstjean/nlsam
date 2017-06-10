#!/usr/bin/env python

params = {}
params['modules'] = ['nlsam.utils',
                     'nlsam.stabilizer']
params['scripts'] = ['scripts/nlsam_denoising']
params['name'] = 'nlsam'
params['author'] = 'Samuel St-Jean'
params['author_email'] = 'samuel@isi.uu.nl'
params['url'] = 'https://github.com/samuelstjean/nlsam'
params['version'] = '0.5.1'
params['dependencies'] = ['numpy>=1.10.4',
                          'scipy>=0.14',
                          'cython>=0.21',
                          'cythongsl>=0.2.1',
                          'nibabel>=2.0',
                          'spams>=2.4']
params['links'] = ['https://github.com/samuelstjean/spams-python/releases/download/0.1/spams-python-v2.5-python3.zip#egg=spams-2.5']

###############################################
# Build stuff is below this line
###############################################

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
