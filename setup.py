#!/usr/bin/env python

import os
import sys

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
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

if sys.platform.startswith('win'):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    gsl_path = os.path.join(dir_path, 'nlsam', 'gsl_windows')
    os.environ["LIB_GSL"] = gsl_path
    sys.path.append(gsl_path)

from nlsam import get_setup_params
params = get_setup_params()
params['include_dirs'] = [cython_gsl.get_include()]
params['packages'] = find_packages()
params['cmdclass'] = {'build_ext': build_ext}

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

# list of pyx modules to compile
modules = ['nlsam.utils',
           'nlsam.stabilizer']
params['ext_modules'] = []

for pyxfile in modules:

    ext_name = os.path.splitext(pyxfile)[0].replace('/', '.')
    source = os.path.join(*pyxfile.split('.')) + '.pyx'

    ext = Extension(pyxfile,
                    [source],
                    libraries=cython_gsl.get_libraries(),
                    library_dirs=[cython_gsl.get_library_dir()],
                    cython_include_dirs=[cython_gsl.get_cython_include_dir()],
                    include_dirs=[numpy.get_include()])

    params['ext_modules'].append(ext)

setup(**params)
