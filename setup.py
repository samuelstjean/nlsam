#!/usr/bin/env python

import os
import sys
import subprocess

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

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

from nlsam import get_setup_params
params = get_setup_params()
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

if sys.platform.startswith('win'):
    gsl_path = 'gsl_windows'
    libext = '.lib'
    params['package_data'] = {"nlsam": ['gsl.dll']}
elif sys.platform.startswith('darwin'):
    gsl_path = 'gsl_mac'
    libext = '.a'
elif sys.platform.startswith('linux'):
    gsl_path = 'gsl_linux'
    libext = '.a'
else:
    gsl_path = None

dir_path = os.path.dirname(os.path.realpath(__file__))
gsl_include = os.path.join(dir_path, 'nlsam', 'gsl_libs')
libs = ['libgsl', 'libgslcblas']

if gsl_path is not None:
    gsl_path = os.path.join(dir_path, 'nlsam', 'gsl_libs', gsl_path)
    gsl_libraries = [os.path.join(gsl_path, lib) for lib in libs]
else:
    # this part hardcodes the .a libs and their name, so it might need to be changed
    # on some system. Also, it requires the static libs version to be available.
    print('Cannot guess current OS, using system GSL libs')
    gsl_path = subprocess.check_output('gsl-config --libs', shell=True).decode('utf-8').split()[0][2:]
    gsl_libraries = [gsl_path]
    libext = '.a'

gsl_libraries_ext = [os.path.join(gsl_path, lib + libext) for lib in libs]
params['include_dirs'] = [gsl_path]

# list of pyx modules to compile
modules = ['nlsam.utils',
           'nlsam.stabilizer']
params['ext_modules'] = []

for pyxfile in modules:

    ext_name = os.path.splitext(pyxfile)[0].replace('/', '.')
    source = os.path.join(*pyxfile.split('.')) + '.pyx'

    ext = Extension(pyxfile,
                    [source],
                    libraries=gsl_libraries,
                    library_dirs=[gsl_path],
                    cython_include_dirs=[gsl_include],
                    include_dirs=[numpy.get_include(), gsl_include],
                    extra_objects=gsl_libraries_ext)

    params['ext_modules'].append(ext)

setup(**params)
