#!/usr/bin/env python

import os
import sys
import io
import subprocess
import numpy

from setuptools import setup, find_packages
from Cython.Distutils import Extension, build_ext

with io.open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

if sys.platform.startswith('win'):
    gsl_path = 'gsl_windows'
    libext = '.lib'
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

# list of pyx modules to compile
modules = ['nlsam.utils',
           'nlsam.stabilizer']
ext_modules = []
include_dirs = [numpy.get_include(), gsl_include]

for pyxfile in modules:

    ext_name = os.path.splitext(pyxfile)[0].replace('/', '.')
    source = os.path.join(*pyxfile.split('.')) + '.pyx'

    ext = Extension(pyxfile,
                    [source],
                    libraries=gsl_libraries,
                    library_dirs=[gsl_path],
                    include_dirs=include_dirs,
                    extra_objects=gsl_libraries_ext)

    ext_modules.append(ext)

install_requires = ['numpy>=1.15.4',
                    'scipy>=1.6',
                    'cython>=0.29',
                    'nibabel>=2.0',
                    'joblib>=0.14.1',
                    'autodmri>=0.2.1',
                    'spams-bin>=2.6.2',
                    'dipy>=0.11']

setup(name='nlsam',
      author='Samuel St-Jean',
      author_email='sstjean1@ualberta.ca',
      url='https://github.com/samuelstjean/nlsam',
      version='0.6.1',
      license='GPLv3',
      description='Implementation of "Non Local Spatial and Angular Matching : Enabling higher spatial resolution diffusion MRI datasets through adaptive denoising"',
      long_description=long_description,
      long_description_content_type='text/markdown',
      scripts=['scripts/nlsam_denoising'],
      install_requires=install_requires,
      include_dirs=[gsl_path],
      packages=find_packages(),
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules)
