#!/usr/bin/env python

import os
from os.path import splitext, join, exists, split

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if exists('MANIFEST'): os.remove('MANIFEST')

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


import numpy
try:
    from nibabel.optpkg import optional_package
except:
    raise ValueError('Could not find nibabel, which is required for building. Try running\n pip install nibabel')

cython_gsl, have_cython_gsl, _ = optional_package("cython_gsl")

if not have_cython_gsl:
    raise ValueError('cannot find gsl package (required for hyp1f1), \
        try pip install cythongsl and sudo apt-get install libgsl0-dev libgsl0ldbl')

params = {}
params['name'] = 'nlsam'
params['version'] = '0.1'
params['requires'] = ['cythongsl>=0.2',
                      'numpy>=1.10',
                      'cython>=0.21']
params['deps'] = ['dipy>=0.11',
                  'scipy>=0.12',
                  'nibabel>=1.3',
                  'spams>=2.4']
params['links'] = ['https://github.com/samuelstjean/spams-python/archive/master.zip#egg=spams-2.5']

ext_modules = []
modlist = ['nlsam.utils',
           'nlsam.stabilizer']

for pyxfile in modlist:

    ext_name = splitext(pyxfile)[0].replace('/', '.')
    source = join(*pyxfile.split('.')) + '.pyx'
    # pyxfile = split(pyxfile)[1]
    print(ext_name, source)
    ext = Extension(pyxfile,
                    [source],
                    libraries=cython_gsl.get_libraries(),
                    library_dirs=[cython_gsl.get_library_dir()],
                    cython_include_dirs=[cython_gsl.get_cython_include_dir()],
                    include_dirs=[numpy.get_include()])

    ext_modules.append(ext)


    # pyx_src = pjoin(*modulename.split('.')) + '.pyx'
    # EXTS.append(Extension(modulename, [pyx_src] + other_sources,
    #                       language=language,
    #                       **deepcopy(ext_kwargs)))  # deepcopy lists

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
    scripts=glob(join('scripts', '*')),
)
