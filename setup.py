import numpy

from setuptools import setup, find_packages
from Cython.Build import cythonize

ext_modules = cythonize("nlsam/*.pyx")
include_dirs = [numpy.get_include()]

setup(
    packages=find_packages(),
    include_dirs=include_dirs,
    ext_modules=ext_modules)
