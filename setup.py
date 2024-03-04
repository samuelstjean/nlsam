import numpy

from setuptools import setup, find_packages
from Cython.Build import cythonize

ext_modules = cythonize("nlsam/*.pyx")
include_dirs = [numpy.get_include()]

setup(
    include_dirs=include_dirs,
    packages=find_packages(),
    ext_modules=ext_modules)
