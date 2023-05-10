import numpy

from setuptools import setup, find_packages
from Cython.Build import cythonize

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

ext_modules = cythonize("nlsam/*.pyx")
include_dirs = [numpy.get_include()]

install_requires = ['numpy>=1.15.4',
                    'scipy>=1.5',
                    'cython>=0.29',
                    'nibabel>=2.0',
                    'joblib>=0.14.1',
                    'autodmri>=0.2.1',
                    'spams-bin>=2.6.2',
                    'dipy>=0.11']

setup(name='nlsam',
      author='Samuel St-Jean',
      url='https://github.com/samuelstjean/nlsam',
      version='0.6.1',
      license='GPLv3',
      description='Implementation of "Non Local Spatial and Angular Matching : Enabling higher spatial resolution diffusion MRI datasets through adaptive denoising"',
      long_description=long_description,
      long_description_content_type='text/markdown',
      scripts=['scripts/nlsam_denoising'],
      install_requires=install_requires,
      packages=find_packages(),
      include_dirs=include_dirs,
      ext_modules=ext_modules)
