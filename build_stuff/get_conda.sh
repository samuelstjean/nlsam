#!/usr/bin/env bash

# Install Anaconda
if [[ "$ANACONDA_PYTHON_VERSION" == "2.7" ]]; then
    export CONDA_VERSION=2;
  else
    export CONDA_VERSION=3;
fi

if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    export CONDA_OS=MacOSX;
  else
    export CONDA_OS=Linux;
fi

wget https://repo.continuum.io/miniconda/Miniconda${CONDA_VERSION}-latest-${CONDA_OS}-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda create -q -n test-environment python=$ANACONDA_PYTHON_VERSION
source activate test-environment
