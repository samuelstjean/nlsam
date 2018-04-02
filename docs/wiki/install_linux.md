# Installation for Linux

For a linux installation, you should find everything you need in your package manager.
These are the gcc compilers, python headers and a blas/lapack implementation such as atlas/openblas intel mkl/etc.

### Debian/Ubuntu and derivatives

```shell
sudo apt-get install build-essential python-dev libopenblas-dev libopenblas-base liblapack-dev
```

### Red hat/Cent OS/Fedora and derivatives

```shell
sudo yum install python-devel atlas-devel blas-devel lapack-devel gcc gcc-c++ gcc-gfortran
```

The gsl lib is now included as a precompiled static library, so no need to install it anymore.

Of course feel free to use your favorite blas/lapack implementation (such as intel MKL),
but I got 5x faster runtimes out of openblas vs atlas for NLSAM just by switching libraries.

## Installing nlsam

For the python dependencies themselves, I recommend a fresh pip install since versions from the repositories tend to get old quite fast.
You will need numpy, scipy, cython, nibabel, dipy and spams.

Get a [release archive](https://github.com/samuelstjean/nlsam/releases) and install it directly from the downloaded file

```shell
pip install file_you_just_downloaded.tar.gz --user
```

and it should grab all the required dependencies if they are not already installed.
If you encounter some errors (e.g. spams needs numpy and blas/lapack headers), install the missing package with pip first and continue the installation afterwards.

Now you can run the main script from your terminal, be sure to have a look at the [example](https://github.com/samuelstjean/nlsam/tree/master/example) for more information about the usage.

```shell
nlsam_denoising --help
```

You may also build and install the package from a local git clone instead of installing stuff with

```shell
pip install -e .
```

After updating your local git copy, you can rebuild the cython files by running

```shell
python setup.py build_ext --inplace
```
