# Instruction for installing on Linux

For a linux installation, you should find everything you need in your package manager. These are the gcc compilers, the GSL lib development headers, python headers and a blas/lapack implementation such as atlas/openblas intel mkl/etc.

### Debian/Ubuntu and derivatives

```shell
sudo apt-get install build-essential libgsl0-dev python-dev libopenblas-dev libopenblas-base liblapack-dev
```

### Red hat/CentOS/Fedora and derivatives

```shell
sudo yum install python-devel atlas-devel blas-devel lapack-devel gsl-devel gcc gcc-c++ gcc-gfortran
```

If you are using an older distribution (such as CentOS 6), you will want to grab a more recent gsl version such as 1.15 from over [here](ftp://ftp.gnu.org/gnu/gsl/) and compile it yourself.

Of course feel free to use your favorite blas/lapack implementation (such as intel MKL),
but I got 5x faster runtimes out of openblas vs atlas for NLSAM just by switching libraries.

## Installing the python dependencies

For the python dependencies themselves, I recommend a fresh pip install since versions from the repositories tend to get old quite fast. You will need numpy, scipy, cython, cython-gsl, nibabel, dipy and spams.

Get a [release archive](https://github.com/samuelstjean/nlsam/releases) and in the folder where you unzipped the nlsam archive

```shell
pip install -r requirements.txt --user --process-dependency-links
```

will grab all the required dependencies. If you encounter some errors (e.g. cython needs to be installed for cython-gsl to install itself, spams needs numpy), simply install the missing dependent package with pip first and continue the installation afterwards.

## Installing nlsam itself

Run at the root of the folder.

```shell
python setup.py install --user
```

Now you can run the main script from your terminal if ~/.local/bin/ is in your $PATH.

You may also build the extension (e.g. you prefer a local git clone instead of installing stuff) and add the library to your $PYTHONPATH.

```shell
python setup.py build_ext --inplace
```