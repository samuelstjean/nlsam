# Installation for Mac OSX

You have two options

+ Easy way - Use the binary release for Mac, no installation needed and perfect for trying it out quickly on your data, see https://github.com/samuelstjean/nlsam/releases

+ Build from source with the instructions below. These steps will walk you through getting [Anaconda](https://www.continuum.io/downloads#_macosx) and a few other things, but will let you easily use a non-released version and other fancy features.

## Python prerequisite

+ While some Mac Os versions come with their own python, it seems [discouraged to use it](https://github.com/MacPython/wiki/wiki/Which-Python) and people suggest to use another distribution. For example, [Anaconda](https://www.continuum.io/downloads#_macosx) will get you an easy way to install it.

+ You will also need a compiler (which might already be installed). If not, people suggest getting [XCode](https://developer.apple.com/xcode/download/) from the app store.

## Spams and openmp support

### A. The lazy way, using conda forge channel

The old installation instruction would have you install a non openmp version, but instead it is much simpler to use an already built one after all.
This can be done by adding the conda forge channel, which has a working openmp build of spams.

~~~bash
conda config --add channels conda-forge
conda install python-spams
~~~

Of course, while it is easy, it also has a downside.
Adding the conda forge channel will also replace all of your existing conda package with their own version which is using openblas instead of apple veclib.
This might not be a problem in general, but you might need to rebuild/compile other python packages installed prior to adding this channel.

### B. The slower way, building spams *without* openmp support

As the apple version of clang does not support openmp, we would need to install another compiler.
The old instructions would have you do `brew install llvm` and a few compiler  paths shenanigans, but it is much easier to use Anaconda and gcc.
It is also possible to install spams without openmp, but some parts will not be multithreaded in that case, making the overall runtime of the algorithm longer.

Building spams with openmp deactivated can be done in one line using

~~~bash
pip install https://github.com/samuelstjean/spams-python/releases/download/0.1/spams-2.6.zip
~~~

which downloads an archive hosted by yours truly.
You can also go grab a newer version on the original authors [website](http://spams-devel.gforge.inria.fr/downloads.html) if needed/available or if you want to build it with openmp support.

## Installing NLSAM

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
