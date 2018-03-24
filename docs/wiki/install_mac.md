# Installation for Mac OSX

You have two options

+ Easy way - Use the binary release for Mac, no installation needed and perfect for trying it out quickly on your data. https://github.com/samuelstjean/nlsam/releases

+ Build from source with the instructions below. These steps will walk you through getting either [Anaconda](https://www.continuum.io/downloads#_macosx) or [Homebrew](http://brew.sh/), python and a few other things, but will let you easily use a non-released version and other fancy features. Do note that the anaconda way is faster as using brew requires you to build some stuff, which takes around 45 minutes.

## Python prerequisite

+ While some Mac Os versions come with their own python, it seems [discouraged to use it](https://github.com/MacPython/wiki/wiki/Which-Python) and people suggest to use another distribution. For example, [Anaconda](https://www.continuum.io/downloads#_macosx) will get you an easy way to install it.

+ You will also need a compiler (which might already be installed). If not, people suggest getting [XCode](https://developer.apple.com/xcode/download/) from the app store. Installing [Homebrew](http://brew.sh/) will also install the command line version of XCode at the same time, so you might just skip directly to the next step as it seems to be sufficient.

## Installing the GNU Scientific Library (GSL)

+ The easiest way is to install it using either [Homebrew](http://brew.sh/) or [Anaconda](https://www.continuum.io/downloads#_macosx) by running

~~~bash
conda install gsl
~~~
or
~~~bash
brew install gsl
~~~

## Spams and openmp support

### A. The easy way, using wheel files without openmp

For convenience, I built a version of spams for which I have _deactivated_ openmp support on Mac OSX as it is not supported out of the box. However, it will most likely result in a slower overall version of spams and is therefore recommended to build it yourself to get the fastest version possible.

+ For python 2.7
~~~bash
pip install https://github.com/samuelstjean/spams-python/releases/download/0.1/spams-2.5-cp27-cp27m-macosx_10_6_x86_64.whl
~~~

+ For python 3.5
~~~bash
pip install https://github.com/samuelstjean/spams-python/releases/download/0.1/spams-2.5-cp35-cp35m-macosx_10_6_x86_64.whl
~~~

This is the same version as if you would build it as downloaded from the requirements.txt with pip.

While it seems strange to remove openmp support, it is not supported out of the box by Apple's clang and you need to use another compiler for that. As NLSAM is using python multiprocessing instead, it is thus not directly needed and deactivated for simplicity.

### B. The proper way, building spams with openmp support

So, if you would like to use spams in another project which might take advantage of its openmp multi-threading capability, be sure to install the [original version](http://spams-devel.gforge.inria.fr/downloads.html) along with an openmp enabled compiler.

As of October 2016, clang 3.8 does support openmp, so it could be used if your Mac OSX version comes with it already. If it is not the case, you can install an openmp enabled version of gcc through anaconda or homebrew

~~~bash
conda install gcc
~~~
or
~~~bash
brew install gcc --without-multilib
~~~

Do note that compiling gcc when using homebrew takes around 45 minutes, so I suggest to go for anaconda or the easy way if you don't feel confortable playing around in the terminal.

Now you can install spams with openmp support. Download this archive https://github.com/samuelstjean/spams-python/releases/download/0.1/spams-2.6.zip and edit the setup.py, commenting line 19 to 21 at the top in order to reactivate openmp support

~~~python
#if osname.startswith("macosx"):
#    cc_flags = ['-fPIC','-w'] # I deactivated -fopenmp as it does not work on clang by default
#    link_flags = ['']
~~~

And now it can finally be installed by running inside the spams folder

~~~bash
python setup.py install
~~~

## Heads up - is your installation much slower than you'd expect?

It seems that sometimes, depending on how you install it, numpy links to a really slow blas library instead of the system one, presumably when installed from older wheel builds.

If you suspect this is the case (as in 4-10x slower than on another comparable computer), try out the example dataset and the binary build (see the very first point at the top of this page) against your current install, which should use the highly optimized system BLAS.

If it runs much faster, then your current installation of numpy might be using a slow(er) BLAS implementation, in which case upgrading to a newer version might help solve the problem. This also seems to only be valid for OSX 10.9 and later though (according to internet).

## Installing NLSAM

Get a [release archive](https://github.com/samuelstjean/nlsam/releases) and run at the root of the folder.

```shell
pip install -r requirements.txt --user --process-dependency-links
python setup.py install --user
```

This will download and build the various dependencies you might not already have installed.

## Running NLSAM

Now you can run the main nlsam_denoising script from your terminal, see the [example](https://github.com/samuelstjean/nlsam/tree/master/example) for guidelines.

You may also build the extension (e.g. you prefer a local git clone instead of installing stuff) and add the library to your $PYTHONPATH.

```shell
python setup.py build_ext --inplace
```