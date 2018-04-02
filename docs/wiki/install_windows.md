# Installation for Windows

## Easy way - grab a binary release

+ To simply run the algorithm, the easiest way is definitely to download the windows binary from the [release](https://github.com/samuelstjean/nlsam/releases) section as you won't need to install python or anything else.
Just unzip the archive and start the program from a command line prompt, that's it!

+ If you want to try out a precompiled dev version from master/another branch, you can find automatic builds [here](https://ci.appveyor.com/project/samuelstjean/nlsam/build/artifacts).
Unless you need a fancy feature which is not yet released, I would stick to the released version for simplicity as the automatic builds are subject to frequent changes.

+ If you would like to study/modify the code, you will need a python distribution and a compiler as outlined below.

## Installing Visual C++ compiler for python 2.7

I suggest using python 2.7 as the installation is much easier on windows. You can install a lightweight version of [Visual C++ Compiler for python 2.7](https://www.microsoft.com/en-us/download/details.aspx?id=44266).

If you would like to install python 3, you will need to install the full visual studio appropriate for your version of python as explained [here](https://wiki.python.org/moin/WindowsCompilers).
As the whole thing is at least around 20 GB, I would strongly recommend to stick with the easy python 2.7 version for now.

## Installing python and dependencies

You will need to get a python distribution and some other stuff.
For starters, grabbing a complete distribution such as [Anaconda](https://www.continuum.io/downloads#_windows) is the easiest way to go as it comes with all the usual scientific packages.

From an anaconda terminal, we can also install a prebuilt version of spams for windows 2.7

```shell
pip install https://github.com/samuelstjean/spams-python/releases/download/0.1/spams-2.4-cp27-none-win_amd64.whl
```

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
