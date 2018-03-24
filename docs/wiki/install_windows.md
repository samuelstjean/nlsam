## Easy way - grab a binary release

+ To simply run the algorithm, the easiest way is definitely to download the windows binary from the [release](https://github.com/samuelstjean/nlsam/releases) section as you won't need to install python or anything else.
Just unzip the archive and start the program from a command line prompt, that's it!

+ If you want to try out a precompiled dev version from master/another branch, you can find automatic builds here https://ci.appveyor.com/project/samuelstjean/nlsam/build/artifacts. Unless you need a feature which is not yet released, I would however not advise to use this method since they are automatically made after each commit and are subject to frequent changes.

+ If you would like to study/modify the code, you will need a python distribution and a compiler as outlined below.

## Installing Visual C++ compiler for python 2.7

I suggest using python 2.7 as the installation is much easier on windows. You can install a lightweight version of [Visual C++ Compiler for python 2.7](https://www.microsoft.com/en-us/download/details.aspx?id=44266).

If you would like to install python 3, you will need to install the full visual studio appropriate for your version of python as explained [here](https://wiki.python.org/moin/WindowsCompilers).

## Installing python

You will need to get a python distribution and some other stuff. For starters, grabbing a complete distribution such as [Anaconda](https://www.continuum.io/downloads#_windows) is the easiest way to go as it comes with all the usual scientific packages.

## Installing NLSAM and dependencies

Go grab a [release archive](https://github.com/samuelstjean/nlsam/releases) and run at the root of the folder.

```shell
pip install https://github.com/samuelstjean/spams-python/releases/download/0.1/spams-2.4-cp27-none-win_amd64.whl
pip install -r requirements.txt --user --process-dependency-links .
```
(Be sure to include the __.__ at the end of the command)

This will download spams for python 2.7 and install the required dependencies. If you are on python 3, you will need to build spams yourself from the original source. Although it can be a bit complicated, the [original website](http://spams-devel.gforge.inria.fr/documentation.html) will walk you through the install process.

Now you can run the main script from your terminal, be sure to have a look at the [example](https://github.com/samuelstjean/nlsam/tree/master/example) for more information about the usage.

```shell
nlsam_denoising --help
```

You may also build and install the extension (e.g. you prefer to use a easy to update local git clone instead of installing stuff) with

```shell
pip install -e .
```