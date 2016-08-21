# Non Local Spatial and Angular Matching (NLSAM) denoising

[release]: https://github.com/samuelstjean/nlsam/releases
[wiki]: https://github.com/samuelstjean/nlsam/wiki
[DOI]: http://dx.doi.org/doi:10.1016/j.media.2016.02.010
[URL]: http://www.sciencedirect.com/science/article/pii/S1361841516000335
[paper]: http://scil.dinf.usherbrooke.ca/wp-content/papers/stjean-etal-media16.pdf
[Anaconda]: https://www.continuum.io/downloads
<!-- [spams-windows]:https://github.com/samuelstjean/spams-python/releases/download/0.1/spams-python-v2.4-svn2013-06-24.win-amd64-py2.7.exe -->
[nlsam_data]:https://github.com/samuelstjean/nlsam_data
<!-- [vspy27]:https://www.microsoft.com/en-us/download/details.aspx?id=44266 -->

The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI.

## How to install

The easiest way is to go grab a [release][], in which case the downloaded zip file contains everything you need (no python installation required,
you can use it straight away without installing anything else).
After extracting the zip file, start a terminal/command line prompt (start button, then type cmd + enter on windows) and navigate to where you extracted the binaries.

Since the tools are command line only, double-clicking it will open and immediately close a dos-like window, hence the need for opening a command line prompt.

If you would like to look at the code and modify it, you can build it from source after grabbing some dependencies,
check out the [wiki][wiki] for detailled intructions about each platforms.

If you have a working python setup already, doing

```shell
pip install https://github.com/samuelstjean/nlsam/archive/master.zip --user --process-dependency-links
```

should give you everything you need.

You can also clone it locally and then build the files with

```shell
git clone https://github.com/samuelstjean/nlsam.git
cd nlsam
python setup.py build_ext -i
```

Don't forget to add the path where you cloned everything to your $PYTHONPATH.


You can also download the datasets used in the paper over [here][nlsam_data].

## Using the NLSAM algorithm

Once installed, there are two main scripts, the stabilization algorithm and the NLSAM algorithm itself.
The first one allows you to transform the data to Gaussian distributed signals if your dataset is Rician or Noncentral chi distributed.

A simple example call for the stabilization would be

```bash
stabilizer dwi.nii.gz dwi_stab.nii.gz 1 sigma.nii.gz -m mask.nii.gz --bvals bvals --bvecs bvecs
```

and for the NLSAM denoising

```bash
nlsam dwi_stab.nii.gz dwi_nlsam.nii.gz 5 bvals bvecs sigma.nii.gz -m mask.nii.gz
```

You can find a detailed usage example and assorted dataset to try out in the
[example](example) folder.
<!--
<a name="Dependencies"></a>
## Dependencies

You will need to have at least numpy, scipy, nibabel, dipy, cython, cython-gsl
and spams installed with python 2.7.
Fortunately, the setup.py will take care of installing everything you need.

+ On Debian/Ubuntu, you will need some development headers which can be installed with

```shell
sudo apt-get install build-essential libgsl0-dev python-dev libopenblas-dev libopenblas-base liblapack-dev
```

Of course feel free to use your favorite blas/lapack implementation (such as intel MKL),
but I got 5x faster runtimes out of openblas vs atlas for NLSAM just by switching libraries.

+ On Windows and Mac OSX, it will be easier to grab a python distribution which includes everything such as [Anaconda][].
Additionally, grab a build of spams for windows [here][spams-windows] if you don't want to build it.
+ Additionally, you might need [Visual C++ Compiler for python 2.7][vspy27]
if you encounter the *Unable to  find vsvarsall.bat* error when installing nlsam on Windows. -->

## Questions / Need help / Think this is great software?

If you need help or would like more information, don't hesitate to drop me a
line at firstname@isi.uu.nl, where of course firstname needs to be replaced with samuel.

## Reference

St-Jean, S., Coupé, P., & Descoteaux, M. (2016).
"[Non Local Spatial and Angular Matching : Enabling higher spatial resolution diffusion MRI datasets through adaptive denoising.][paper]"
Medical Image Analysis, 32(2016), 115–130. [DOI] [URL]

## License

As the main solver I use (spams) is GPL licensed and the stabilization script
uses the GNU GSL library, the NLSAM main codebase is also licensed under the
GPL v3, see the file LICENSE for more information.

If you would like to reuse parts of this work under another project/license,
feel free to drop me an email and I will gladly re-license the files you need
as MIT/BSD/whatever else.
