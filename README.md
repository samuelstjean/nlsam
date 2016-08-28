# Non Local Spatial and Angular Matching (NLSAM) denoising

[release]: https://github.com/samuelstjean/nlsam/releases
[wiki]: https://github.com/samuelstjean/nlsam/wiki
[DOI]: http://dx.doi.org/doi:10.1016/j.media.2016.02.010
[URL]: http://www.sciencedirect.com/science/article/pii/S1361841516000335
[paper]: http://scil.dinf.usherbrooke.ca/wp-content/papers/stjean-etal-media16.pdf
[nlsam_data]:https://github.com/samuelstjean/nlsam_data
[spams]: http://spams-devel.gforge.inria.fr/

The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI.

## How to install

The easiest way is to go grab a [release][], in which case the downloaded zip file contains everything you need (no python installation required,
you can use it straight away without installing anything else).
After extracting the zip file, start a terminal/command line prompt (start button, then type cmd + enter on windows) and navigate to where you extracted the binaries.

Since the tools are command line only, double-clicking it will open and immediately close a dos-like window, hence the need for opening a command line prompt.

If you have a working python setup already, the next command should give you everything you need.

```shell
pip install https://github.com/samuelstjean/nlsam/archive/master.zip --user --process-dependency-links
```

If you would like to look at the code and modify it, you can also clone it locally
and then install everything through pip after grabbing some dependencies

```shell
git clone https://github.com/samuelstjean/nlsam.git
pip install -e nlsam
```

Check out the [wiki][wiki] for detailed instructions about each platforms.

+ [Windows](https://github.com/samuelstjean/nlsam/wiki/Installation-for-Windows)
+ [Mac OSX](https://github.com/samuelstjean/nlsam/wiki/Installation-for-Mac-OSX)
+ [Linux](https://github.com/samuelstjean/nlsam/wiki/Installation-for-linux)

You can also download the datasets used in the paper over [here][nlsam_data].

## Using the NLSAM algorithm

Once installed, there is now a single script to do the whole processing.
Feel free to have a look if you want to build your own python pipeline as it
wraps the various parts of the algorithm provided inside the python-part library.

The process is to first transform your data to Gaussian distributed signals if your dataset is
Rician or Noncentral chi distributed and then proceed to the NLSAM denoising part itself.

A quickstart example call would be

```bash
nlsam_denoising dwi.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask.nii.gz
```

For more fine grained control and explanation of arguments,
have a look at the possible command line options with nlsam_denoising --help

You can find a detailed usage example and assorted dataset to try out in the
[example](example) folder.

## Questions / Need help / Think this is great software?

If you need help or would like more information, don't hesitate to drop me a
line at firstname@isi.uu.nl, where of course firstname needs to be replaced with samuel.

## Reference

St-Jean, S., Coupé, P., & Descoteaux, M. (2016).
"[Non Local Spatial and Angular Matching : Enabling higher spatial resolution diffusion MRI datasets through adaptive denoising.][paper]"
Medical Image Analysis, 32(2016), 115–130. [DOI] [URL]

## License

As the main solver I use [spams][] is GPL licensed and the stabilization script
uses the GNU GSL library, the NLSAM main codebase is also licensed under the
GPL v3, see the file LICENSE for more information.

If you would like to reuse parts of this work under another project/license,
feel free to drop me an email and I will gladly re-license the files you need
as MIT/BSD/whatever else.
