# Non Local Spatial and Angular Matching (NLSAM) denoising

[release]: https://github.com/samuelstjean/nlsam/releases
[DOI]: http://dx.doi.org/doi:10.1016/j.media.2016.02.010
[URL]: http://www.sciencedirect.com/science/article/pii/S1361841516000335
[paper]: https://arxiv.org/pdf/1606.07239.pdf
[autodmri_paper]: https://arxiv.org/abs/1805.12071
[nlsam_data]: https://github.com/samuelstjean/nlsam_data
[spams]: http://spams-devel.gforge.inria.fr/
[rtd]: https://nlsam.readthedocs.io/en/latest/
[koay_bias]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2765718/
[docker]: https://hub.docker.com/search?type=edition&offering=community

The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI.

## Quick links

+ [Source downloads + precompiled binaries](https://github.com/samuelstjean/nlsam/releases)
+ [Example + Usage guide](example/README.md)

You can find the latest documentation and installation instructions over [here](http://nlsam.readthedocs.io/en/latest) with a downloadable version of the documentation [here](https://readthedocs.org/projects/nlsam/downloads).

## How to install

The new easy way is to use the ```Dockerfile```. for which you'll need to install [docker][] (probably already available in your favorite linux distribution).

The old easiest way is to go grab a [release][], in which case the downloaded zip file contains everything you need (no python installation required,
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

You can also download the datasets used in the paper over [here][nlsam_data].

## Using the NLSAM algorithm

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

## References

The NLSAM denoising algorithm itself is detailed in

> St-Jean, S., Coupé, P., & Descoteaux, M. (2016).
> "[Non Local Spatial and Angular Matching : Enabling higher spatial resolution diffusion MRI datasets through adaptive denoising][paper]"
> Medical Image Analysis, 32(2016), 115–130. [DOI] [URL]

The bias correction framework is a reimplementation of

> Koay, CG, Özarslan, E and Basser, PJ
> [A signal transformational framework for breaking the noise floor and its applications in MRI][koay_bias],
> Journal of Magnetic Resonance, Volume 197, Issue 2, 2009

The automatic estimation of the noise distribution is computed with

> St-Jean, S., , De Luca A., Viergever M. A., Leemans A. (2018)
> "[Automatic, fast and robust characterization of noise distributions for diffusion MRI][autodmri_paper]"
> MICCAI 2018. Lecture Notes in Computer Science, vol 11070. Springer, Cham [DOI](https://dx.doi.org/doi:10.1007/978-3-030-00928-1_35) [URL](https://link.springer.com/chapter/10.1007%2F978-3-030-00928-1_35)

And here is a premade bibtex entry.

    @article{St-Jean2016a,
      author = {St-Jean, Samuel and Coup{\'{e}}, Pierrick and Descoteaux, Maxime},
      doi = {10.1016/j.media.2016.02.010},
      journal = {Medical Image Analysis},
      pages = {115--130},
      title = {{Non Local Spatial and Angular Matching : Enabling higher spatial resolution diffusion MRI datasets through adaptive denoising}},
      volume = {32},
      year = {2016}
      }
