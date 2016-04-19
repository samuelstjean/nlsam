# Non Local Spatial and Angular Matching (NLSAM) denoising
=====

[release]: https://github.com/samuelstjean/nlsam/releases
[DOI]: http://dx.doi.org/doi:10.1016/j.media.2016.02.010
[URL]: http://www.sciencedirect.com/science/article/pii/S1361841516000335
[paper]: http://scil.dinf.usherbrooke.ca/wp-content/papers/stjean-etal-media16.pdf
[Anaconda]: https://www.continuum.io/downloads
[spams-windows]:https://github.com/samuelstjean/spams-python/releases/download/0.1/spams-python-v2.4-svn2013-06-24.win-amd64-py2.7.exe
[nlsam_data]:https://github.com/samuelstjean/nlsam_data

The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI.

## How to install

Go grab a [release][] (recommended) or build it from source with the [instructions](#Dependencies).
You can also download the datasets used in the paper over [here][nlsam_data].

## Using the NLSAM algorithm

Once installed, there are two main scripts, the stabilization algorithm and the NLSAM algorithm itself.
The first one allows you to transform the data to Gaussian distributed signals if your dataset is Rician or Noncentral chi distributed.

A typical example call requires only a diffusion weighted dataset (dwi.nii.gz) and the number of coils from the acquisition (N=1),
but it is recommended to also have a brain mask (brain_mask.nii.gz) to greatly reduce computation time.


I computed the brain mask using FSL bet for this example, but anything giving you a binary segmentation mask will do fine as the computation
will only take place inside this mask.


I also supplied the bvals/bvecs pair since the default option is to use a spherical harmonics fit for initialization.

```shell
stabilizer dwi.nii.gz dwi_stab.nii.gz 1 sigma.nii.gz -m brain_mask.nii.gz --bvals bvals --bvecs bvecs
```

The stabilized output is dwi_stab.nii.gz and the estimated noise standard deviation is sigma.nii.gz.

More options are available by using stabilizer --help.
Once your data is Gaussian distributed, the nlsam denoising itself can now be used with the outputs from the previous algorithm.
Here the number of angular neighbors is set to 5, which is the number of DWI which are equidistant in q-space to each volume in this example dataset.

```shell
nlsam dwi_stab.nii.gz dwi_nlsam.nii.gz 5 bvals bvecs sigma.nii.gz --mask brain_mask.nii.gz
```

The final nlsam denoised output is then dwi_nlsam.nii.gz.


Once again, nlsam --help will give you more options to be used beyond the defaults.

<a name="Dependencies"></a>
## Dependencies

You will need to have at least numpy, scipy, nibabel, dipy, cython, cython-gsl and spams installed with python 2.7.
Fortunately, the setup.py will take care of installing everything you need.

+ On Debian/Ubuntu, you will need some development headers which can be installed with
```shell
sudo apt-get install build-essential libgsl0-dev python-dev libopenblas-dev libopenblas-base liblapack-dev
```
Of course feel free to use your favorite blas/lapack implementation (such as intel MKL), but I got 5x faster runtimes out of openblas vs atlas for nlsam just by switching for example.

+ On Windows and Mac OSX, it will be easier to grab a python distribution which includes everything such as [Anaconda][].
Additionally, grab a build of spams for windows [here][spams-windows] if you don't want to build it.

If you have a working python setup already, doing

```shell
pip install git+https://github.com/samuelstjean/nlsam.git --user --process-dependency-links
```

should give you everything you need.

You can also clone it locally and then build the files with

```shell
git clone https://github.com/samuelstjean/nlsam.git
cd path/to/git/repo
python setup.py build_ext -i
```

Don't forget to add the path where you cloned everything to your PYTHONPATH.

## Reference
St-Jean, S., P. Coupé, and M. Descoteaux.
"[Non Local Spatial and Angular Matching : Enabling higher spatial resolution diffusion MRI datasets through adaptive denoising.][paper]"
Medical Image Analysis, 2016. [DOI] [URL]

## License
As the main solver I use (spams) is GPL licensed and the stabilization script uses the GNU GSL library,
the nlsam main codebase is also licensed under the GPL v3, see the file LICENSE for more information.


If you would like to reuse parts of this work under another project/license,
feel free to drop me an email and I will gladly re-license the files you need
as MIT/BSD/whatever else.
