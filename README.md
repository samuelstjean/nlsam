# Non Local Spatial and Angular Matching denoising
=====

The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI.

## How to install

Go grab a [release](https://github.com/samuelstjean/nlsam/releases) (recommended) or build it from source.


## Dependencies

You will need to have at least numpy, scipy, nibabel, dipy, cython, cython-gsl and spams.
Fortunately, the setup.py will take care of installing everything you need for you.

If you have a working python setup already, doing

```shell
pip install git+https://github.com/samuelstjean/nlsam.git@dev --user
```
should give you everything you need.

If you get build error about missing headers on linux, you will also need some development headers for python, the gsl, blas/lapack/etc. like this

```shell
sudo apt-get install build-essential libgsl0-dev python-dev libblas-dev liblapack-dev
```


You can also just clone it locally and then build the files with

```shell
git clone https://github.com/samuelstjean/nlsam.git --branch dev
cd path/to/git/repo
python setup.py build_ext -i
```

Don't forget to add the path where you cloned everything to your PYTHONPATH.
<!---

#### 1.a. Windows and Mac : Get a python 2.7 distribution, which can be easily installed with http://continuum.io/downloads#all

#### 1.b. Linux (assuming a Debian/Ubuntu based distribution): Get python 2.7 and required dependencies :

```shell
sudo apt-get install python-numpy python-scipy python-pip libgsl0-dev
```

Get the GSL, either through your distribution package manager or by using this cmake version : git clone https://github.com/samuelstjean/gsl
Prebuilt windows gsl :
+ 1.15 http://code.google.com/p/oscats/downloads/list
+ 1.16 for VS 2013 https://azylstra.net/blog/content/gsl-1.16_winbin.zip

You will also need a compiler and required build tools, which would be
+ On Windows, Visual Studio http://www.visualstudio.com/en-us/products/visual-studio-community-vs
+ On Mac, XCode
+ On Ubuntu/Linux, GCC and company : sudo apt-get install build-essential

#### 2. Get some more dependencies with pip

```shell
pip install cython nibabel cythongsl
```

#### 3. Build the cython files.
*prendre ma branche de scilpy add_stabilizer_script à la place*
https://bitbucket.org/sciludes/scilpy/pull-request/104/stabilisation-script/diff

From the NLSAM root folder, run
```shell
python setup.py build_ext -i
cd nlsam/spams_third_party
python setup.py build_ext -i
```
-->


## Using the NLSAM algorithm

To be updated when scilpy stabilisation is back in

<!---
For now, get the stabilisation script from scilpy, https://bitbucket.org/sciludes/scilpy/pull-request/104/stabilisation-script or you can skip it if you don't have terribly noisy data. The nlsam subfolder has my old personal version, which might do weird imports.

Run the denoising itself, like this
```shell
nlsam noisy_data.nii.gz N_neighbors bval bvec -o denoised_data.nii.gz -mask_data mask.nii.gz
```
where N_neighbors is the number of angular neighbors in a block, I personnaly suggest 5. Afterward, go take a long coffee break/come back tomorrow. You should also have at least 12/16 go of ram for a large dataset (1.2mm at 41 DWIs takes approx 16go of ram).
-->

