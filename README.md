# Non Local Spatial and Angular Matching denoising
=====

The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI.

+ Term of use : yes, there are a lot of prints, but they are useful.
+ There are two bottlenecks I am aware of, the neighboring functions and the cutting into blocks functions.
+ Don't look at the code, it's horrible
+ Sorry Elef, I use gpl dependencies (but I'm open to using **working** replacements).

## How to install

#### 1.a. Windows and Mac : Get a python 2.7 distribution, which can be easily installed with http://continuum.io/downloads#all

#### 1.b. Linux (assuming a Debian/Ubuntu based distribution): Get python 2.7 and required dependencies :

```shell
sudo apt-get install python-numpy python-scipy python-pip libgsl0-dev
```
*P.S. Ca vous prend la gsl pour mac (mrtrix l'utilise aussi), ca s'installe comment?*

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
pip install cython nibabel cythongsl scikit-learn
```

#### 3. Build the cython files.
From the NLSAM root folder, run
```shell
python setup.py build_ext -i
cd nlsam/spams_third_party
python setup.py build_ext -i
```

This will build the NLSAM cython files and also the spams library, but feel free to use your own local version of spams as needed.

#### 4. Check everything went well
Run the tests


## Using the NLSAM algorithm
For now, get the stabilisation script from scilpy, https://bitbucket.org/sciludes/scilpy/pull-request/104/stabilisation-script or you can skip it if you don't have terribly noisy data. The script subfolder has my old personal version, which might do weird imports. 

Run the denoising itself, like this
```shell
caller.py noisy_data.nii.gz N_neighbors bval bvec -sigma sigma.nii.gz -o denoised_data.nii.gz -mask_data mask.nii.gz
```
where N_neighbors is the number of angular neighbors in a block, I personnaly suggest 5. Afterward, go take a long coffee break/come back tomorrow. You should also have at least 12/16 go of ram for a large dataset (1.2mm at 41 DWIs takes approx 16go of ram).




## Old run instructions
-----------------------------
Go to the script subfolder and run
```shell
stabilizer.py input.nii.gz -N N -o output
```

Where N is the number of coils and output the output prefix filename (it's appended with _stabilized.nii.gz automatically)

Now use the stabilized version as the input for the denoising script

stabilizer.py output_stabilized.nii.gz number_of_angular_neighbors bvals bvecs

*5 a l'air de bien marche comme N voisins angulaires, c'est a peu pres le nombre que l'on retrouve
a egale distance d'un point donne sur des shell bien distribuees, j'ai un script pour compter
une tonne de metriques pour aider a choisir selon le degre max/degre moyen etc., dont un de visualisation pour
voir rapidement des points sur la sphere*

nlsam_gui.py __wishlist__
