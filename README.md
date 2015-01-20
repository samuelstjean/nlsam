# Non Local Spatial and Angular Matching denoising
=====

The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI.

## How to install

#### 1.a. Windows and Mac : Get a python 2.7 distribution, which can be easily installed with http://continuum.io/downloads#all

#### 1.b. Linux (assuming a Debian/Ubuntu based distribution): Get python 2.7 and required dependencies :

```shell
sudo apt-get install python-numpy python-scipy python-pip libgsl0-dev
```
*P.S. Ca vous prend la gsl pour mac (mrtrix l'utilise aussi), ca s'installe comment?*

Get the GSL, either through your distribution package manager or by using this cmake version : git clone https://github.com/samuelstjean/gsl
http://www2.lawrence.edu/fast/GREGGJ/CMSC110/gsl/gsl.html

You will also need a compiler and required build tools, which would be
+ On Windows, Visual Studio
+ On Mac, XCode
+ On Linux, GCC and company : sudo apt-get install build-essential

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
