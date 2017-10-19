Denoising a small example dataset
----------------------------------

This tutorial shows how to denoise a crop of the 1.2 mm dataset
[(Full dataset also available)](https://github.com/samuelstjean/nlsam_data).
I also assume you have installed NLSAM and everything is running fine beforehand.

This example will walk you through the required step to go from a noisy image like this

![](images/noisy.png)

to the final denoised result like this

![](images/nlsam.png)

We also showcase some advanced options for highly noisy datasets, but for your regular
everyday processing the default options should work just fine.

# Table of Contents
1. [Prerequisite](#prerequisite)
2. [Processing steps](#steps)
3. [The result](#result)
4. [Using the python API](#api)

<a name="prerequisite"></a>
## 1. Prerequisite

#### 1.1 Get a binary mask of the brain

This will reduce the computation time by only denoising the voxels which are included inside the mask.
For this example, I used bet2 from fsl to create mask.nii.gz using only the b0 image, but feel free to use your favorite tool of course.

```bash
bet2 b0.nii.gz brain -m -f 0.1
```

#### 1.2 Minimally required data for processing

You will need a set of diffusion weighted images and the associated bvals/bvecs files as used by FSL.

If your data is not in the bvals/bvecs format, you will first need to convert it.
Your favorite diffusion MRI processing tool probably has a function to help you with the conversion
(Scilpy, MRtrix3 and ExploreDTI all offer options for this conversion for example).

#### 1.3 Required command line inputs

A typical example call requires only a diffusion weighted dataset (dwi.nii.gz), the bvals/bvecs file
and the number of coils from the acquisition (N),
but it is recommended to also have a brain mask (mask.nii.gz) to greatly reduce computation time.

For this example dataset, we used a SENSE reconstruction (Philips, GE), which leads to spatially varying Rician noise, so we set N = 1.
If your scanner instead implements a GRAPPA reconstruction (Siemens), you would need to specify N as the number of coils in the acquisition.
While this value can be difficult to estimate, asking help from your friendly MR physicist is advised (or check subsection 2.1a).

In the meantime, you can still run the algorithm with N = 1 to use a Rician correction and check the result, in which case there would be a slight intensity
bias left in the image, but which would be lower than not correcting it in the first place.

<a name="steps"></a>
## 2. Processing steps

#### 2.1 Correcting the noise bias

Once installed, the first processing step allows you to transform the data to Gaussian distributed
signals if your dataset is Rician or Noncentral chi distributed.

Of course if your dataset is already bias corrected or you would like to use another method for noise estimation,
you can skip this step and proceed to the denoising itself by passing the option **--no_stabilization**.
The correction for Rician or Noncentral chi distributed noise would then be left to any other method of your choosing.

#### 2.1a Advanced techniques for estimating N (optional topic)

See the [wiki](https://github.com/samuelstjean/nlsam/wiki/Advanced-noise-estimation)
for a discussion on the subject.

#### 2.2 Algorithms for noise estimation

To initialize the estimation for the stabilization algorithm, we will use a spherical harmonics fit (which is the default),
to remove extreme/implausible signals. In case you have few directions (per shell), you can deactivate this option
by passing **--sh_order 0** or lowering the order if your data does not have enough dwi volumes (the script will warn you in that case).

The default is to use a noise estimation based on piesno, but since this dataset is fairly noisy and has no background,
we will instead use an estimation based on the local standard deviation with the option **--noise_est local_std**

If you data is really noisy and the S0 signal is low, you might want to use the option
**--fix_implausible** which will ensure that the b0 image always has the highest value through the volume.
This option was implicitly used in NLSAM versions before 0.5 and now need to be activated if needed.

#### 2.3 Required command line inputs

There are 6 required command line inputs (their order is important) which are

+ The input dataset (dwi.nii.gz)
+ The output dataset (dwi_nlsam.nii.gz)
+ The effective number of coils for our acquisition (see section 2.1a)
+ The b-values file for our input dataset (bvals)
+ The b-vectors file for our input dataset (bvecs)
+ The number of angular neighbors (N)

The bvals/bvecs files are needed for identifying
the angular neighbors and we need to choose how many we want to denoise at once.

Here I selected 5 as it is the number of dwis which are roughly equidistant on the sphere.
Using a larger number could mean more blurring if we mix q-space points which are too far part.

For a multishell acquisition, only the direction (as opposed to the norm)
of the b-vector is taken into account, so you can freely mix dwi from different
shells to favor picking radial decay in the denoising.

#### 2.4 Advanced options

More options are available for various advanced usage, which can be viewed with **nlsam_denoising --help**.
Some are mostly useful for debugging and saving intermediate steps, such as **--verbose** for
printing various useful information or **--log logfile.txt** for processing a large number of file
and saving the various outputs.

Feel free to check them out if you want finer grained controls over the denoising process.

#### 2.5 Example call to the nlsam_denoising script

Finally, the required call to the nlsam_denoising script for this example would be

```bash
nlsam_denoising dwi.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask.nii.gz --noise_est local_std --fix_implausible  --verbose
```

The script will output the time taken at each denoising iterations so you can ballpark estimate the total time required.
On my current computer (an intel xeon quad core at 3.5 GHz, much faster than the one reported originally in the paper),
it took around 40 s per iteration for this example, for a total of 8 minutes.

The full dataset required 23 mins per iteration, for a total processing time of 276 mins.
As a side note, using piesno for the noise estimation (which is the default) resulted in an iteration taking only 8 mins,
so the whole dataset could be denoised in around ~100 mins with the default options.

<a name="result"></a>
## 3. The result

At the end, you can continue your regular diffusion MRI pipeline with the denoised version of the dataset,
here named dwi_nlsam.nii.gz for the purposes of this example.

From an input noisy image

![](images/noisy.png)

This is the final, NLSAM denoised result

![](images/nlsam.png)

<a name="api"></a>
## 4. Using the python API

This example went through the classical command line interface nlsam_denoising, which is actually
a fancy script which set up stuff for us. Here is the same example, but using the python API.

For those wanting to extend the functionality of the algorithm or embed it in their python workflow,
I suggest having a look at the [script itself](../scripts/nlsam_denoising) as it contains much more options.

~~~python
from __future__ import division, print_function

import nibabel as nib
import numpy as np

from multiprocessing import cpu_count

from nlsam.denoiser import nlsam_denoise
from nlsam.smoothing import sh_smooth, local_standard_deviation
from nlsam.stabilizer import stabilization, corrected_sigma

from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

# Set up the options for nlsam
input_data = 'dwi.nii.gz'
output_data = 'dwi_nlsam.nii.gz'
mask_data = 'mask.nii.gz'
bvals_file = 'bvals'
bvecs_file = 'bvecs'

N = 1
block_size = (3, 3, 3, 5)
b0_threshold = 10
n_cores = cpu_count()
subsample = True
is_symmetric = False
n_iter = 10
sh_order = 8

# Load up the data
vol = nib.load(input_data)
data = np.asarray(vol.get_data(caching='unchanged'), dtype=np.float32)
affine = vol.affine
header = vol.header
header.set_data_dtype(np.float32)

mask = np.asarray(nib.load(mask_data).get_data(caching='unchanged'))
bvals, bvecs = read_bvals_bvecs(bvals_file, bvecs_file)
gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)

# Fix the implausible signals, note this 'option' is only in the script and not a function.
# I could make one if there is enough demand I guess, but anyway, it's a one liner as you can see.
data[..., gtab.b0s_mask] = np.max(data, axis=-1, keepdims=True)

#########################
#  Noise estimation part
#########################

sigma = local_standard_deviation(data, n_cores=n_cores)
sigma = np.broadcast_to(sigma[..., None], data.shape)
mask_4D = np.broadcast_to(mask[..., None], data.shape)
sigma = corrected_sigma(data, sigma, mask_4D, N, n_cores=n_cores)

##################
#  Stabilizer part
##################

m_hat = sh_smooth(data, gtab, sh_order=sh_order).clip(min=0)
data_stabilized = stabilization(data, m_hat, mask, sigma, N, n_cores=n_cores)

##################
#  Denoising part
##################

# We need a 3D sigma map down for later, but we just estimated a 4D one as we did
# a voxelwise correction, so we just pick the median along the 4th dimension.
sigma = np.median(sigma, axis=-1)

data_denoised = nlsam_denoise(data_stabilized, sigma, bvals, bvecs, block_size,
                              mask=mask,
                              is_symmetric=is_symmetric,
                              n_cores=n_cores,
                              subsample=subsample,
                              n_iter=n_iter,
                              b0_threshold=b0_threshold)

nib.save(nib.Nifti1Image(data_denoised.astype(np.float32), affine, header), output_data)
~~~

As you can see the bulk of the work is loading the data and making sure everything plays nicely regarding options or dimensions,
so feel free to adapt your workflow based on the main nlsam_denoising processing script.
