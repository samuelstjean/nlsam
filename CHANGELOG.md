# Changelog

## [0.8]

- Removed `PIESNO` as a noise estimation method. Use `auto` instead, which is the new default since 0.7, as it will automatically estimate `N` for you.
- Support for scipy 1.17 and up
- Use multithreading instead of loky for the frozen binaries to prevent issues and hangups with pyinstaller

## [0.7.2] - 2024-07-25

- Support for numpy 2.0 and python 3.9 and up
- Fixes for cython 3 and newer scipy

## [0.7.1] - 2023-07-03

- Some speed improvements internally
- Some more functions in parallel
- A new progress bar with tqdm
- New non-frozen builds for the standalone versions
- Mac M1/M2 arm64 binary wheels now available

## [0.7] - 2023-05-20

- **Breaking changes in the command line parser**
    - The previously required options __N__ and __angular_block_size__ are now optional.
    - A mask is now required to be passed with __-m__ or __--mask__ to only sample data. It was previously possible to be unlucky and only sample background noise in the reconstruction process, taking forever to practically do nothing in practice, passing a mask with only the data to sample and reconstruct should prevent this issue.

    - A new call would now looks like

    ~~~bash
    nlsam_denoising input output bvals bvecs -m mask.nii.gz
    ~~~

- New command line arguments, now subclassed into categories.
    + __--load_mhat__ file, to load a volume for initializing the bias correction, the default is to use the data itself.
    + __--save_difference__ file, to save a volume showing the removed signal parts as abs(original_data - denoised_data)
    + __--save_eta__ file, to save the result of the estimated underlying signal value for debugging purposes.
    + Deprecation of options __--implausible_signal_fix__ and __--sh_order__, use __--load_mhat__ instead for initialization.

- Support for non-integer values of N.
- Support for supplying a volume to be loaded as values of N.
- New module nlsam.bias_correction, which contains an easier to use interface to the C functions in nlsam.stabilizer
- New online documentation available at http://nlsam.readthedocs.io/ for the current (and future) versions.
- The dictionary learning part of the algorithm now respects **--cores** instead of ignoring it and always using all available processors.
- joblib is now used for parallel processing.
    - The frozen executable is now using dask and performs a bit slower than the normal version until joblib.loky is fixed to work with pyinstaller.
    - Binary wheels are now available for all platforms instead.
- A new option to estimate automatically the noise distribution (sigma and N) is now available by passing **auto** to both N and **--noise_est**.
    - This option is also the new default now.
- A new option to process each shell separately is now available with **--split_shell**.
- Probably other stuff I forgot.

## [0.6.1] - 2017-11-17
- Fixed a numerical issue in the Marcum Q function when computing probabilities used in the stabilizer framework.
- Scipy >= 0.19.1 is now required.
- nlsam.stabilizer.stabilization now accepts the keyword clip_eta (default True), which can be used to allow returning negatives values for eta.
    - The option __--no_clip_eta__ from nlsam_denoising can be used to activate this feature.
    - The previous versions forced negative values to zero and is still the default behavior.

## [0.6] - 2017-10-22

- PIESNO will now warn if less than 1% of noisy voxels were identified, which might indicate that something has gone wrong during the noise estimation.
- On python >= 3.4, __--mp_method__ [a_valid_start_method](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods) can now be used to control behavior in the multiprocessing loop.
- A new option __--split_b0s__ can be specified to split the b0s equally amongst the training data.
- A new (kind of experimental) option __--use_f32__ can be specified to use the float32 mode of spams and reduce ram usage.
- A new option __--use_threading__ can be specified to disable python multiprocessing and solely rely on threading capabilities of the linear algebra libs during denoising.
- Fixed crash in option __--noise_est__ local_std when __--cores__ 1 was also supplied.
- setup.py and requirements.txt will now fetch spams v2.6, with patches for numpy 1.12 support.
- The GSL library and associated headers are now bundled for all platforms.
- Some deprecation fixes and other performance improvements.

## [0.5.1] - 2016-09-27

- Fixed a bug in local piesno (option __--noise_map__) where the noise would be underestimated.
- Fixed a bug introduced in v0.5 where datasets with multiple b0s would be incorrectly reshaped and dropped some volumes. Thanks to Samuel Deslauriers-Gauthier for reporting.

## [0.5] - 2016-08-30

- Heavy refactor of the library. There is now a single script named nlsam_denoising
to replace both stabilizer and nlsam in one go.
The new usage is now

~~~bash
nlsam_denoising input output N bvals bvecs n_angular_neighbors
~~~

- There is also new cool command line options (such as logging) to discover with nlsam_denoising -h
- Some code from the previous scripts was moved in the library part of nlsam,
so now it is easier to hook in with other python projects.
- (Un)Official python 3 support. You will also need to grab an
[unofficial spams build](https://github.com/samuelstjean/spams-python/releases) which has been patched for python 3 support.

## [0.3.1] - 2016-07-11

- The original header is now saved back as is to prevent potential conflicts
with other processing tools. Thanks to Derek Pisner for reporting.

## [0.3] - 2016-05-13

- sh_smooth now uses order 8 by default and a regularized pseudo-inverse for the fit.
The data is also internally converted to float32 to prevent overflowing on uint dtypes. Thanks to Felix Morency for reporting.
- Updated nibabel min version to 2.0, as older version do not have the cache unload function. Thanks to Rutger Fick for reporting.
- Scripts are now more memory friendly with nibabel uncaching.
- The example was moved to a subfolder with an available test dataset.
- Fix multiprocessing freeze_support in windows binaries.
- Scipy >= 0.14 is now required to efficiently deal with sparse matrices.

## [0.2.1] - 2016-04-26

- Fixed a bug in the nlsam script where .nii would be memmaped and crash with an invalid dimension broadcast. Thanks to Jelle Veraart for reporting.

## [0.2] - 2016-04-19

- stabilization script is now clipping values to zero (Bai et al. 2014) as used in the paper.
The previous release was using the Koay et al. 2009 approach, which could produce negative values in really noisy cases.

- More doc to the readme.
- Added link to the synthetic and in vivo data used in the experiments.
- Removed the source archive and wheel files, as one can grab the source archive from github instead.

## [0.1] - 2016-03-22

- First release.
- Windows and Linux binaries available for download.
