# Changelog

## [0.5] - unreleased

- Heavy refactor of the library. There is now a single script named nlsam_denoising
to replace both stabilizer and nlsam in one go.
The new usage is now

~~~bash
nlsam_denoising input output N bvals bvecs n_angular_neighbors
~~~

- There is also new cool command line options (such as logging) to discover with nlsam_denoising -h
- Some code from the previous scripts was moved in the library part of nlsam,
so now it is easier to hook in with other python projects.

## [0.3.1] - 2016-07-11

- The original header is now saved back as-is to prevent potential conflicts
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
