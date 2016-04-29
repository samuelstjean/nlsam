# Changelog

- Updated nibabel min version to 2.0, as older version do not have the cache unload function. Thanks to Rutger Fick for reporting.
- Scripts are now more memory friendly with nibabel uncaching.
- Fix multiprocessing freeze_support in windows binaries.
- Linux binaries now needs glibc >= 2.13 (Debian 7 and newer) instead of glibc >= 2.21 (Ubuntu 15.04 and newer).

## [0.2.1] - 2016-04-26

- Fixed a bug in the nlsam script where .nii would be memmaped and crash with an invalid dimension broadcast. Thanks to Jelle Veraart for reporting.
- Fix multiprocessing freeze_support in windows binaries.

## [0.2] - 2016-04-19

- stabilization script is now clipping values to zero (Bai et al. 2014) as used in the paper.
The previous release was using the Koay et al. 2009 approach, which could produce negative values in really noisy cases.

- More doc to the readme.
- Added link to the synthetic and in vivo data used in the experiments.
- Removed the source archive and wheel files, as one can grab the source archive from github instead.

## [0.1] - 2016-03-22

- First release.
- Windows and Linux binaries available for download.
