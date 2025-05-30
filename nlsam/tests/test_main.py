import os
import gzip
import shutil
import subprocess

import pytest

from pathlib import Path

def prepare_datasets():
    def unzip(zip, file):
        with gzip.open(zip, 'rb') as f_in:
            with open(file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    zip_files = ['dwi_crop.nii.gz',
                 'mask_crop.nii.gz',
                 'sigma.nii.gz']

    for file in zip_files:
        unzipped = file.removesuffix('.gz')
        if os.path.isfile(file) and not os.path.isfile(unzipped):
            unzip(file, unzipped)


commands = [
    ('nlsam_denoising', 'dwi_crop.nii.gz', 'dwi_nlsam.nii.gz', 'bvals', 'bvecs', '-f', '-N', '1', '--noise_est', 'local_std', '--sh_order', '0', '--cores', '1', '-m', 'mask_crop.nii.gz', '-v'),
    ('nlsam_denoising', 'dwi_crop.nii.gz', 'dwi_nlsam.nii.gz', 'bvals', 'bvecs', '-m', 'mask_crop.nii.gz', '-f', '-N', '1', '--noise_est', 'local_std', '--sh_order', '6', '--iterations', '5', '--verbose', '--save_sigma', 'sigma.nii.gz', '--log', 'log.txt'),
    ('nlsam_denoising', 'dwi_crop.nii.gz', 'dwi_nlsam.nii.gz', 'bvals', 'bvecs', '-m', 'mask_crop.nii.gz', '-f', '--noise_est', 'auto', '--sh_order', '6', '--iterations', '5', '--verbose', '--save_sigma', 'sigma.nii.gz', '--save_N', 'N.nii.gz'),
    ('nlsam_denoising', 'dwi_crop.nii.gz', 'dwi_nlsam.nii.gz', 'bvals', 'bvecs', '-m', 'mask_crop.nii.gz', '-f', '--verbose', '--sh_order', '0', '--b0_threshold', '10', '--noise_mask', 'pmask.nii.gz'),
    ('nlsam_denoising', 'dwi_crop.nii', 'dwi_nlsam.nii', 'bvals', 'bvecs', '-m', 'mask_crop.nii', '-f', '--verbose', '--sh_order', '0', '-N', '1', '--no_stabilization', '--load_sigma', 'sigma.nii', '--is_symmetric', '--use_threading', '--save_difference', 'diff.nii'),
    ('nlsam_denoising', 'dwi_crop.nii', 'dwi_nlsam.nii', 'bvals', 'bvecs', '-m', 'mask_crop.nii', '-f', '--verbose', '--sh_order', '0', '--no_denoising', '--save_sigma', 'sigma.nii', '--save_stab', 'stab.nii', '--load_mhat', 'dwi_crop.nii', '--save_eta', 'eta.nii'),
    ('nlsam_denoising', 'dwi_crop.nii.gz', 'dwi_nlsam.nii.gz', 'bvals', 'bvecs', '-m', 'mask_crop.nii.gz', '-f', '--verbose', '-N', '1', '--noise_est', 'local_std', '--sh_order', '0', '--block_size', '2,2,2', '--save_sigma', 'sigma.nii.gz', '--cores', '1'),
    ('nlsam_denoising', 'dwi_crop.nii.gz', 'dwi_nlsam.nii.gz', 'bvals', 'bvecs', '-m', 'mask_crop.nii.gz', '-f', '--verbose', '--sh_order', '0', '-N', '1', '--load_sigma', 'sigma.nii.gz', '--no_subsample', '--fix_implausible', '--no_clip_eta'),
    ('nlsam_denoising', 'dwi_crop.nii.gz', 'dwi_nlsam.nii.gz', 'bvals', 'bvecs', '-m', 'mask_crop.nii.gz', '-f', '--verbose', '--sh_order', '0', '--noise_map', 'noise.nii.gz', '--noise_mask', 'pmask.nii.gz', '--use_f32'),
    ('nlsam_denoising', 'dwi_crop.nii.gz', 'dwi_nlsam.nii.gz', 'bvals', 'bvecs', '-m', 'mask_crop.nii.gz', '-f', '--verbose', '--sh_order', '0', '-N', '1', '--load_sigma', 'sigma.nii.gz', '--no_stabilization', '--no_denoising'),
    ('nlsam_denoising', 'dwi_crop.nii.gz', 'dwi_nlsam.nii.gz', 'bvals', 'bvecs', '-m', 'mask_crop.nii.gz', '-f', '--verbose', '--sh_order', '0', '-N', '1', '--load_sigma', 'sigma.nii.gz', '--no_denoising', '--no_clip_eta'),
    ('nlsam_denoising', 'dwi_crop.nii.gz', 'dwi_nlsam.nii.gz', 'bvals', 'bvecs', '-m', 'mask_crop.nii.gz', '-f', '--verbose', '--sh_order', '0', '-N', '1', '--noise_est', 'local_std', '--no_denoising'),
    ('nlsam_denoising', 'dwi_crop.nii.gz', 'dwi_nlsam.nii.gz', 'bvals', 'bvecs', '-m', 'mask_crop.nii.gz', '-f', '--verbose', '--sh_order', '0', '--noise_est', 'auto', '--no_denoising', '--cores', '4'),
    ('nlsam_denoising', 'dwi_crop.nii.gz', 'dwi_nlsam.nii.gz', 'bvals', 'bvecs', '-m', 'mask_crop.nii.gz', '-f', '--verbose', '--sh_order', '0', '--no_denoising')
    ]

cwd = Path(__file__).parent / Path("datasets")

@pytest.mark.parametrize('command', commands)
def test_script(command):
    os.chdir(cwd)
    prepare_datasets()
    subprocess.run(command, cwd=cwd, check=True)
