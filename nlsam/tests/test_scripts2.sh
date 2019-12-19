# We should be in the tester subdir
cd ../example
pwd


check_return_code ()
{
    if [ $1 -ne 0 ]; then
        echo 'Something failed in python apparently'
        exit 1
    fi

    echo 'stuff finished'
}

# Crop example dataset
python -c 'import nibabel as nib; import numpy as np; d = nib.load("dwi.nii.gz").get_fdata(); nib.save(nib.Nifti1Image(d[40:50],np.eye(4)), "dwi_crop.nii.gz")'
python -c 'import nibabel as nib; import numpy as np; d = nib.load("mask.nii.gz").get_fdata(); nib.save(nib.Nifti1Image(d[40:50],np.eye(4)), "mask_crop.nii.gz")'
python -c 'import nibabel as nib; import numpy as np; d = nib.load("mask.nii.gz").get_fdata(); nib.save(nib.Nifti1Image(np.random.rayleigh(10, d[40:50].shape),np.eye(4)), "noise.nii.gz")'

# Test on example dataset
nlsam_denoising dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --verbose --noise_est local_std --sh_order 0 --block_size 2,2,2 --save_sigma sigma.nii.gz --cores 1
check_return_code $?

nlsam_denoising dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --verbose --sh_order 0 --load_sigma sigma.nii.gz --no_subsample --fix_implausible --no_clip_eta #--mp_method spawn
check_return_code $?

nlsam_denoising dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --verbose --sh_order 0 --noise_map noise.nii.gz --noise_mask pmask.nii.gz --use_f32
check_return_code $?

nlsam_denoising dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --verbose --sh_order 0 --load_sigma sigma.nii.gz --no_stabilization --no_denoising
check_return_code $?

nlsam_denoising dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --verbose --sh_order 0 --load_sigma sigma.nii.gz --no_denoising --no_clip_eta
check_return_code $?

nlsam_denoising dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --verbose --sh_order 0 --noise_est local_std --no_denoising
check_return_code $?

nlsam_denoising dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --verbose --sh_order 0 --no_denoising
check_return_code $?
