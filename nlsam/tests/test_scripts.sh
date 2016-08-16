# We should be in the tester subdir
cd ../example
pwd


check_return_code ()
{
    if [ $1 -ne 0 ]; then
        echo 'Something failed in python apparently'
        exit 1
    fi
}

# Crop example dataset
python -c 'import nibabel as nib; import numpy as np; d = nib.load("dwi.nii.gz").get_data(); nib.save(nib.Nifti1Image(d,np.eye(4)), "dwi_crop.nii.gz")'
python -c 'import nibabel as nib; import numpy as np; d = nib.load("mask.nii.gz").get_data(); nib.save(nib.Nifti1Image(d,np.eye(4)), "mask_crop.nii.gz")'

# Test on example dataset
# nlsam_denoising dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -f --noise_est local_std --sh_order 0 --log log
# check_return_code $?

# nlsam_denoising dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --noise_est local_std --sh_order 6 --iterations 5 --verbose
# check_return_code $?

# nlsam_denoising dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --noise_est piesno --sh_order 0 --b0_threshold 10 --noise_mask pmask.nii.gz
# check_return_code $?

# nlsam_denoising dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --noise_est local_std --sh_order 0 --block_size 4,4,4 --save_sigma sigma.nii.gz
# check_return_code $?

nlsam_denoising dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --sh_order 0 --load_sigma sigma.nii.gz --no_subsample --fix_implausible
check_return_code $?

nlsam_denoising dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --sh_order 0 --noise_map sigma.nii.gz --cores 1 --noise_est piesno --noise_mask pmask.nii.gz
check_return_code $?

nlsam_denoising dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --sh_order 0 --noise_map sigma.nii.gz --no_stabilization --no_denoising
check_return_code $?

# Test on niftis
gunzip dwi_crop.nii.gz mask_crop.nii.gz

nlsam_denoising dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --sh_order 0 --noise_map sigma.nii.gz --no_stabilization --load_sigma sigma.nii.gz --is_symmetric
check_return_code $?

nlsam_denoising dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --sh_order 0 --noise_map sigma.nii.gz --no_denoising --save_sigma sigma.nii.gz --save_stab stab.nii.gz
check_return_code $?
