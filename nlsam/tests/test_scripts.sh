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
python -c 'import nibabel as nib; import numpy as np; d = nib.load("dwi.nii.gz").get_data(); nib.save(nib.Nifti1Image(d[:,:,90:110],np.eye(4)), "dwi_crop.nii.gz")'
python -c 'import nibabel as nib; import numpy as np; d = nib.load("mask.nii.gz").get_data(); nib.save(nib.Nifti1Image(d[:,:,90:110],np.eye(4)), "mask_crop.nii.gz")'

# Test on example dataset
nlsam dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -f --noise_est local_std --smooth no_smoothing --log log
check_return_code $?

nlsam dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --noise_est local_std --smooth sh_smooth --sh_order 6 --iterations 5 -v
check_return_code $?

nlsam dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --noise_est piesno --smooth no_smoothing --b0_threshold 10 --noise_mask pmask.nii.gz
check_return_code $?

nlsam dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --noise_est local_std --smooth no_smoothing --block_size 4,4,4 --save_sigma sigma.nii.gz
check_return_code $?

nlsam dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --smooth no_smoothing --load_sigma sigma.nii.gz --no_subsample
check_return_code $?

nlsam dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --smooth no_smoothing --noise_map sigma.nii.gz --cores 1
check_return_code $?

nlsam dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --smooth no_smoothing --noise_map sigma.nii.gz --no_stabilization --no_denoising
check_return_code $?

nlsam dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --smooth no_smoothing --noise_map sigma.nii.gz --no_stabilization --load_sigma sigma.nii.gz
check_return_code $?

nlsam dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --smooth no_smoothing --noise_map sigma.nii.gz --no_denoising --save_sigma sigma.nii.gz --save_stab stab.nii.gz
check_return_code $?

# # Test on niftis
# gunzip dwi.nii.gz mask.nii.gz


# stabilizer dwi.nii dwi_stab_localstd.nii 1 sigma_localstd.nii -m mask.nii --bvals bvals --bvecs bvecs --noise_est local_std --smooth no_smoothing --sh_order 6
# check_return_code $?

# stabilizer dwi.nii dwi_stab_piesno.nii 1 sigma_piesno.nii -m mask.nii --bvals bvals --bvecs bvecs --sh_order 6
# check_return_code $?

# nlsam dwi_stab_localstd.nii dwi_nlsam_localstd.nii 5 bvals bvecs sigma_localstd.nii -m mask.nii -f
# check_return_code $?

# nlsam dwi_stab_piesno.nii dwi_nlsam_piesno.nii 5 bvals bvecs sigma_piesno.nii -m mask.nii -f
# check_return_code $?
