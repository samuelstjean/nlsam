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
nlsam_denoising dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -f --noise_est local_std --sh_order 0 --log log --cores 1
check_return_code $?

nlsam_denoising dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --noise_est local_std --sh_order 6 --iterations 5 --verbose --save_sigma sigma.nii.gz
check_return_code $?

nlsam_denoising dwi_crop.nii.gz dwi_nlsam.nii.gz auto bvals bvecs 5 -m mask_crop.nii.gz -f --noise_est auto --sh_order 6 --iterations 5 --verbose --save_sigma sigma.nii.gz --save_N N.nii.gz
check_return_code $?

nlsam_denoising dwi_crop.nii.gz dwi_nlsam.nii.gz 1 bvals bvecs 5 -m mask_crop.nii.gz -f --verbose --sh_order 0 --b0_threshold 10 --noise_mask pmask.nii.gz
check_return_code $?

# Test on niftis
gunzip dwi_crop.nii.gz mask_crop.nii.gz sigma.nii.gz

nlsam_denoising dwi_crop.nii dwi_nlsam.nii 1 bvals bvecs 5 -m mask_crop.nii -f --verbose --sh_order 0 --no_stabilization --load_sigma sigma.nii --is_symmetric --use_threading --save_difference diff.nii
check_return_code $?

nlsam_denoising dwi_crop.nii dwi_nlsam.nii 1 bvals bvecs 5 -m mask_crop.nii -f --verbose --sh_order 0 --no_denoising --save_sigma sigma.nii --save_stab stab.nii --load_mhat dwi_crop.nii --save_eta eta.nii
check_return_code $?
