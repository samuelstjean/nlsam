# We should be in the tester subdir
cd ../example
pwd


function check_return_code()
{
    if [ $1 -ne 0 ]; then
        echo 'Something failed in python apparently'
        exit 1
    fi
}
# Test on example dataset
stabilizer dwi.nii.gz dwi_stab_localstd.nii.gz 1 sigma_localstd.nii.gz -m mask.nii.gz --bvals bvals --bvecs bvecs --noise_est local_std --smooth no_smoothing
check_return_code $?

stabilizer dwi.nii.gz dwi_stab_piesno.nii.gz 1 sigma_piesno.nii.gz -m mask.nii.gz --bvals bvals --bvecs bvecs
check_return_code $?

nlsam dwi_stab_localstd.nii.gz dwi_nlsam_localstd.nii.gz 5 bvals bvecs sigma_localstd.nii.gz -m mask.nii.gz
check_return_code $?

nlsam dwi_stab_piesno.nii.gz dwi_nlsam.nii.gz 5 bvals bvecs sigma_piesno.nii.gz -m mask.nii.gz
check_return_code $?

# Test on niftis
gunzip dwi.nii.gz mask.nii.gz
stabilizer dwi.nii dwi_stab_localstd.nii 1 sigma_localstd.nii -m mask.nii --bvals bvals --bvecs bvecs --noise_est local_std --smooth no_smoothing
check_return_code $?

stabilizer dwi.nii dwi_stab_piesno.nii 1 sigma_piesno.nii -m mask.nii --bvals bvals --bvecs bvecs
check_return_code $?

nlsam dwi_stab_localstd.nii dwi_nlsam_localstd.nii 5 bvals bvecs sigma_localstd.nii -m mask.nii
check_return_code $?

nlsam dwi_stab_piesno.nii dwi_nlsam_piesno.nii 5 bvals bvecs sigma_piesno.nii -m mask.nii
check_return_code $?
