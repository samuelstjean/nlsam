# We should be in the tester subdir
cd ../example
pwd

# Test on example dataset
stabilizer dwi.nii.gz dwi_stab_localstd.nii.gz 1 sigma_localstd.nii.gz -m mask.nii.gz --bvals bvals --bvecs bvecs --noise_est local_std --smooth no_smoothing
stabilizer dwi.nii.gz dwi_stab_piesno.nii.gz 1 sigma_piesno.nii.gz -m mask.nii.gz --bvals bvals --bvecs bvecs
nlsam dwi_stab.nii.gz dwi_nlsam.nii.gz 5 bvals bvecs sigma.nii.gz -m mask.nii.gz
nlsam dwi_stab_piesno.nii.gz dwi_nlsam.nii.gz 5 bvals bvecs sigma_piesno.nii.gz -m mask.nii.gz

# Test on niftis
gunzip dwi.nii.gz mask.nii.gz
stabilizer dwi.nii dwi_stab_localstd.nii 1 sigma_localstd.nii -m mask.nii --bvals bvals --bvecs bvecs --noise_est local_std --smooth no_smoothing
stabilizer dwi.nii dwi_stab_piesno.nii 1 sigma_piesno.nii -m mask.nii --bvals bvals --bvecs bvecs
nlsam dwi_stab.nii dwi_nlsam.nii 5 bvals bvecs sigma.nii -m mask.nii
nlsam dwi_stab_piesno.nii dwi_nlsam.nii 5 bvals bvecs sigma_piesno.nii -m mask.nii
