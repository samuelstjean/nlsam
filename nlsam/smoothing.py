import logging
import numpy as np

from joblib import Parallel, delayed

from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import sph_harm_ind_list, real_sph_harm, smooth_pinv

from scipy.ndimage import convolve, gaussian_filter

logger = logging.getLogger('nlsam')


def sh_smooth(data, bvals, bvecs, sh_order=4, b0_threshold=1.0, similarity_threshold=50, regul=0.006):
    """Smooth the raw diffusion signal with spherical harmonics.

    data : ndarray
        The diffusion data to smooth.
    gtab : gradient table object
        Corresponding gradients table object to data.
    b0_threshold : float, default 1.0
        Threshold to consider this bval as a b=0 image.
    sh_order : int, default 8
        Order of the spherical harmonics to fit.
    similarity_threshold : int, default 50
        All bvalues such that |b_1 - b_2| < similarity_threshold
        will be considered as identical for smoothing purpose.
        Must be lower than 200.
    regul : float, default 0.006
        Amount of regularization to apply to sh coefficients computation.

    Return
    ---------
    pred_sig : ndarray
        The smoothed diffusion data, fitted through spherical harmonics.
    """

    if similarity_threshold > 200:
        error = f"similarity_threshold = {similarity_threshold}, which is higher than 200, please use a lower value"
        raise ValueError(error)

    if b0_threshold > 20:
        error = f"b0_threshold = {b0_threshold}, which is higher than 20, please use a lower value"
        raise ValueError(error)

    m, n = sph_harm_ind_list(sh_order)
    L = -n * (n + 1)
    where_b0s = bvals <= b0_threshold
    pred_sig = np.zeros_like(data, dtype=np.float32)


    # Round similar bvals together for identifying similar shells
    rounded_bvals = np.zeros_like(bvals)

    for unique_bval in np.unique(bvals):
        idx = np.abs(unique_bval - bvals) < similarity_threshold
        rounded_bvals[idx] = unique_bval

    # process each bvalue separately
    for unique_bval in np.unique(rounded_bvals):
        idx = rounded_bvals == unique_bval

        # Just give back the signal for the b0s since we can't really do anything about it
        if np.all(idx == where_b0s):
            if np.sum(where_b0s) > 1:
                pred_sig[..., idx] = np.mean(data[..., idx], axis=-1, keepdims=True)
            else:
                pred_sig[..., idx] = data[..., idx]
            continue


        x, y, z = bvecs[:, idx]
        _, theta, phi = cart2sphere(x, y, z)

        # Find the sh coefficients to smooth the signal
        B_dwi = real_sph_harm(m, n, theta[:, None], phi[:, None])
        invB = smooth_pinv(B_dwi, np.sqrt(regul) * L)
        sh_coeff = np.dot(data[..., idx], invB.T)
        # Find the smoothed signal from the sh fit for the given gtab
        pred_sig[..., idx] = np.dot(sh_coeff, B_dwi.T)
    return pred_sig


def _local_standard_deviation(arr):
    """Standard deviation estimation from local patches.

    Estimates the local variance on patches by using convolutions
    to estimate the mean. This is the multiprocessed function.

    Parameters
    ----------
    arr : 3D or 4D ndarray
        The array to be estimated

    Returns
    -------
    sigma : ndarray
        Map of standard deviation of the noise.
    """

    size = (3, 3, 3)
    k = np.ones(size) / np.sum(np.ones(size))

    low_pass_arr = np.empty_like(arr, dtype=np.float32)
    mean_squared_high_freq = np.empty_like(arr, dtype=np.float32)
    mean_high_freq = np.empty_like(arr, dtype=np.float32)

    # A noise field estimation is made by subtracting the data
    # from it's low pass filtered version
    convolve(arr, k, mode='reflect', output=low_pass_arr)
    high_freq = arr - low_pass_arr

    # Compute the variance of the estimated noise field
    # First use a convolution to get the sum of the squared mean, then
    # compute the mean with another convolution.
    convolve(high_freq**2, k, mode='reflect', output=mean_squared_high_freq)
    convolve(high_freq, k, mode='reflect', output=mean_high_freq)

    # Variance = mean(x**2) - mean(x)**2,
    # but we work with the standard deviation so we return the square root
    return np.sqrt(mean_squared_high_freq - mean_high_freq**2)


def local_standard_deviation(arr, n_cores=-1, verbose=False):
    """Standard deviation estimation from local patches.

    The noise field is estimated by subtracting the data from it's low pass
    filtered version, from which we then compute the variance on a local
    neighborhood basis.

    Parameters
    ----------
    arr : 3D or 4D ndarray
        The array to be estimated

    n_cores : int
        Number of cores to use for multiprocessing, default : all of them

    verbose: int
        If True, prints progress information. A higher number prints more often

    Returns
    -------
    sigma : ndarray
        Map of standard deviation of the noise.
    """

    # No multiprocessing for 3D array since we smooth on each separate volume
    if arr.ndim == 3:
        sigma = _local_standard_deviation(arr)
    else:
        # Did we ask for verbose at the module level?
        if not verbose:
            verbose = logger.getEffectiveLevel() <= 20  # Info or debug level

        result = Parallel(n_jobs=n_cores,
                          verbose=verbose)(delayed(_local_standard_deviation)(arr[..., i]) for i in range(arr.shape[-1]))

        # Reshape the multiprocessed list as an array
        result = np.rollaxis(np.asarray(result), 0, arr.ndim)
        sigma = np.median(result, axis=-1)

    # http://en.wikipedia.org/wiki/Full_width_at_half_maximum
    # This defines a normal distribution similar to specifying the variance.
    full_width_at_half_max = 10
    blur = full_width_at_half_max / np.sqrt(8 * np.log(2))

    return gaussian_filter(sigma, blur, mode='reflect')
