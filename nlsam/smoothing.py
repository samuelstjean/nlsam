from __future__ import division

import numpy as np

from multiprocessing import Pool, cpu_count

from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import sph_harm_ind_list, real_sph_harm, lazy_index
from scipy.ndimage.filters import convolve, gaussian_filter
from scipy.special import digamma


def sh_smooth(data, gtab, sh_order=4):
    """Smooth the raw diffusion signal with spherical harmonics

    data : ndarray
        The diffusion data to smooth.

    gtab : gradient table object
        Corresponding gradients table object to data.

    sh_order : int, default 4
        Order of the spherical harmonics to fit.

    Return
    ---------
    pred_sig : ndarray
        The smoothed diffusion data, fitted through spherical harmonics.
    """

    m, n = sph_harm_ind_list(sh_order)
    where_b0s = lazy_index(gtab.b0s_mask)
    where_dwi = lazy_index(~gtab.b0s_mask)

    x, y, z = gtab.gradients[where_dwi].T
    r, theta, phi = cart2sphere(x, y, z)

    # Find the sh coefficients to smooth the signal
    B_dwi = real_sph_harm(m, n, theta[:, None], phi[:, None])
    sh_coeff = np.linalg.lstsq(B_dwi, data[..., where_dwi].reshape(np.prod(data.shape[:-1]), -1).T)[0]

    # Find the smoothed signal from the sh fit for the given gtab
    smoothed_signal = np.dot(B_dwi, sh_coeff).T.reshape(data.shape[:-1] + (-1,))
    pred_sig = np.zeros(smoothed_signal.shape[:-1] + (gtab.bvals.shape[0],))
    pred_sig[..., ~gtab.b0s_mask] = smoothed_signal

    # Just give back the signal for the b0s since we can't really do anything about it
    if np.sum(gtab.b0s_mask) > 1:
        pred_sig[..., where_b0s] = np.mean(data[..., where_b0s], axis=-1, keepdims=True)
    else:
        pred_sig[..., where_b0s] = data[..., where_b0s]

    return pred_sig


def _local_standard_deviation(arr):
    """Standard deviation estimation from local patches

    This is the multiprocessed function.

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
    mean_squared_noise = np.empty_like(arr, dtype=np.float32)
    mean_noise = np.empty_like(arr, dtype=np.float32)

    convolve(arr, k, mode='reflect', output=low_pass_arr)
    noise = arr - low_pass_arr

    convolve(noise**2, k, mode='reflect', output=mean_squared_noise)
    convolve(noise, k, mode='reflect', output=mean_noise)

    # Variance = mean(x**2) - mean(x)**2
    return np.sqrt(mean_squared_noise - mean_noise**2)


def local_standard_deviation(arr, n_cores=None):
    """Standard deviation estimation from local patches.

    Parameters
    ----------
    arr : 3D or 4D ndarray
        The array to be estimated

    n_cores : int
        Number of cores to use for multiprocessing, default : all of them

    Returns
    -------
    sigma : ndarray
        Map of standard deviation of the noise.
    """

    # No multiprocessing for 3D array since we smooth on each separate volume
    if arr.ndim == 3:
        arr = arr[..., None]
        n_cores = 1

    if n_cores == 1:
        result = _local_standard_deviation(arr)

    else:

        list_arr = []
        for i in range(arr.shape[-1]):
            list_arr += [arr[..., i]]

        if n_cores is None:
            n_cores = cpu_count()

        pool = Pool(n_cores)
        result = pool.map(_local_standard_deviation, list_arr)
        pool.close()
        pool.join()

        # Reshape the multiprocessed list as an array
        result = np.rollaxis(np.asarray(result), 0, arr.ndim)

    fwhm = 10
    blur = fwhm / np.sqrt(8 * np.log(2))
    sigma = np.median(result, axis=-1)

    # size = (5, 5, 5)
    # k = np.ones(size) / np.sum(np.ones(size))
    # convolve(sigma, k, mode='reflect') #

    return gaussian_filter(sigma, blur, mode='reflect')


def homomorphic_noise_estimation(data):

    euler_mascheroni = -digamma(1)

    conv_out = np.empty_like(data[..., 0], dtype=np.float32)
    m_hat = np.empty_like(data, dtype=np.float32)
    low_pass = np.empty_like(data, dtype=np.float32)

    blur = 4.8
    size = (5, 5, 5)
    k = np.ones(size) / np.sum(size)

    for idx in range(data.shape[-1]):
        convolve(data[..., idx], k, mode='reflect', output=conv_out)
        m_hat[...,  idx] = np.log(np.abs(data[..., idx] - conv_out) + 1e-6)
        low_pass[..., idx] = gaussian_filter(m_hat[..., idx], blur, mode='reflect')

    # low_pass = np.median(low_pass, axis=-1)

    # return np.sqrt(2) * np.exp(low_pass + euler_mascheroni/2)
    return np.sqrt(np.exp(low_pass) * 2/np.sqrt(2) * np.exp(euler_mascheroni/2))


def local_noise_map_std(noise_map):

    size = (3, 3, 3)
    k = np.ones(size) / np.sum(np.ones(size))

    mean_squared_noise = np.empty_like(noise_map, dtype=np.float32)
    mean_noise = np.empty_like(noise_map, dtype=np.float32)

    convolve(noise_map**2, k, mode='reflect', output=mean_squared_noise)
    convolve(noise_map, k, mode='reflect', output=mean_noise)

    # Variance = mean(x**2) - mean(x)**2
    local_std = np.sqrt(mean_squared_noise - mean_noise**2)

    fwhm = 10
    blur = fwhm / np.sqrt(8 * np.log(2))

    return gaussian_filter(local_std, blur, mode='reflect')
