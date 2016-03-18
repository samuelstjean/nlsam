from __future__ import division

import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
# import warnings

from multiprocessing import Pool, cpu_count
from warnings import warn

from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import sph_harm_ind_list, real_sph_harm, lazy_index
from dipy.denoise.noise_estimate import piesno

from scipy.ndimage.filters import convolve, gaussian_filter
from scipy.ndimage.interpolation import zoom
# from scipy.special import digamma


def sh_smooth(data, gtab, sh_order=4, similarity_threshold=50):
    """Smooth the raw diffusion signal with spherical harmonics.

    data : ndarray
        The diffusion data to smooth.

    gtab : gradient table object
        Corresponding gradients table object to data.

    sh_order : int, default 4
        Order of the spherical harmonics to fit.

    similarity_threshold : int, default 50
        All bvalues such that |b_1 - b_2| < similarity_threshold
        will be considered as identical for smoothing purpose.
        Must be lower than 200.

    Return
    ---------
    pred_sig : ndarray
        The smoothed diffusion data, fitted through spherical harmonics.
    """

    if similarity_threshold > 200:
        raise ValueError("similarity_threshold = {}, which is higher than 200, \
            please use a lower value".format(similarity_threshold))

    m, n = sph_harm_ind_list(sh_order)
    where_b0s = lazy_index(gtab.b0s_mask)
    where_dwi = lazy_index(~gtab.b0s_mask)
    pred_sig = np.zeros_like(data)

    # Round similar bvals together for identifying similar shells
    bvals = gtab.bvals[where_dwi]
    rounded_bvals = np.zeros_like(bvals)

    for unique_bval in np.unique(bvals):
        idx = np.abs(unique_bval - bvals) < similarity_threshold
        rounded_bvals[idx] = unique_bval

    # process each b-value separately
    for unique_bval in np.unique(rounded_bvals):
        idx = rounded_bvals == unique_bval

        # Check if enough data for requested sh order
        if np.sum(idx) < (sh_order + 1) * (sh_order + 2) / 2:
            warn("bval {} has not enough values for sh order {}.\nPutting back the original values.").format(unique_bval, sh_order)
            pred_sig[..., idx] = data[..., idx]

            continue

        # Just give back the signal for the b0s since we can't really do anything about it
        if np.all(idx == where_b0s):

            if np.sum(gtab.b0s_mask) > 1:
                pred_sig[..., idx] = np.mean(data[..., idx], axis=-1, keepdims=True)
            else:
                pred_sig[..., idx] = data[..., idx]

            continue

        x, y, z = gtab.gradients[idx].T
        r, theta, phi = cart2sphere(x, y, z)

        # Find the sh coefficients to smooth the signal
        B_dwi = real_sph_harm(m, n, theta[:, None], phi[:, None])
        sh_coeff = np.linalg.lstsq(B_dwi, data[..., idx].reshape(np.prod(data.shape[:-1]), -1).T)[0]

        # Find the smoothed signal from the sh fit for the given gtab
        pred_sig[..., idx] = np.dot(B_dwi, sh_coeff).T.reshape(data.shape[:-1] + (-1,))

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

    # A noise field estimation is made by substracting the data
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


def local_standard_deviation(arr, n_cores=None):
    """Standard deviation estimation from local patches.

    The noise field is estimated by substrating the data from it's low pass
    filtered version, from which we then compute the variance on a local
    neighborhood basis.

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

    # http://en.wikipedia.org/wiki/Full_width_at_half_maximum
    # This defines a normal distribution similar to specifying the variance.
    full_width_at_half_max = 10
    blur = full_width_at_half_max / np.sqrt(8 * np.log(2))

    sigma = np.median(result, axis=-1)

    return gaussian_filter(sigma, blur, mode='reflect')


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


def local_piesno(data, N, size=5, return_mask=True):

    m_out = np.zeros(data.shape[:-1], dtype=np.bool)
    reshaped_maps = sliding_window(data, (size, size, size, data.shape[-1]))

    sigma = np.zeros(reshaped_maps.shape[0], dtype=np.float32)
    mask = np.zeros((reshaped_maps.shape[0], size**3), dtype=np.bool)

    for i in range(reshaped_maps.shape[0]):
        cur_map = reshaped_maps[i].reshape(size**3, 1, -1)
        sigma[i], m = piesno(cur_map, N=N, return_mask=True)
        mask[i] = np.squeeze(m)
        sigma[i] = np.std(cur_map)

    s_out = sigma.reshape(data.shape[0]//size, data.shape[1]//size, data.shape[2]//size)

    n = 0
    for i in np.ndindex(s_out.shape):
        i = np.array(i) * size
        j = i + size
        m_out[i[0]:j[0], i[1]:j[1], i[2]:j[2]] = mask[n].reshape(size, size, size)
        n += 1

    interpolated = np.zeros_like(data[..., 0], dtype=np.float32)
    x, y, z = np.array(s_out.shape) * size
    interpolated[:x, :y, :z] = zoom(s_out, size, order=1)

    if return_mask:
        return interpolated, m_out

    return interpolated


# Stolen from http://www.johnvinyard.com/blog/?p=268
def sliding_window(a, ws, ss=None, flatten=True):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError('a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError('ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape), str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a, shape=newshape, strides=newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i: i != 1, dim)
    return strided.reshape(dim)


def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')
