import logging
import numpy as np

from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm

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
    sh_order : int, default 4
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

        # Find the sh coefficients to predict the signal
        B_dwi = real_sh_descoteaux_from_index(m, n, theta[:, None], phi[:, None])
        invB = smooth_pinv(B_dwi, np.sqrt(regul) * L)
        sh_coeff = data[..., idx] @ invB.T

        # Find the predicted signal from the sh fit for the given bvecs
        pred_sig[..., idx] = sh_coeff @ B_dwi.T
    return pred_sig


def _local_standard_deviation(arr, current_slice=None):
    """Standard deviation estimation from local patches.

    Estimates the local variance on patches by using convolutions
    to estimate the mean. This is the multiprocessed function.

    Parameters
    ----------
    arr : 3D or 4D ndarray
        The array to be estimated

    current_slice: numpy slice object
        current slice to evaluate if we are running in parallel

    Returns
    -------
    sigma : ndarray
        Map of standard deviation of the noise.
    """

    if current_slice is not None:
        arr = arr[current_slice]

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
        slicer = [np.index_exp[..., k] for k in range(arr.shape[-1])]

        if verbose:
            slicer = tqdm(slicer)

        result = Parallel(n_jobs=n_cores)(delayed(_local_standard_deviation)(arr, current_slice) for current_slice in slicer)
        sigma = np.median(result, axis=0)

    # http://en.wikipedia.org/wiki/Full_width_at_half_maximum
    # This defines a normal distribution similar to specifying the variance.
    full_width_at_half_max = 10
    blur = full_width_at_half_max / np.sqrt(8 * np.log(2))

    return gaussian_filter(sigma, blur, mode='reflect')


# Copyright (c) 2008-2024, dipy developers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.

#     * Neither the name of the dipy developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

def cart2sphere(x, y, z):
    r"""Return angles for Cartesian 3D coordinates `x`, `y`, and `z`

    This is the standard physics convention where `theta` is the
    inclination (polar) angle, and `phi` is the azimuth angle.

    $0\le\theta\mathrm{(theta)}\le\pi$ and $-\pi\le\phi\mathrm{(phi)}\le\pi$

    Parameters
    ----------
    x : array_like
       x coordinate in Cartesian space
    y : array_like
       y coordinate in Cartesian space
    z : array_like
       z coordinate

    Returns
    -------
    r : array
       radius
    theta : array
       inclination (polar) angle
    phi : array
       azimuth angle

    """
    r = np.sqrt(x * x + y * y + z * z)
    cos = np.divide(z, r, where=r > 0)
    theta = np.arccos(cos, where=(cos >= -1) & (cos <= 1))
    theta = np.where(r > 0, theta, 0.0)
    phi = np.arctan2(y, x)
    r, theta, phi = np.broadcast_arrays(r, theta, phi)
    return r, theta, phi


def sph_harm_ind_list(sh_order_max):
    """
    Returns the order (``l``) and phase_factor (``m``) of all the symmetric
    spherical harmonics of order less then or equal to ``sh_order_max``.
    The results, ``m_list`` and ``l_list`` are kx1 arrays, where k depends on
    ``sh_order_max``.

    Parameters
    ----------
    sh_order_max : int
        The maximum order ($l$) of the spherical harmonic basis.
        Even int > 0, max order to return

    Returns
    -------
    m_list : array of int
        phase factors ($m$) of even spherical harmonics
    l_list : array of int
        orders ($l$) of even spherical harmonics
    """
    if sh_order_max % 2 != 0:
        raise ValueError("sh_order_max must be an even integer >= 0")
    l_range = np.arange(0, sh_order_max + 1, 2, dtype=np.int16)
    ncoef = int((sh_order_max + 2) * (sh_order_max + 1) // 2)

    l_list = np.repeat(l_range, l_range * 2 + 1)
    offset = 0
    m_list = np.empty(ncoef, dtype=np.int16)
    for ii in l_range:
        m_list[offset : offset + 2 * ii + 1] = np.arange(-ii, ii + 1)
        offset = offset + 2 * ii + 1

    # makes the arrays ncoef by 1, allows for easy broadcasting later in code
    return m_list, l_list


def real_sh_descoteaux_from_index(m_values, l_values, theta, phi):
    r"""Compute real spherical harmonics.

    The definition adopted here follows Descoteaux2007 where the
    real harmonic $Y_l^m$ is defined to be:

    .. math::
       :nowrap:

        Y_l^m =
        \begin{cases}
            \sqrt{2} * \Im(Y_l^m) \; if m > 0 \\
            Y^0_l \; if m = 0 \\
            \sqrt{2} * \Re(Y_l^m)  \; if m < 0 \\
        \end{cases}

    This may take scalar or array arguments. The inputs will be broadcast
    against each other.

    Parameters
    ----------
    m_values : array of int ``|m| <= l``
        The phase factors ($m$) of the harmonics.
    l_values : array of int ``l >= 0``
        The orders ($l$) of the harmonics.
    theta : float [0, pi]
        The polar (colatitudinal) coordinate.
    phi : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.

    Returns
    -------
    real_sh : real float
        The real harmonic $Y_l^m$ sampled at ``theta`` and ``phi``.
    """

    def spherical_harmonics(m_values, l_values, theta, phi):
        try:
            from scipy.special import sph_harm_y
            sph_harm_out = sph_harm_y(l_values, m_values, phi, theta)
        except ImportError:
            from scipy.special import sph_harm
            sph_harm_out = sph_harm(m_values, l_values, theta, phi)

        return sph_harm_out

    sh = spherical_harmonics(np.abs(m_values), l_values, phi, theta)

    real_sh = np.where(m_values > 0, sh.imag, sh.real)
    real_sh *= np.where(m_values == 0, 1.0, np.sqrt(2))

    return real_sh


def smooth_pinv(B, L):
    """Regularized pseudo-inverse

    Computes a regularized least square inverse of B

    Parameters
    ----------
    B : array_like (n, m)
        Matrix to be inverted
    L : array_like (m,)

    Returns
    -------
    inv : ndarray (m, n)
        regularized least square inverse of B

    Notes
    -----
    In the literature this inverse is often written $(B^{T}B+L^{2})^{-1}B^{T}$.
    However here this inverse is implemented using the pseudo-inverse because
    it is more numerically stable than the direct implementation of the matrix
    product.

    """
    L = np.diag(L)
    inv = np.linalg.pinv(np.concatenate((B, L)))
    return inv[:, : len(B)]
