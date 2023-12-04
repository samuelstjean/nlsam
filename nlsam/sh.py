# Some functions from

# https://github.com/dipy/dipy/blob/master/dipy/core/geometry.py
# https://github.com/dipy/dipy/blob/master/dipy/reconst/shm.py

# Copyright (c) 2008-2023, dipy developers
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

import logging
import numpy as np

from scipy.special import sph_harm

logger = logging.getLogger('nlsam')


def cart2sphere(x, y, z):
    r""" Return angles for Cartesian 3D coordinates `x`, `y`, and `z`

    See doc for ``sphere2cart`` for angle conventions and derivation
    of the formulae.

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
    theta = np.arccos(np.divide(z, r, where=r > 0))
    theta = np.where(r > 0, theta, 0.)
    phi = np.arctan2(y, x)
    r, theta, phi = np.broadcast_arrays(r, theta, phi)
    return r, theta, phi


def sph_harm_ind_list(sh_order, full_basis=False):
    """
    Returns the degree (``m``) and order (``n``) of all the symmetric spherical
    harmonics of degree less then or equal to ``sh_order``. The results,
    ``m_list`` and ``n_list`` are kx1 arrays, where k depends on ``sh_order``.
    They can be passed to :func:`real_sh_descoteaux_from_index` and
    :func:``real_sh_tournier_from_index``.

    Parameters
    ----------
    sh_order : int
        even int > 0, max order to return
    full_basis: bool, optional
        True for SH basis with even and odd order terms

    Returns
    -------
    m_list : array
        degrees of even spherical harmonics
    n_list : array
        orders of even spherical harmonics

    See Also
    --------
    shm.real_sh_descoteaux_from_index, shm.real_sh_tournier_from_index

    """
    if full_basis:
        n_range = np.arange(0, sh_order + 1, dtype=int)
        ncoef = int((sh_order + 1) * (sh_order + 1))
    else:
        if sh_order % 2 != 0:
            raise ValueError('sh_order must be an even integer >= 0')
        n_range = np.arange(0, sh_order + 1, 2, dtype=int)
        ncoef = int((sh_order + 2) * (sh_order + 1) // 2)

    n_list = np.repeat(n_range, n_range * 2 + 1)
    offset = 0
    m_list = np.empty(ncoef, dtype=int)
    for ii in n_range:
        m_list[offset:offset + 2 * ii + 1] = np.arange(-ii, ii + 1)
        offset = offset + 2 * ii + 1

    # makes the arrays ncoef by 1, allows for easy broadcasting later in code
    return m_list, n_list


def real_sh_descoteaux_from_index(m, n, theta, phi):
    """ Compute real spherical harmonics as in Descoteaux et al. 2007 [1]_,
    where the real harmonic $Y^m_n$ is defined to be:

        Imag($Y^m_n$) * sqrt(2)      if m > 0
        $Y^0_n$                      if m = 0
        Real($Y^m_n$) * sqrt(2)      if m < 0

    This may take scalar or array arguments. The inputs will be broadcast
    against each other.

    Parameters
    ----------
    m : int ``|m| <= n``
        The degree of the harmonic.
    n : int ``>= 0``
        The order of the harmonic.
    theta : float [0, pi]
        The polar (colatitudinal) coordinate.
    phi : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.

    Returns
    -------
    real_sh : real float
        The real harmonic $Y^m_n$ sampled at ``theta`` and ``phi``.

    References
    ----------
     .. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
           Regularized, Fast, and Robust Analytical Q-ball Imaging.
           Magn. Reson. Med. 2007;58:497-510.
    """

    def spherical_harmonics(m, n, theta, phi):
        return sph_harm(m, n, theta, phi, dtype=complex)

    sh = spherical_harmonics(m, n, phi, theta)

    real_sh = np.where(m > 0, sh.imag, sh.real)
    real_sh *= np.where(m == 0, 1, np.sqrt(2))

    return real_sh


def smooth_pinv(B, L):
    """Regularized pseudo-inverse

    Computes a regularized least square inverse of B

    Parameters
    ----------
    B : array_like (n, m)
        Matrix to be inverted
    L : array_like (n,)

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
    return inv[:, :len(B)]


def sh_smooth(data, bvals, bvecs, sh_order=4, b0_threshold=10, similarity_threshold=50, regul=0.006):
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
        B_dwi = real_sh_descoteaux_from_index(m, n, theta[:, None], phi[:, None])
        invB = smooth_pinv(B_dwi, np.sqrt(regul) * L)
        sh_coeff = np.dot(data[..., idx], invB.T)
        # Find the smoothed signal from the sh fit for the current shell
        pred_sig[..., idx] = np.dot(sh_coeff, B_dwi.T)
    return pred_sig
