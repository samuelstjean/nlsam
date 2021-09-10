from __future__ import division

import numpy as np
import logging

from nlsam.stabilizer import fixed_point_finder, chi_to_gauss, root_finder, xi
from joblib import Parallel, delayed

logger = logging.getLogger('nlsam')

# Vectorised versions of the above, so we can use implicit broadcasting and stuff
vec_fixed_point_finder = np.vectorize(fixed_point_finder, [np.float64])
vec_chi_to_gauss = np.vectorize(chi_to_gauss, [np.float64])
vec_xi = np.vectorize(xi, [np.float64])
vec_root_finder = np.vectorize(root_finder, [np.float64])


def stabilization(data, m_hat, sigma, N, mask=None, clip_eta=True, return_eta=False, n_cores=-1, verbose=False):

    data = np.asarray(data)
    m_hat = np.asarray(m_hat)
    sigma = np.atleast_3d(sigma)
    N = np.atleast_3d(N)

    if mask is None:
        mask = np.ones(data.shape[:-1], dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    if N.ndim < data.ndim:
        N = np.broadcast_to(N[..., None], data.shape)

    if sigma.ndim == (data.ndim - 1):
        sigma = np.broadcast_to(sigma[..., None], data.shape)

    # Check all dims are ok
    if (data.shape != sigma.shape):
        raise ValueError('data shape {} is not compatible with sigma shape {}'.format(data.shape, sigma.shape))

    if (data.shape[:-1] != mask.shape):
        raise ValueError('data shape {} is not compatible with mask shape {}'.format(data.shape, mask.shape))

    if (data.shape != m_hat.shape):
        raise ValueError('data shape {} is not compatible with m_hat shape {}'.format(data.shape, m_hat.shape))

    arglist = ((data[..., idx, :],
                m_hat[..., idx, :],
                mask[..., idx],
                sigma[..., idx, :],
                N[..., idx, :],
                clip_eta)
               for idx in range(data.shape[-2]))

    # Did we ask for verbose at the module level?
    if not verbose:
        verbose = logger.getEffectiveLevel() <= 20  # Info or debug level

    output = Parallel(n_jobs=n_cores,
                      verbose=verbose)(delayed(multiprocess_stabilization)(*args) for args in arglist)

    data_stabilized = np.zeros_like(data, dtype=np.float32)
    eta = np.zeros_like(data, dtype=np.float32)

    for idx, content in enumerate(output):
        data_stabilized[..., idx, :] = content[0]
        eta[..., idx, :] = content[1]

    if return_eta:
        return data_stabilized, eta
    return data_stabilized


def multiprocess_stabilization(data, m_hat, mask, sigma, N, clip_eta):
    """Helper function for multiprocessing the stabilization part."""

    if mask.ndim == (sigma.ndim - 1):
        mask = mask[..., None]

    mask = np.logical_and(sigma > 0, mask)
    out = np.zeros_like(data, dtype=np.float32)
    eta = np.zeros_like(data, dtype=np.float32)

    eta[mask] = vec_fixed_point_finder(m_hat[mask], sigma[mask], N[mask], clip_eta=clip_eta)
    out[mask] = vec_chi_to_gauss(data[mask], eta[mask], sigma[mask], N[mask], use_nan=False)

    return out, eta


def corrected_sigma(eta, sigma, N, mask=None):
    logger.warning('The function nlsam.bias_correction.corrected_sigma was replaced by nlsam.bias_correction.root_finder_sigma')
    return root_finder_sigma(eta, sigma, N, mask=mask)


def root_finder_sigma(data, sigma, N, mask=None):
    """Compute the local corrected standard deviation for the adaptive nonlocal
    means according to the correction factor xi.

    Input
    --------
    data : ndarray
        Signal intensity
    sigma : ndarray
        Noise magnitude standard deviation
    N : ndarray or double
        Number of coils of the acquisition (N=1 for Rician noise)
    mask : ndarray, optional
        Compute only the corrected sigma value inside the mask.

    Return
    --------
    output, ndarray
        Corrected sigma value, where sigma_gaussian = sigma / sqrt(xi)
    """
    data = np.array(data)
    sigma = np.array(sigma)
    N = np.array(N)

    if mask is None:
        mask = np.ones_like(sigma, dtype=bool)
    else:
        mask = np.array(mask, dtype=bool)

    # Force 3D/4D broadcasting if needed
    if sigma.ndim == (data.ndim - 1):
        sigma = np.broadcast_to(sigma[..., None], data.shape)

    if N.ndim < data.ndim:
        N = np.broadcast_to(N[..., None], data.shape)

    corrected_sigma = np.zeros_like(data, dtype=np.float32)

    # To not murder people ram, we process it slice by slice and reuse the arrays in a for loop
    gaussian_SNR = np.zeros(np.count_nonzero(mask), dtype=np.float32)
    theta = np.zeros_like(gaussian_SNR)

    for idx in range(data.shape[-1]):
        theta[:] = data[..., idx][mask] / sigma[..., idx][mask]
        gaussian_SNR[:] = vec_root_finder(theta, N[..., idx][mask])
        corrected_sigma[..., idx][mask] = sigma[..., idx][mask] / np.sqrt(vec_xi(gaussian_SNR, 1, N[..., idx][mask]))

    return corrected_sigma
