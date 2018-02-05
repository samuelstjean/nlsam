from __future__ import division

import numpy as np

from nlsam.multiprocess import multiprocesser
from nlsam.stabilizer import fixed_point_finder, chi_to_gauss, root_finder
from nlsam.stabilizer import _test_xi as xi


# Vectorised version of the above, so we can use implicit broadcasting and stuff
vec_fixed_point_finder = np.vectorize(fixed_point_finder, [np.float64])
vec_chi_to_gauss = np.vectorize(chi_to_gauss, [np.float64])
vec_xi = np.vectorize(xi, [np.float64])
vec_root_finder = np.vectorize(root_finder, [np.float64])


def stabilization(data, m_hat, sigma, N, mask=None, clip_eta=True, return_eta=False, n_cores=None, mp_method=None):

    data = np.asarray(data)
    m_hat = np.asarray(m_hat)
    sigma = np.asarray(sigma)

    if mask is None:
        mask = np.ones(data.shape[:-1], dtype=np.bool)
    else:
        mask = np.asarray(mask, dtype=np.bool)

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
                N,
                clip_eta)
               for idx in range(data.shape[-2]))

    parallel_stabilization = multiprocesser(multiprocess_stabilization, n_cores=n_cores, mp_method=mp_method)
    output = parallel_stabilization(arglist)

    data_stabilized = np.zeros_like(data, dtype=np.float32)
    eta = np.zeros_like(data, dtype=np.float32)

    for idx, content in enumerate(output):
        data_stabilized[..., idx, :] = content[0]
        eta[..., idx, :] = content[1]

    if return_eta:
        return data_stabilized, eta
    return data_stabilized


def multiprocess_stabilization(data, m_hat, mask, sigma, N, clip_eta=True):
    """Helper function for multiprocessing the stabilization part."""

    if mask.ndim == (sigma.ndim - 1):
        mask = mask[..., None]

    mask = np.logical_and(sigma > 0, mask)
    out = np.zeros_like(data, dtype=np.float64)
    eta = np.zeros_like(data, dtype=np.float64)

    eta[mask] = vec_fixed_point_finder(m_hat[mask], sigma[mask], N, clip_eta)
    out[mask] = vec_chi_to_gauss(data[mask], eta[mask], sigma[mask], N)

    return out, eta


def corrected_sigma(eta, sigma, N, mask=None):
    """Compute the local corrected standard deviation for the adaptive nonlocal
    means according to the correction factor xi.

    Input
    --------
    eta : double
        Signal intensity
    sigma : double
        Noise magnitude standard deviation
    N : int
        Number of coils of the acquisition (N=1 for Rician noise)
    mask : ndarray, optional
        Compute only the corrected sigma value inside the mask.

    Return
    --------
    output, ndarray
        Corrected sigma value, where sigma_gaussian = sigma / sqrt(xi)
    """
    eta = np.array(eta)
    sigma = np.array(sigma)

    if mask is None:
        mask = np.ones_like(sigma, dtype=np.bool)
    else:
        mask = np.array(mask, dtype=np.bool)

    # Force 3D/4D broadcasting if needed
    if sigma.ndim == (eta.ndim - 1):
        sigma = np.broadcast_to(sigma[..., None], eta.shape)

    if mask.ndim == (sigma.ndim - 1):
        mask = np.broadcast_to(mask[..., None], sigma.shape)

    output = np.zeros_like(eta, dtype=np.float32)
    output[mask] = sigma[mask] / np.sqrt(vec_xi(eta[mask], sigma[mask], N))

    return output


def root_finder_sigma(data, sigma, N, mask=None):

    data = np.array(data)
    sigma = np.array(sigma)

    if mask is None:
        mask = np.ones_like(sigma, dtype=np.bool)
    else:
        mask = np.array(mask, dtype=np.bool)

    # Force 3D/4D broadcasting if needed
    if sigma.ndim == (data.ndim - 1):
        sigma = np.broadcast_to(sigma[..., None], data.shape)

    if mask.ndim == (sigma.ndim - 1):
        mask = np.broadcast_to(mask[..., None], sigma.shape)

    corrected_sigma = np.zeros_like(data, dtype=np.float32)
    gaussian_SNR = np.zeros_like(data, dtype=np.float32)
    theta = np.zeros_like(data, dtype=np.float32)

    theta[mask] = data[mask] / sigma[mask]
    gaussian_SNR[mask] = sigma[mask] / np.sqrt(vec_root_finder(theta[mask], N))
    corrected_sigma[mask] = sigma[mask] / np.sqrt(vec_xi(gaussian_SNR[mask], 1, N))

    return corrected_sigma
