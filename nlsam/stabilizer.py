from __future__ import division

import numpy as np

from nlsam.multiprocess import multiprocesser
from nlsam._stabilizer import fixed_point_finder, chi_to_gauss


def stabilization(data, m_hat, mask, sigma, N, n_cores=None, mp_method=None, clip_eta=True, return_eta=False):

    if sigma.ndim == (data.ndim - 1):
        sigma = sigma[..., None]

    # Check all dims are ok
    if (data.shape != sigma.shape):
        raise ValueError('data shape {} is not compatible with sigma shape {}'.format(data.shape, sigma.shape))

    if (data.shape[:-1] != mask.shape):
        raise ValueError('data shape {} is not compatible with mask shape {}'.format(data.shape, mask.shape))

    if (data.shape != m_hat.shape):
        raise ValueError('data shape {} is not compatible with m_hat shape {}'.format(data.shape, m_hat.shape))

    arglist = [(data[..., idx, :],
                m_hat[..., idx, :],
                mask[..., idx],
                sigma[..., idx, :],
                N,
                clip_eta)
               for idx in range(data.shape[-2])]

    parallel_stabilization = multiprocesser(_multiprocess_stabilization, n_cores=n_cores, mp_method=mp_method)
    output = parallel_stabilization(arglist)

    data_stabilized = np.empty(data.shape, dtype=np.float32)
    eta = np.empty(data.shape, dtype=np.float32)

    for idx, content in enumerate(output):
        data_stabilized[..., idx, :] = content[idx][0]
        eta[..., idx, :] = content[idx][1]

    if return_eta:
        return data_stabilized, eta
    return data_stabilized


def _multiprocess_stabilization(args):
    return multiprocess_stabilization(*args)


def multiprocess_stabilization(data, m_hat, mask, sigma, N, clip_eta=True):
    """Helper function for multiprocessing the stabilization part."""

    mask = np.logical_and(sigma > 0, mask)
    eta = np.zeros_like(data, dtype=np.float64)
    out = np.zeros_like(data, dtype=np.float64)

    # vec_fixed_point_finder = np.frompyfunc(fixed_point_finder, 4, 1)
    # vec_chi_to_gauss = np.frompyfunc(chi_to_gauss, 4, 1)

    vec_fixed_point_finder = np.vectorize(fixed_point_finder, [np.float64])
    vec_chi_to_gauss = np.vectorize(chi_to_gauss, [np.float64])

    eta[mask] = vec_fixed_point_finder(m_hat[mask], sigma[mask], N, clip_eta)
    out[mask] = vec_chi_to_gauss(data[mask], eta[mask], sigma[mask], N)

    return out, eta
