from __future__ import division

import numpy as np
from nlsam._stabilizer import fixed_point_finder, chi_to_gauss


def stabilization(data, m_hat, mask, sigma, N, n_cores=None, mp_method=None, clip_eta=True):
    last_dim = len(data.shape) - 1
    # Check all dims are ok
    if (data.shape != sigma.shape):
      raise ValueError('data shape {} is not compatible with sigma shape {}'.format(data.shape, sigma.shape))

    if (data.shape[:last_dim] != mask.shape):
      raise ValueError('data shape {} is not compatible with mask shape {}'.format(data.shape, mask.shape))

    if (data.shape != m_hat.shape):
      raise ValueError('data shape {} is not compatible with m_hat shape {}'.format(data.shape, m_hat.shape))

    size = data.shape[last_dim - 1]
    mask = np.broadcast_to(mask[..., None], data.shape)
    arglist = [(data[..., idx, :],
              m_hat[..., idx, :],
              mask[..., idx, :],
              sigma[..., idx, :],
              N,
              clip_eta)
             for idx in range(size)]

    parallel_stabilization = multiprocesser(_multiprocess_stabilization, n_cores=n_cores, mp_method=mp_method)
    data_out = parallel_stabilization(arglist)
    data_stabilized = np.empty(data.shape, dtype=np.float32)

    for idx in range(len(data_out)):
      data_stabilized[..., idx, :] = data_out[idx]

    return data_stabilized


def _multiprocess_stabilization(args):
    return multiprocess_stabilization(*args)


def multiprocess_stabilization(data, m_hat, mask, sigma, N, clip_eta=True, return_eta=False):
    """Helper function for multiprocessing the stabilization part."""

    data = data.astype(np.float64)
    m_hat = m_hat.astype(np.float64)
    sigma = sigma.astype(np.float64)
    N = int(N)

    out = np.zeros(data.shape, dtype=np.float32)
    np.logical_and(sigma > 0, mask, out=mask)

    for idx in np.ndindex(data.shape):
        if mask[idx]:
            eta = fixed_point_finder(m_hat[idx], sigma[idx], N, clip_eta)
            out[idx] = chi_to_gauss(data[idx], eta, sigma[idx], N)

    if return_eta:
        return out, eta
    return out
