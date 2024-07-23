import numpy as np
import logging

from nlsam.stabilizer import root_finder_loop, multiprocess_stabilization
from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm

logger = logging.getLogger('nlsam')


def stabilization(data, m_hat, sigma, N, mask=None, clip_eta=True, return_eta=False, n_cores=-1, verbose=False):

    data = np.asarray(data)
    m_hat = np.asarray(m_hat)
    sigma = np.atleast_3d(sigma).astype(np.float32)
    N = np.atleast_3d(N).astype(np.float32)

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
        raise ValueError(f'data shape {data.shape} is not compatible with sigma shape {sigma.shape}')

    if (data.shape[:-1] != mask.shape):
        raise ValueError(f'data shape {data.shape} is not compatible with mask shape {mask.shape}')

    if (data.shape != m_hat.shape):
        raise ValueError(f'data shape {data.shape} is not compatible with m_hat shape {m_hat.shape}')

    slicer = [np.index_exp[..., k] for k in range(data.shape[-1])]

    if verbose:
        slicer = tqdm(slicer)

    with Parallel(n_jobs=n_cores, prefer='threads') as parallel:
        output = parallel(delayed(multiprocess_stabilization)(data[current_slice],
                                                              m_hat[current_slice],
                                                              mask,
                                                              sigma[current_slice],
                                                              N[current_slice],
                                                              clip_eta) for current_slice in slicer)

    data_stabilized = np.zeros_like(data, dtype=np.float32)
    eta = np.zeros_like(data, dtype=np.float32)

    for idx, content in enumerate(output):
        data_stabilized[..., idx] = content[0]
        eta[..., idx] = content[1]

    if return_eta:
        return data_stabilized, eta
    return data_stabilized


def root_finder_sigma(data, sigma, N, mask=None, verbose=False, n_cores=-1):
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
    verbose : bool, optional
        displays a progress bar if True
    n_cores : int, optional
        number of cores to use for parallel processing

    Return
    --------
    output, ndarray
        Corrected sigma value, where sigma_gaussian = sigma / sqrt(xi)
    """
    data = np.array(data)
    sigma = np.array(sigma)
    N = np.array(N, dtype=np.float64)

    if mask is None:
        mask = np.ones(data.shape[:-1], dtype=bool)
    else:
        mask = np.array(mask, dtype=bool)

    # Force 3D/4D broadcasting if needed
    if sigma.ndim == (data.ndim - 1):
        sigma = np.broadcast_to(sigma[..., None], data.shape)

    if N.ndim < data.ndim:
        N = np.broadcast_to(N[..., None], data.shape)

    corrected_sigma = np.zeros_like(data, dtype=np.float32)

    # The mask is only 3D, so this will make a 1D array to loop through
    data = data[mask]
    sigma = sigma[mask]
    N = N[mask]

    slicer = [np.index_exp[..., k] for k in range(data.shape[-1])]

    if verbose:
        slicer = tqdm(slicer)

    with Parallel(n_jobs=n_cores, prefer='threads') as parallel:
        output = parallel(delayed(root_finder_loop)(data[current_slice],
                                                    sigma[current_slice],
                                                    N[current_slice]) for current_slice in slicer)

    for idx, content in enumerate(output):
        corrected_sigma[mask, idx] = content

    return corrected_sigma
