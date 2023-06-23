import numpy as np
import logging

# from nlsam.stabilizer import fixed_point_finder, chi_to_gauss, root_finder, xi
from nlsam.stabilizer import root_finder_loop, multiprocess_stabilization
from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm

logger = logging.getLogger('nlsam')

# Vectorised versions of the above, so we can use implicit broadcasting and stuff
# vec_fixed_point_finder = np.vectorize(fixed_point_finder, [np.float64])
# vec_chi_to_gauss = np.vectorize(chi_to_gauss, [np.float64])
# vec_xi = np.vectorize(xi, [np.float64])
# vec_root_finder = np.vectorize(root_finder, [np.float64])


def stabilization(data, m_hat, sigma, N, mask=None, clip_eta=True, return_eta=False, n_cores=-1, verbose=False):

    data = np.asarray(data)
    m_hat = np.asarray(m_hat)
    sigma = np.atleast_3d(sigma.astype(np.float32))
    N = np.atleast_3d(N.astype(np.float32))

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


# def multiprocess_stabilization(data, m_hat, mask, sigma, N, clip_eta, current_slice):
#     """Helper function for multiprocessing the stabilization part."""
#     data = data[current_slice]
#     m_hat = m_hat[current_slice]
#     mask = mask[current_slice]
#     sigma = sigma[current_slice]
#     N = N[current_slice]

#     if mask.ndim == (sigma.ndim - 1):
#         mask = mask[..., None]

#     mask = np.logical_and(sigma > 0, mask)
#     out = np.zeros_like(data, dtype=np.float32)
#     eta = np.zeros_like(data, dtype=np.float32)

#     eta[mask] = vec_fixed_point_finder(m_hat[mask], sigma[mask], N[mask], clip_eta=clip_eta)
#     out[mask] = vec_chi_to_gauss(data[mask], eta[mask], sigma[mask], N[mask], use_nan=False)

#     return out, eta


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
    N = np.array(N)

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

    # for idx in ranger:
    #     theta[:] = data[:, idx] / sigma[:, idx]
    #     gaussian_SNR[:] = vec_root_finder(theta, N[:, idx])
    #     corrected_sigma[mask, idx] = sigma[:, idx] / np.sqrt(vec_xi(gaussian_SNR, 1, N[:, idx]))


    for idx, content in enumerate(output):
        corrected_sigma[mask, idx] = content

    return corrected_sigma
