import numpy as np
import logging

from time import time
from itertools import cycle

from nlsam.utils import im2col_nd, col2im_nd
from nlsam.angular_tools import angular_neighbors, split_shell, greedy_set_finder
from autodmri.blocks import extract_patches

from scipy.sparse import lil_matrix
from joblib import Parallel, delayed

import spams

logger = logging.getLogger('nlsam')


def nlsam_denoise(data, sigma, bvals, bvecs, block_size,
                  mask=None, is_symmetric=False, n_cores=-1, split_b0s=False, split_shell=False,
                  subsample=True, n_iter=10, b0_threshold=10, dtype=np.float64, verbose=False):
    """Main nlsam denoising function which sets up everything nicely for the local
    block denoising.

    Input
    -----------
    data : ndarray
        Input volume to denoise.
    sigma : ndarray
        Noise standard deviation estimation at each voxel.
        Converted to variance internally.
    bvals : 1D array
        the N b-values associated to each of the N diffusion volume.
    bvecs : N x 3 2D array
        the N 3D vectors for each acquired diffusion gradients.
    block_size : tuple, length = data.ndim
        Patch size + number of angular neighbors to process at once as similar data.

    Optional parameters
    -------------------
    mask : ndarray, default None
        Restrict computations to voxels inside the mask to reduce runtime.
    is_symmetric : bool, default False
        If True, assumes that for each coordinate (x, y, z) in bvecs,
        (-x, -y, -z) was also acquired.
    n_cores : int, default -1
        Number of processes to use for the denoising. Default is to use
        all available cores.
    split_b0s : bool, default False
        If True and the dataset contains multiple b0s, a different b0 will be used for
        each run of the denoising. If False, the b0s are averaged and the average b0 is used instead.
    split_shell : bool, default False
        If True and the dataset contains multiple b-values, each shell is processed independently.
        If False, all the data is used at the same time for computing angular neighbors.
    subsample : bool, default True
        If True, find the smallest subset of indices required to process each
        dwi at least once.
    n_iter : int, default 10
        Maximum number of iterations for the reweighted l1 solver.
    b0_threshold : int, default 10
        A b-value below b0_threshold will be considered as a b0 image.
    dtype : np.float32 or np.float64, default np.float64
        Precision to use for inner computations. Note that np.float32 should only be used for
        very, very large datasets (that is, your ram starts swapping) as it can lead to numerical precision errors.
    verbose : bool, default False
        print useful messages.

    Output
    -----------
    data_denoised : ndarray
        The denoised dataset
    """

    if verbose:
        logger.setLevel(logging.INFO)

    if mask is None:
        mask = np.ones(data.shape[:-1], dtype=bool)

    if data.shape[:-1] != mask.shape:
        raise ValueError(f'data shape is {data.shape}, but mask shape {mask.shape} is different!')

    if data.shape[:-1] != sigma.shape:
        raise ValueError(f'data shape is {data.shape}, but sigma shape {sigma.shape} is different!')

    if len(block_size) != len(data.shape):
        raise ValueError(f'Block shape {data.shape} and data shape {block_size.shape} are not of the same length')

    if not ((dtype == np.float32) or (dtype == np.float64)):
        raise ValueError(f'dtype should be either np.float32 or np.float64, but is {dtype}')

    b0_loc = np.where(bvals <= b0_threshold)[0]
    dwis = np.where(bvals > b0_threshold)[0]
    num_b0s = len(b0_loc)
    variance = sigma**2

    # We also convert bvecs associated with b0s to exactly (0,0,0), which
    # is not always the case when we hack around with the scanner.
    bvecs = np.where(bvals[:, None] <= b0_threshold, 0, bvecs)

    logger.info(f"Found {num_b0s} b0s at position {b0_loc}")

    # Average all b0s if we don't split them in the training set
    if num_b0s > 1 and not split_b0s:
        num_b0s = 1
        data[..., b0_loc] = np.mean(data[..., b0_loc], axis=-1, keepdims=True)

    # Split the b0s in a cyclic fashion along the training data
    # If we only had one, cycle just return b0_loc indefinitely,
    # else we go through all indexes.
    np.random.shuffle(b0_loc)
    split_b0s_idx = cycle(b0_loc)

    # Double bvecs to find neighbors with assumed symmetry if needed
    if is_symmetric:
        logger.info('Data is assumed to be already symmetric.')
        sym_bvecs = bvecs
    else:
        sym_bvecs = np.vstack((bvecs, -bvecs))

    neighbors = angular_neighbors(sym_bvecs, block_size[-1] - 1) % data.shape[-1]
    neighbors = neighbors[:data.shape[-1]]  # everything was doubled for symmetry

    # Full overlap for dictionary learning
    overlap = np.array(block_size, dtype=np.int16) - 1

    full_indexes = [(dwi,) + tuple(neighbors[dwi]) for dwi in range(data.shape[-1]) if dwi in dwis]

    if subsample:
        indexes = greedy_set_finder(full_indexes)
    else:
        indexes = full_indexes

    # If we have more b0s than indexes, then we have to add a few more blocks since
    # we won't do a full cycle. If we have more b0s than indexes after that, then it breaks.
    if num_b0s > len(indexes):
        the_rest = [rest for rest in full_indexes if rest not in indexes]
        indexes += the_rest[:(num_b0s - len(indexes))]

    if num_b0s > len(indexes):
        error = (f'Seems like you still have more b0s {num_b0s} than available blocks {len(indexes)},'
                 ' either average them or deactivate subsampling.')
        raise ValueError(error)

    b0_block_size = tuple(block_size[:-1]) + ((block_size[-1] + 1,))
    data_denoised = np.zeros(data.shape, np.float32)
    divider = np.zeros(data.shape[-1])

    # Put all idx + b0 in this array in each iteration
    to_denoise = np.empty(data.shape[:-1] + (block_size[-1] + 1,), dtype=dtype)

    for i, idx in enumerate(indexes, start=1):
        b0_loc = tuple((next(split_b0s_idx),))
        to_denoise[..., 0] = data[..., b0_loc].squeeze()
        to_denoise[..., 1:] = data[..., idx]
        divider[list(b0_loc + idx)] += 1

        logger.info(f'Now denoising volumes {b0_loc + idx} / block {i} out of {len(indexes)}.')

        data_denoised[..., b0_loc + idx] += local_denoise(to_denoise,
                                                          b0_block_size,
                                                          overlap,
                                                          variance,
                                                          n_iter=n_iter,
                                                          mask=mask,
                                                          dtype=dtype,
                                                          n_cores=n_cores,
                                                          verbose=verbose)

    data_denoised /= divider
    return data_denoised


def local_denoise(data, block_size, overlap, variance, n_iter=10, mask=None,
                  dtype=np.float64, n_cores=-1, verbose=False):
    if verbose:
        logger.setLevel(logging.INFO)

    if mask is None:
        mask = np.ones(data.shape[:-1], dtype=bool)

    X = extract_patches(data, block_size, [1, 1, 1, block_size[-1]]).reshape(-1, np.prod(block_size)).T

    # Solving for D
    param_alpha = {}
    param_alpha['pos'] = True
    param_alpha['mode'] = 1

    param_D = {}
    param_D['verbose'] = False
    param_D['posAlpha'] = True
    param_D['posD'] = True
    param_D['mode'] = 2
    param_D['lambda1'] = 1.2 / np.sqrt(np.prod(block_size))
    param_D['K'] = int(2 * np.prod(block_size))
    param_D['iter'] = 150
    param_D['batchsize'] = 500
    param_D['numThreads'] = n_cores

    if 'D' in param_alpha:
        param_D['D'] = param_alpha['D']

    mask_col = extract_patches(mask, block_size[:-1], (1, 1, 1), flatten=False)
    axis = tuple(range(mask_col.ndim//2, mask_col.ndim))
    train_idx = np.sum(mask_col, axis=axis).ravel() > (np.prod(block_size[:-1]) / 2.)

    train_data = np.asfortranarray(X[:, train_idx])
    train_data /= np.sqrt(np.sum(train_data**2, axis=0, keepdims=True), dtype=dtype)

    param_alpha['D'] = spams.trainDL(train_data, **param_D)
    param_alpha['D'] /= np.sqrt(np.sum(param_alpha['D']**2, axis=0, keepdims=True, dtype=dtype))
    param_D['D'] = param_alpha['D']

    del train_idx, train_data, X, mask_col

    param_alpha['numThreads'] = 1
    param_D['numThreads'] = 1

    arglist = ((data[:, :, k:k + block_size[2]],
                mask[:, :, k:k + block_size[2]],
                variance[:, :, k:k + block_size[2]],
                block_size,
                overlap,
                param_alpha,
                param_D,
                dtype,
                n_iter)
               for k in range(data.shape[2] - block_size[2] + 1))

    time_multi = time()
    data_denoised = Parallel(n_jobs=n_cores,
                             verbose=verbose)(delayed(processer)(*args) for args in arglist)
    logger.info(f'Multiprocessing done in {(time() - time_multi) / 60} mins.')

    # Put together the multiprocessed results
    data_subset = np.zeros_like(data, dtype=np.float32)
    divider = np.zeros_like(data, dtype=np.int16)

    for k, content in enumerate(data_denoised):
        data_subset[:, :, k:k + block_size[2]] += content
        divider[:, :, k:k + block_size[2]] += 1

    data_subset /= divider
    return data_subset


def processer(data, mask, variance, block_size, overlap, param_alpha, param_D,
              dtype=np.float64, n_iter=10, gamma=3., tau=1., tolerance=1e-5):

    orig_shape = data.shape
    mask_array = im2col_nd(mask, block_size[:-1], overlap[:-1])
    train_idx = np.sum(mask_array, axis=0) > (mask_array.shape[0] / 2.)

    # If mask is empty, return a bunch of zeros as blocks
    if not np.any(train_idx):
        return np.zeros_like(data)

    X = im2col_nd(data, block_size, overlap)
    var_mat = np.median(im2col_nd(variance, block_size[:-1], overlap[:-1])[:, train_idx], axis=0)
    X_full_shape = X.shape
    X = X[:, train_idx].astype(dtype)

    param_alpha['L'] = int(0.5 * X.shape[0])

    D = param_alpha['D']

    alpha = lil_matrix((D.shape[1], X.shape[1]))
    W = np.ones(alpha.shape, dtype=dtype)

    DtD = np.asfortranarray(np.dot(D.T, D))
    DtX = np.dot(D.T, X)
    DtXW = np.empty_like(DtX, order='F')
    DtDW = np.empty_like(DtD, order='F')

    alpha_old = np.ones(alpha.shape, dtype=dtype)
    not_converged = np.ones(alpha.shape[1], dtype=bool)
    nonzero_ind = np.zeros(alpha.shape, dtype=bool)
    arr = np.empty(alpha.shape)

    xi = np.random.randn(X.shape[0], X.shape[1]) * var_mat
    var_mat *= (X.shape[0] + gamma * np.sqrt(2 * X.shape[0]))
    eps = np.max(np.abs(np.dot(D.T, xi)), axis=0)

    for _ in range(n_iter):
        DtXW[:, not_converged] = DtX[:, not_converged] / W[:, not_converged]

        for i in range(alpha.shape[1]):
            if not_converged[i]:
                param_alpha['lambda1'] = var_mat[i]
                DtDW[:] = (1. / W[..., None, i]) * DtD * (1. / W[:, i])
                alpha[:, i:i + 1] = spams.lasso(X[:, i:i + 1], Q=DtDW, q=DtXW[:, i:i + 1], **param_alpha)

        alpha.toarray(out=arr)
        nonzero_ind[:] = arr != 0
        arr[nonzero_ind] /= W[nonzero_ind]
        not_converged[:] = np.max(np.abs(alpha_old - arr), axis=0) > tolerance

        if not np.any(not_converged):
            break

        alpha_old[:] = arr
        W[:] = 1. / (np.abs(alpha_old**tau) + eps)

    weigths = np.ones(X_full_shape[1], dtype=dtype, order='F')
    weigths[train_idx] = 1. / (alpha.getnnz(axis=0) + 1.)

    X = np.zeros(X_full_shape, dtype=dtype, order='F')
    X[:, train_idx] = np.dot(D, arr)

    return col2im_nd(X, block_size, orig_shape, overlap, weigths)
