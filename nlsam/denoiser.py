from __future__ import division, print_function

import numpy as np
import warnings
import logging

from time import time
from itertools import cycle

from nlsam.utils import im2col_nd, col2im_nd
from nlsam.angular_tools import angular_neighbors
from nlsam.multiprocess import multiprocesser

from scipy.sparse import lil_matrix

try:
    import spams
    warnings.filterwarnings("ignore", category=FutureWarning, module='spams')
except ImportError:
    raise ImportError("Couldn't find spams library, is the package correctly installed?")

logger = logging.getLogger('nlsam')


def nlsam_denoise(data, sigma, bvals, bvecs, block_size,
                  mask=None, is_symmetric=False, n_cores=None, split_b0s=False,
                  subsample=True, n_iter=10, b0_threshold=10, verbose=False, mp_method=None):
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
    n_cores : int, default None
        Number of processes to use for the denoising. Default is to use
        all available cores.
    split_b0s : bool, default False
        If True and the dataset contains multiple b0s, a different b0 will be used for
        each run of the denoising. If False, the b0s are averaged and the average b0 is used instead.
    subsample : bool, default True
        If True, find the smallest subset of indices required to process each
        dwi at least once.
    n_iter : int, default 10
        Maximum number of iterations for the reweighted l1 solver.
    b0_threshold : int, default 10
        A b-value below b0_threshold will be considered as a b0 image.
    verbose : bool, default False
        print useful messages.
    mp_method : string
        Dispatch method for multiprocessing,

    Output
    -----------
    data_denoised : ndarray
        The denoised dataset
    """

    if verbose:
        logger.setLevel(logging.INFO)

    if mask is None:
        mask = np.ones(data.shape[:-1], dtype=np.bool)

    if data.shape[:-1] != mask.shape:
        raise ValueError('data shape is {}, but mask shape {} is different!'.format(data.shape, mask.shape))

    if data.shape[:-1] != sigma.shape:
        raise ValueError('data shape is {}, but sigma shape {} is different!'.format(data.shape, sigma.shape))

    if len(block_size) != len(data.shape):
        raise ValueError('Block shape {} and data shape {} are not of the same '
                         'length'.format(data.shape, block_size.shape))

    b0_loc = np.where(bvals <= b0_threshold)[0]
    dwis = np.where(bvals > b0_threshold)[0]
    num_b0s = len(b0_loc)
    variance = sigma**2

    logger.info("Found {} b0s at position {}".format(str(num_b0s), str(b0_loc)))

    # Average all b0s if we don't split them in the training set
    if num_b0s > 1 and not split_b0s:
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

    indexes = [(dwi,) + tuple(neighbors[dwi]) for dwi in range(data.shape[-1]) if dwi in dwis]

    if subsample:
        indexes = greedy_set_finder(indexes)

    b0_block_size = tuple(block_size[:-1]) + ((block_size[-1] + 1,))
    data_denoised = np.zeros(data.shape, np.float32)
    divider = np.zeros(data.shape[-1])

    # Put all idx + b0 in this array in each iteration
    to_denoise = np.empty(data.shape[:-1] + (block_size[-1] + 1,), dtype=np.float64)

    for i, idx in enumerate(indexes):
        logger.info('Now denoising volumes {} / block {} out of {}.'.format(idx, i + 1, len(indexes)))

        b0_loc = tuple((next(split_b0s_idx),))
        to_denoise[..., 0] = data[..., b0_loc].squeeze()
        to_denoise[..., 1:] = data[..., idx]
        divider[list(b0_loc + idx)] += 1

        data_denoised[..., b0_loc + idx] += local_denoise(to_denoise,
                                                          b0_block_size,
                                                          overlap,
                                                          variance,
                                                          n_iter=n_iter,
                                                          mask=mask,
                                                          dtype=np.float64,
                                                          n_cores=n_cores,
                                                          verbose=verbose,
                                                          mp_method=mp_method)

    data_denoised /= divider
    return data_denoised


def local_denoise(data, block_size, overlap, variance, n_iter=10, mask=None,
                  dtype=np.float64, n_cores=None, verbose=False, mp_method=None):
    if verbose:
        logger.setLevel(logging.INFO)

    if mask is None:
        mask = np.ones(data.shape[:-1], dtype=np.bool)

    # no overlapping blocks for training
    no_over = (0, 0, 0, 0)
    X = im2col_nd(data, block_size, no_over)

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

    if 'D' in param_alpha:
        param_D['D'] = param_alpha['D']

    mask_col = im2col_nd(np.broadcast_to(mask[..., None], data.shape), block_size, no_over)
    train_idx = np.sum(mask_col, axis=0) > (mask_col.shape[0] / 2.)

    train_data = X[:, train_idx]
    train_data = np.asfortranarray(train_data[:, np.any(train_data != 0, axis=0)], dtype=dtype)
    train_data /= np.sqrt(np.sum(train_data**2, axis=0, keepdims=True), dtype=dtype)

    param_alpha['D'] = spams.trainDL(train_data, **param_D)
    param_alpha['D'] /= np.sqrt(np.sum(param_alpha['D']**2, axis=0, keepdims=True, dtype=dtype))
    param_D['D'] = param_alpha['D']

    del train_data, X

    param_alpha['numThreads'] = 1
    param_D['numThreads'] = 1

    arglist = [(data[:, :, k:k + block_size[2]],
                mask[:, :, k:k + block_size[2]],
                variance[:, :, k:k + block_size[2]],
                block_size,
                overlap,
                param_alpha,
                param_D,
                dtype,
                n_iter)
               for k in range(data.shape[2] - block_size[2] + 1)]

    time_multi = time()
    parallel_processer = multiprocesser(_processer, n_cores=n_cores, mp_method=mp_method)
    data_denoised = parallel_processer(arglist)
    logger.info('Multiprocessing done in {0:.2f} mins.'.format((time() - time_multi) / 60.))

    # Put together the multiprocessed results
    data_subset = np.zeros_like(data, dtype=np.float32)
    divider = np.zeros_like(data, dtype=np.int16)
    ones = np.ones_like(data_denoised[0], dtype=np.int16)

    for k in range(len(data_denoised)):
        data_subset[:, :, k:k + block_size[2]] += data_denoised[k]
        divider[:, :, k:k + block_size[2]] += ones

    data_subset /= divider
    return data_subset


def greedy_set_finder(sets):
    """Returns a list of subsets that spans the input sets with a greedy algorithm
    http://en.wikipedia.org/wiki/Set_cover_problem#Greedy_algorithm"""

    sets = [set(s) for s in sets]
    universe = set()

    for s in sets:
        universe = universe.union(s)

    output = []

    while len(universe) != 0:

        max_intersect = 0

        for i, s in enumerate(sets):

            n_intersect = len(s.intersection(universe))

            if n_intersect > max_intersect:
                max_intersect = n_intersect
                element = i

        output.append(tuple(sets[element]))
        universe = universe.difference(sets[element])

    return output


def _processer(args):
    return processer(*args)


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
    X = X[:, train_idx]

    param_alpha['L'] = int(0.5 * X.shape[0])

    D = param_alpha['D']

    alpha = lil_matrix((D.shape[1], X.shape[1]))
    W = np.ones(alpha.shape, dtype=dtype, order='F')

    DtD = np.dot(D.T, D)
    DtX = np.dot(D.T, X)
    DtXW = np.empty_like(DtX, order='F')

    alpha_old = np.ones(alpha.shape, dtype=dtype)
    has_converged = np.zeros(alpha.shape[1], dtype=np.bool)
    arr = np.empty(alpha.shape)

    xi = np.random.randn(X.shape[0], X.shape[1]) * var_mat
    eps = np.max(np.abs(np.dot(D.T, xi)), axis=0)

    for _ in range(n_iter):
        not_converged = np.equal(has_converged, False)
        DtXW[:, not_converged] = DtX[:, not_converged] / W[:, not_converged]

        for i in range(alpha.shape[1]):
            if not has_converged[i]:
                param_alpha['lambda1'] = var_mat[i] * (X.shape[0] + gamma * np.sqrt(2 * X.shape[0]))
                DtDW = (1. / W[..., None, i]) * DtD * (1. / W[:, i])
                alpha[:, i:i + 1] = spams.lasso(X[:, i:i + 1], Q=np.asfortranarray(DtDW), q=DtXW[:, i:i + 1], **param_alpha)

        alpha.toarray(out=arr)
        nonzero_ind = arr != 0
        arr[nonzero_ind] /= W[nonzero_ind]
        has_converged = np.max(np.abs(alpha_old - arr), axis=0) < tolerance

        if np.all(has_converged):
            break

        alpha_old[:] = arr
        W[:] = 1. / (np.abs(alpha_old**tau) + eps)

    weigths = np.ones(X_full_shape[1], dtype=dtype, order='F')
    weigths[train_idx] = 1. / (alpha.getnnz(axis=0) + 1.)

    X = np.zeros(X_full_shape, dtype=dtype, order='F')
    X[:, train_idx] = np.dot(D, arr)

    return col2im_nd(X, block_size, orig_shape, overlap, weigths)
