from __future__ import division, print_function

import numpy as np
import warnings
import logging

from time import time
from itertools import repeat, product
from multiprocessing import Pool

from nlsam.utils import im2col_nd, col2im_nd
from nlsam.angular_tools import angular_neighbors

from scipy.sparse import lil_matrix

warnings.simplefilter("ignore", category=FutureWarning)

try:
    import spams
except ImportError:
    raise ImportError("Couldn't find spams library, is the package correctly installed?")


def nlsam_denoise(data, sigma, bvals, bvecs, block_size,
                  mask=None, is_symmetric=False, rejection=None, n_cores=None,
                  subsample=True, n_iter=10, b0_threshold=10, verbose=False):
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
    rejection : tuple, default None
        List of indexes to discard from the training set.
        b0s images will be completely discarded from the image and replaced with the mean b0s:
        DWIs will still be reconstructed, so this is useful for excluding datasets
        heavily corrupted by artifacts if they affect the whole reconstructed data.
    n_cores : int, default None
        Number of processes to use for the denoising. Default is to use
        all available cores.
    subsample : bool, default True
        If True, find the smallest subset of indices required to process each
        dwi at least once.
    n_iter : int, default 10
        Maximum number of iterations for the reweighted l1 solver.
    b0_threshold : int, default 10
        A b-value below b0_threshold will be considered as a b0 image.

    Output
    -----------
    data_denoised : ndarray
        The denoised dataset
    """

    logger = logging.getLogger('nlsam')

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

    b0_loc = tuple(np.where(bvals <= b0_threshold)[0])
    num_b0s = len(b0_loc)
    variance = sigma**2
    orig_shape = data.shape

    logger.info("Found {} b0s at position {}".format(str(num_b0s), str(b0_loc)))

    if rejection is not None:
        # Rejection happens later, but the indices are converted without b0s, so this is the actual user input
        logger.info("Volumes {} will be excluded from the training set.".format(str(rejection)))

        bad_b0s = tuple()

        for r in rejection:
            if r in b0_loc:
                logger.info("b0 {} will be excluded as a whole".format(str(r)))
                bad_b0s += (r,)
                num_b0s -= 1

        if num_b0s == 0:
            raise ValueError('It seems like all b0s {} have been excluded from the rejection set {}'.format(str(b0_loc), str(rejection)))

    # Average multiple b0s, and just use the average for the rest of the script
    # patching them in at the end
    if num_b0s > 1:

        if any(bad_b0s):
            good_b0s = tuple(x for x in b0_loc if x not in bad_b0s)
        else:
            good_b0s = b0_loc

        mean_b0 = np.mean(data[..., good_b0s], axis=-1)
        dwis = tuple(np.where(bvals > b0_threshold)[0])
        data = data[..., dwis]
        bvals = np.take(bvals, dwis, axis=0)
        bvecs = np.take(bvecs, dwis, axis=0)

        rest_of_b0s = b0_loc[1:]
        b0_loc = b0_loc[0]

        data = np.insert(data, b0_loc, mean_b0, axis=-1)
        bvals = np.insert(bvals, b0_loc, [0.], axis=0)
        bvecs = np.insert(bvecs, b0_loc, [0., 0., 0.], axis=0)
        b0_loc = tuple([b0_loc])
        num_b0s = 1
    else:
        rest_of_b0s = None

    # We need to shift the indexes for rejection by 1 for each b0s we removed which are located afterwards
    if rejection is not None:
        rejection = np.array(rejection)
        rejection = np.where(b0_loc < rejection, rejection - 1, rejection)

        if rest_of_b0s is not None:
            for loc in rest_of_b0s:
                rejection = np.where(loc < rejection, rejection - 1, rejection)

    # Double bvecs to find neighbors with assumed symmetry if needed
    if is_symmetric:
        logger.info('Data is assumed to be already symmetrized.')
        sym_bvecs = np.delete(bvecs, b0_loc, axis=0)
    else:
        sym_bvecs = np.vstack((np.delete(bvecs, b0_loc, axis=0), np.delete(-bvecs, b0_loc, axis=0)))

    neighbors = (angular_neighbors(sym_bvecs, block_size[-1] - num_b0s) % (data.shape[-1] - num_b0s))[:data.shape[-1] - num_b0s]

    # Full overlap for dictionary learning
    overlap = np.array(block_size, dtype=np.int16) - 1
    b0 = np.squeeze(data[..., b0_loc])
    data = np.delete(data, b0_loc, axis=-1)

    indexes = []
    for i in range(len(neighbors)):
        indexes += [(i,) + tuple(neighbors[i])]

    if subsample:
        indexes = greedy_set_finder(indexes)

    if rejection is not None:
        indexes, to_reject = reject_from_training(indexes, rejection)
    else:
        to_reject = np.zeros(len(indexes), dtype=np.bool)

    b0_block_size = tuple(block_size[:-1]) + ((block_size[-1] + num_b0s,))

    denoised_shape = data.shape[:-1] + (data.shape[-1] + num_b0s,)
    data_denoised = np.zeros(denoised_shape, np.float32)

    # Put all idx + b0 in this array in each iteration
    to_denoise = np.empty(data.shape[:-1] + (block_size[-1] + 1,), dtype=np.float64)

    # Solving parameters for D
    param_alpha = {}
    param_alpha['numThreads'] = 1  # This one is in the multiprocess loop, so we always use 1 core only
    param_alpha['pos'] = True
    param_alpha['mode'] = 1

    param_D = {}
    param_D['verbose'] = False
    param_D['posAlpha'] = True
    param_D['posD'] = True
    param_D['mode'] = 2
    param_D['lambda1'] = 1.2 / np.sqrt(np.prod(b0_block_size))
    param_D['K'] = int(2 * np.prod(b0_block_size))
    param_D['iter'] = 150
    param_D['batchsize'] = 500

    if n_cores is not None:
        param_D['numThreads'] = n_cores
    else:
        param_D['numThreads'] = -1

    for i, idx in enumerate(indexes):
        dwi_idx = tuple(np.where(tuple(idx) <= b0_loc, idx, np.array(idx) + num_b0s))
        to_denoise[..., 0] = b0
        to_denoise[..., 1:] = data[..., idx]

        logger.info('Now denoising volumes {} / block {} out of {}.'.format(dwi_idx, i + 1, len(indexes)))

        data_denoised[..., b0_loc + dwi_idx] += local_denoise(to_denoise,
                                                              b0_block_size,
                                                              overlap,
                                                              variance,
                                                              param_alpha,
                                                              param_D,
                                                              n_iter=n_iter,
                                                              reject=to_reject[i],
                                                              mask=mask,
                                                              dtype=np.float64,
                                                              n_cores=n_cores,
                                                              verbose=verbose)

    divider = np.bincount(np.array(indexes, dtype=np.int16).ravel())
    divider = np.insert(divider, b0_loc, len(indexes))

    data_denoised = data_denoised[:orig_shape[0],
                                  :orig_shape[1],
                                  :orig_shape[2],
                                  :orig_shape[3]] / divider

    # Put back the original number of b0s
    if rest_of_b0s is not None:

        b0_denoised = np.squeeze(data_denoised[..., b0_loc])
        data_denoised_insert = np.empty(orig_shape, dtype=np.float32)
        n = 0

        for i in range(orig_shape[-1]):
            if i in rest_of_b0s:
                data_denoised_insert[..., i] = b0_denoised
                n += 1
            else:
                data_denoised_insert[..., i] = data_denoised[..., i - n]

        data_denoised = data_denoised_insert

    return data_denoised


def local_denoise(data, block_size, overlap, variance, param_alpha, param_D,
                  n_iter=10, reject=False, mask=None, dtype=np.float64,
                  n_cores=None, verbose=False):

    logger = logging.getLogger('nlsam')

    if verbose:
        logger.setLevel(logging.INFO)

    if mask is None:
        mask = np.ones(data.shape[:-1], dtype=np.bool)

    # Do we train on this set or use D from the previous one?
    if reject:
        if 'D' not in param_alpha:
            raise ValueError('D is in not in param_alpha, but we are supposed to '
                             'skip training for this set.')
        # param_D['D'] = param_alpha['D']
        print('currently rejecting stuff')

    else:
        # no overlapping blocks for training
        no_over = (0, 0, 0, 0)
        X = im2col_nd(data, block_size, no_over)

        # Warm start from previous iteration
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

    time_multi = time()
    pool = Pool(processes=n_cores)

    arglist = [(data[:, :, k:k + block_size[2]],
                mask[:, :, k:k + block_size[2]],
                variance[:, :, k:k + block_size[2]],
                block_size_subset,
                overlap_subset,
                param_alpha_subset,
                param_D_subset,
                dtype_subset,
                n_iter_subset)
               for k, block_size_subset, overlap_subset, param_alpha_subset, param_D_subset, dtype_subset, n_iter_subset
               in zip(range(data.shape[2] - block_size[2] + 1),
                      repeat(block_size),
                      repeat(overlap),
                      repeat(param_alpha),
                      repeat(param_D),
                      repeat(dtype),
                      repeat(n_iter))]

    data_denoised = pool.map(processer, arglist)
    pool.close()
    pool.join()

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


def reject_from_training(indexes, rejection):
    """Puts the subsets from indexes specified in rejection at the end of indexes"""

    indexes = np.array(indexes)
    to_reject = np.zeros(len(indexes), dtype=np.bool)

    for r, i in product(rejection, range(len(indexes))):
        if r in indexes[i]:
            to_reject[i] = True

    sorted_args = np.argsort(to_reject)
    return indexes[sorted_args], to_reject[sorted_args]


def processer(arglist):
    data, mask, variance, block_size, overlap, param_alpha, param_D, dtype, n_iter = arglist
    return _processer(data, mask, variance, block_size, overlap, param_alpha, param_D, dtype=dtype, n_iter=n_iter)


def _processer(data, mask, variance, block_size, overlap, param_alpha, param_D,
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
    W = np.ones(alpha.shape, dtype=dtype)

    DtD = np.dot(D.T, D)
    DtX = np.dot(D.T, X)
    DtXW = np.zeros_like(DtX, order='F')
    DtDW = np.zeros_like(DtD, order='F')

    alpha_old = np.zeros(alpha.shape, dtype=dtype)
    not_converged = np.ones(alpha.shape[1], dtype=np.bool)
    arr = np.zeros(alpha.shape, dtype=dtype)
    nonzero_ind = np.zeros(alpha.shape, dtype=np.bool)

    xi = np.random.randn(X.shape[0], X.shape[1]) * var_mat
    eps = np.max(np.abs(np.dot(D.T, xi)), axis=0)

    for _ in range(n_iter):
        DtXW[:, not_converged] = DtX[:, not_converged] / W[:, not_converged]

        for i in range(X.shape[1]):
            if not_converged[i]:
                param_alpha['lambda1'] = var_mat[i] * (X.shape[0] + gamma * np.sqrt(2 * X.shape[0]))
                DtDW[:] = (1. / W[..., None, i]) * DtD * (1. / W[:, i])
                alpha[:, i:i + 1] = spams.lasso(X[:, i:i + 1], Q=DtDW, q=DtXW[:, i:i + 1], **param_alpha)

        alpha.toarray(out=arr)
        np.not_equal(arr, 0, out=nonzero_ind)
        arr[nonzero_ind] /= W[nonzero_ind]
        not_converged[:] = np.abs(alpha_old - arr).max(axis=0) > tolerance

        if not np.any(not_converged):
            break

        alpha_old[:] = arr
        W[:] = 1. / (np.abs(alpha_old**tau) + eps)

    weigths = np.ones(X_full_shape[1], dtype=dtype, order='F')
    weigths[train_idx] = 1. / (alpha.getnnz(axis=0) + 1.)

    X = np.zeros(X_full_shape, dtype=dtype, order='F')
    X[:, train_idx] = np.dot(D, arr)

    return col2im_nd(X, block_size, orig_shape, overlap, weigths)
