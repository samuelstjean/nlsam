import numpy as np
import logging

from time import time
from itertools import cycle

from nlsam.utils import im2col_nd, col2im_nd
from nlsam.angular_tools import angular_neighbors, split_per_shell, greedy_set_finder
from nlsam.path_stuff import path_stuff
from autodmri.blocks import extract_patches

from itertools import pairwise
from joblib import Parallel, delayed
from tqdm import tqdm

import spams
import scipy.sparse as ssp

logger = logging.getLogger('nlsam')

def nlsam_denoise(data, sigma, bvals, bvecs, block_size,
                  mask=None, is_symmetric=False, n_cores=-1, split_b0s=False, split_shell=False,
                  subsample=True, n_iter=10, b0_threshold=10, bval_threshold=25, dtype=np.float64, verbose=False):
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
        the N bvalues associated to each of the N diffusion volume.
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
        If True and the dataset contains multiple bvalues, each shell is processed independently.
        If False, all the data is used at the same time for computing angular neighbors.
    subsample : bool, default True
        If True, find the smallest subset of indices required to process each
        dwi at least once.
    n_iter : int, default 10
        Maximum number of iterations for the reweighted l1 solver.
    b0_threshold : int, default 10
        A bvalue below b0_threshold will be considered as a b0 image.
    bval_threshold : int, default 25
        Any bvalue within += bval_threshold of each others will be considered on the same shell (e.g. b=990 and b=1000 are on the same shell).
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
    angular_size = block_size[-1]

    # We also convert bvecs associated with b0s to exactly (0,0,0), which
    # is not always the case when we hack around with the scanner.
    bvecs = np.where(bvals[:, None] <= b0_threshold, 0, bvecs)

    logger.info(f"Found {num_b0s} b0s at position {b0_loc}")

    # Average all b0s if we don't split them in the training set
    if num_b0s > 1 and not split_b0s:
        num_b0s = 1
        data[..., b0_loc] = np.mean(data[..., b0_loc], axis=-1, keepdims=True)
        average_b0s = True
    else:
        average_b0s = False

    # Split the b0s in a cyclic fashion along the training data
    # If we only had one, cycle just return b0_loc indefinitely, else we go through all indexes.
    np.random.shuffle(b0_loc)
    split_b0s_idx = cycle(b0_loc)

    # Double bvecs to find neighbors with assumed symmetry if needed
    if is_symmetric:
        logger.info('Data is assumed to be already symmetric.')
        sym_bvecs = bvecs
    else:
        sym_bvecs = np.vstack((bvecs, -bvecs))

    if split_shell:
        logger.info('Data will be split in neighborhoods for each shells separately.')
        neighbors = split_per_shell(bvals, bvecs, angular_size, dwis, is_symmetric=is_symmetric, bval_threshold=bval_threshold)
        # print(type(neighbors))
        if angular_size >= len(bvals):
            all_data = True
            # print(list(b0_loc))
            neighbors.insert(0, [tuple(b0_loc)])
            # print(neighbors)
        else:
            all_data = False

        if subsample:
            for n in range(len(neighbors)):
                neighbors[n] = greedy_set_finder(neighbors[n])

        indexes = [x for shell in neighbors for x in shell]
    else:
        if angular_size >= len(bvals):
            indexes = [np.arange(len(bvals))]
            all_data = True
        else:
            all_data = False
            local_size = min(angular_size, len(dwis)) - len(b0_loc)
            neighbors = angular_neighbors(sym_bvecs, local_size) % data.shape[-1]
            neighbors = neighbors[:data.shape[-1]]  # everything was doubled for symmetry

            full_indexes = [(dwi,) + tuple(neighbors[dwi]) for dwi in dwis]
            # for dwi in dwis:
            #     print((dwi,) + tuple(neighbors[dwi]))
            #     print(np.min(neighbors[dwi]), np.max(neighbors[dwi]))
            # 1/0
            if subsample:
                indexes = greedy_set_finder(full_indexes)
            else:
                indexes = full_indexes
    print('dwis is ', dwis, bvecs.shape, data.shape, sym_bvecs.shape)
    # print('full_indexes is ', full_indexes)
    print('b0_loc is ', b0_loc)
    print('indexes is ', len(indexes))
    for ll in indexes:
        print(len(ll), ll)
    # print('neighbors is ', np.min(neighbors), np.max(neighbors), )
    #     for ff in full_indexes:
            # print(ff, len(ff), ff[:30][-1], local_size, data.shape, bvecs.shape)
    # 1/0
    # If we have more b0s than indexes, then we have to add a few more blocks since
    # we won't do a full cycle. If we have more b0s than indexes after that, then it breaks.
    if split_shell or all_data:
        pass
    else:
        if num_b0s > len(indexes):
            the_rest = [rest for rest in full_indexes if rest not in indexes]
            indexes += the_rest[:(num_b0s - len(indexes))]

        if num_b0s > len(indexes):
            error = (f'Seems like you still have more b0s {num_b0s} than available blocks {len(indexes)},'
                    ' either average them or deactivate subsampling.')
            raise ValueError(error)

    data_denoised = np.zeros(data.shape, np.float32)
    divider = np.zeros(data.shape[-1])

    # Put all idx + b0 in this array in each iteration

    for i, idx in enumerate(indexes, start=1):
        # print(dwis)
        # print(idx)
        # idx = dwis[idx]
        if all_data:
            to_denoise = np.empty(data.shape[:-1] + (len(idx),), dtype=dtype)
            b0_block_size = tuple(block_size[:-1]) + ((len(idx),))
            # Full overlap for dictionary learning

            # current_b0 = 0
            volumes = list(idx)

            # to_denoise[..., 0] = data[..., current_b0].squeeze()
            to_denoise[:] = data[..., idx]
        else:
            to_denoise = np.empty(data.shape[:-1] + (len(idx) + 1,), dtype=dtype)
            b0_block_size = tuple(block_size[:-1]) + ((len(idx) + 1,))
            # Full overlap for dictionary learning

            current_b0 = tuple((next(split_b0s_idx),))
            volumes = list(current_b0 + idx)

            to_denoise[..., 0] = data[..., current_b0].squeeze()
            to_denoise[..., 1:] = data[..., idx]

        overlap = np.array(b0_block_size, dtype=np.int16) - 1
        print('overlap 1', overlap)
        # overlap = 0,0,0, overlap[-1]
        divider[volumes] += 1

        logger.info(f'Now denoising volumes {volumes} / block {i} out of {len(indexes)}.')

        data_denoised[..., volumes] += local_denoise(to_denoise,
                                                    b0_block_size,
                                                    overlap,
                                                    variance,
                                                    bvals[volumes],
                                                    bvecs[volumes],
                                                    n_iter=n_iter,
                                                    mask=mask,
                                                    dtype=dtype,
                                                    n_cores=n_cores,
                                                    verbose=verbose)
    data_denoised /= divider

    # If we averaged b0s but didn't go through them because we did not have enough blocks,
    # we just put back the value to prevent empty volumes
    filled_vols = divider > 0

    if average_b0s and (np.sum(filled_vols) < data_denoised.shape[-1]):
        filled_b0s = []
        empty_b0s = []

        for b0s in b0_loc:
            if filled_vols[b0s]:
                filled_b0s += [b0s]
            else:
                empty_b0s += [b0s]

        if len(filled_b0s) > 1:
            b0s = np.mean(data_denoised[..., filled_b0s], axis=-1, keepdims=True)
        else:
            b0s = data_denoised[..., filled_b0s]

        logger.info(f'Filling in b0s volumes {empty_b0s} from the average of b0s volumes {filled_b0s}.')

        data_denoised[..., empty_b0s] = b0s

    return data_denoised

def local_denoise(data, block_size, overlap, variance, bvals, bvecs, n_iter=10, mask=None,
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
    param_D['modeD'] = 1
    param_D['lambda1'] = 1.2 / np.sqrt(np.prod(block_size))
    param_D['lambda2'] = 1e-6
    param_D['gamma1'] = 0.05
    param_D['gamma2'] = 0.05
    param_D['K'] = int(np.prod(block_size) * 1.5)
    param_D['iter'] = 2
    param_D['regul'] = 'graph-ridge' #'fused-lasso'
    param_D['batchsize'] = 150
    param_D['numThreads'] = n_cores
    param_D['verbose'] = True

    if 'D' in param_alpha:
        param_D['D'] = param_alpha['D']
        ratio = param_alpha['D'].shape[1] / param_alpha['D'].shape[0]
    else:
        ratio = param_D['K'] / np.prod(block_size)

    graph = make_groups(bvals, ratio)
    # print(param_D['K'], graph['groups_var'].shape)
    param_D['graph'] = graph
    param_D['tree'] = None
    param_D['K'] = graph['groups_var'].shape[0] # to account for the variability from np.ceil

    mask_col = extract_patches(mask, block_size[:-1], (1, 1, 1), flatten=False)
    axis = tuple(range(mask_col.ndim//2, mask_col.ndim))
    train_idx = np.sum(mask_col, axis=axis).ravel() > (np.prod(block_size[:-1]) / 2)

    print('in small slicer', data.shape, mask.shape, X.shape, train_idx.shape, block_size)
    train_data = np.asfortranarray(X[:, train_idx])
    train_data /= np.sqrt(np.sum(train_data**2, axis=0, keepdims=True), dtype=dtype)
    print('weights', graph['eta_g'], graph['groups_var'].shape)

    tt = time()
    param_alpha['D'] = spams.structTrainDL(train_data, **param_D)
    param_alpha['D'][np.isnan(param_alpha['D'])] = 0
    print('time train', time() - tt)
    print(param_alpha['D'])
    del train_idx, train_data, X, mask_col
    # param_alpha['D'] /= np.sqrt(np.sum(param_alpha['D']**2, axis=0, keepdims=True, dtype=dtype))
    # prune zeros
    bad = np.abs(param_alpha['D']).sum(axis=0) == 0
    param_alpha['D'] = param_alpha['D'][:, ~bad]
    param_D['D'] = param_alpha['D']

    print('D stuff', param_alpha['D'].shape, np.sum(param_alpha['D'] != 0),  np.sum(param_alpha['D'] == 0), np.sum(np.isnan(param_alpha['D'])), param_alpha['D'].size, np.nanmin(param_alpha['D']), np.nanmax(param_alpha['D']), np.nanmean(param_alpha['D']))
    # print('X stuff', X.shape, np.nanmin(X), np.nanmax(X), np.nanmean(X))
    # print('train_data stuff', np.nanmin(train_data), np.nanmax(train_data), np.nanmean(train_data))
    # print(param_D['D'].shape)
    # print(param_D['D'].sum(axis=1))
    # print(param_D['D'].sum(axis=0))
    # print(bad)
    # print('D stuff', param_alpha['D'].shape, np.sum(param_alpha['D'] != 0),  np.sum(param_alpha['D'] == 0), np.sum(np.isnan(param_alpha['D'])), param_alpha['D'].size, np.nanmin(param_alpha['D']), np.nanmax(param_alpha['D']), np.nanmean(param_alpha['D']))
    # 1/0
    param_alpha['numThreads'] = n_cores
    param_D['numThreads'] = n_cores
    # print(param_D['D'])
    # print(np.linalg.norm(param_D['D'], axis=0))
    # 1/0
    # slicer = [np.index_exp[:, :, k:k + block_size[2]] for k in range((data.shape[2] - block_size[2] + 1))]
    slicer = [np.index_exp[:, :, k:k + block_size[2]] for k in range(30,40)]

    if verbose:
        progress_slicer = tqdm(slicer) # This is because tqdm consumes the (generator) slicer, but we also need it later :/
    else:
        progress_slicer = slicer

    time_multi = time()
    data_denoised = Parallel(n_jobs=-1)(delayed(processer)(data,
                                                                mask,
                                                                variance,
                                                                block_size,
                                                                overlap,
                                                                param_alpha,
                                                                param_D,
                                                                current_slice,
                                                                bvals,
                                                                bvecs,
                                                                graph,
                                                                dtype,
                                                                n_iter)
                                                                for current_slice in progress_slicer)

    logger.info(f'Multiprocessing done in {(time() - time_multi) / 60:.2f} mins.')

    # Put together the multiprocessed results
    data_subset = np.zeros_like(data, dtype=np.float32)
    divider = np.zeros_like(data, dtype=np.int16)

    for current_slice, content in zip(slicer, data_denoised):
        data_subset[current_slice] += content
        divider[current_slice] += 1

    data_subset /= divider
    return data_subset


def processer(data, mask, variance, block_size, overlap, param_alpha, param_D, current_slice, bvals, bvecs, graph,
              dtype=np.float64, n_iter=10, gamma=3, tau=1, tolerance=1e-5):

    # Fetch the current slice for parallel processing since now the arrays are dumped and read from disk
    # instead of passed around as smaller slices by the function to 'increase performance'

    # print(f'current slice is {current_slice}')
    data = data[current_slice]
    mask = mask[current_slice]
    # variance = variance[current_slice]

    orig_shape = data.shape
    mask_array = im2col_nd(mask, block_size[:-1], overlap[:-1])
    train_idx = np.sum(mask_array, axis=0) > (mask_array.shape[0] / 2)
    # print('after blocking')
    # If mask is empty, return a bunch of zeros as blocks
    if not np.any(train_idx):
        return np.zeros_like(data)

    Y = im2col_nd(data, block_size, overlap)
    Y_full_shape = Y.shape
    print('Yfull', Y.mean(), Y.shape, mask_array.shape)
    Y = Y[:, train_idx].astype(dtype)
    print('after blocking 2', Y.shape,Y_full_shape,data.shape, param_alpha['D'].shape, block_size, overlap)
    print('Ymask', Y.mean(), Y.shape, mask_array.shape)
    1/0

    X = param_alpha['D']

    # sorter = frobenius_sort(bvals, bvecs)
    # unsorter = sorter.argsort()
    # sorter = np.arange(len(bvals))
    # unsorter = np.arange(len(bvals))

    # print('before sort',X.shape, Y.shape)
    # X = X[sorter]
    # Y = Y[sorter]

    # best_sols, best_recon, best_lambdas, best_freedom = path_stuff(X, Y)
    # best_recon = best_recon[:, unsorter]
    best_sols = np.linalg.lstsq(X, Y, rcond=-1)[0].T

    X = np.asfortranarray(X)
    Y = np.asfortranarray(Y)
    best_sols = np.asfortranarray(best_sols.T)
    # best_lambdas = np.ones(Y.shape[1]) * 0.05

    # print(X.shape, Y.shape, best_sols.shape, np.median(best_lambdas), np.mean(best_lambdas), best_lambdas.min(), best_lambdas.max())

    # lbda = np.mean(best_lambdas)
    param_alpha = {}
    param_alpha['regul'] = 'sparse-group-lasso-l2' #'multi-task-graph'# 'fused-lasso' 'l1l2' #
    param_alpha['loss'] = 'square'
    param_alpha['pos'] = True
    param_alpha['intercept'] = True
    param_alpha['L0'] = 0.1
    param_alpha['lambda1'] = 0.05
    param_alpha['lambda2'] = 0.05
    # param_alpha['lambda3'] = 1e-6

    alpha = np.zeros_like(best_sols)
    # best_sols[:] = 0
    # best_lambdas[best_lambdas == 0] = 1e-6
    if param_alpha['pos']:
        best_sols.clip(min=0, out=best_sols)

    # graph = make_groups(bvals, X.shape[1] / X.shape[0])
    print(graph['eta_g'].shape)
    print(graph['groups'].shape)
    print(graph['groups_var'].shape)
    print(Y.shape, X.shape, best_sols.shape, X.shape[1] / X.shape[0])
    print('Ymean', Y.mean())
    print('Xmean', X.mean())
    # alpha = spams.fistaFlat(Y, X, best_sols, **param_alpha)
    alpha = spams.fistaGraph(Y, X, best_sols, graph, **param_alpha)
    print('nonzero stuff', alpha.shape, alpha.size, alpha.min(), alpha.max(), np.sum(alpha==0), np.sum(alpha!=0), 'ratio nz', np.sum(alpha!=0)/alpha.size)
    # for nn in tqdm(range(alpha.shape[1])):
    #     param_alpha['lambda1'] = best_lambdas[nn]
    #     alpha[:, nn:nn+1] = spams.fistaFlat(Y[:, nn:nn+1], X, best_sols[:, nn:nn+1], **param_alpha)
    #     # print(nn, np.sum(alpha != 0), alpha.size, param_alpha['lambda2'], Y[:, nn:nn+1].mean(), best_lambdas[nn], X.shape, alpha.shape)
    best_recon = (X @ alpha).T
    # best_recon = (X @ alpha)[unsorter].T
    del X, alpha, Y, best_sols

    # best_recon = np.linalg.lstsq(X, Y, rcond=-1)[0].T @ X.T
    # print(X.shape, Y.shape, best_recon.shape, np.linalg.lstsq(X, Y, rcond=-1)[0].shape)
    # best_recon = Y.copy().T
    # print('sort stuff')
    # print(best_recon.shape, X.shape, Y.shape)
    # print(best_recon.shape, X.shape, Y.shape)
    # print(sorter)
    # print(unsorter)
    # print(best_recon.shape)

    # # param_alpha['L'] = int(0.5 * Y.shape[0])
    # param_alpha['pos'] = True
    # param_alpha['regul'] = 'fused-lasso'
    # # param_alpha['admm'] = True
    # param_alpha['loss'] = 'square'
    # # param_alpha['verbose'] = True
    # param_alpha['admm'] = False

    # # lambda_max = 1

    # # def soft_thresh(x, gamma):
    # #     return np.sign(x) * np.max(0, np.abs(x) - gamma)

    # # maxlambda = np.linalg.norm(soft_thresh(X.T @ Y / Y.shape[1]))

    # # lambda_min = lambda_max / 100
    # # nlambas = 1
    # # lambda_path = np.logspace(lambda_max, lambda_min, nlambas)
    # # alphaw = 0.95

    # X = param_alpha['D']
    # W = np.zeros((X.shape[1], Y.shape[1]), dtype=dtype, order='F')
    # W0 = np.zeros((X.shape[1], Y.shape[1]), dtype=dtype, order='F')

    # # from scipy.optimize import lsq_linear
    # from time import time
    # tt = time()
    # # W0 = lsq_linear(X, Y, bounds=(0, np.inf))['x']
    # # print(f'time nnls {time() - tt}')
    # W0 = np.linalg.lstsq(X, Y, rcond=None)[0]

    # if param_alpha['pos']:
    #     W0.clip(min=0)

    # del param_alpha['mode']
    # del param_alpha['D']
    # from time import time

    # for ii, lbda in enumerate(lambda_path):
    #     tt = time()
    #     print(ii, f'current lambda {lbda}')
    #     param_alpha['lambda1'] = alphaw * lbda
    #     param_alpha['lambda2'] = (1 - alphaw) * lbda
    #     W0 = np.linalg.lstsq(X, Y, rcond=None)[0]
    #     W0.clip(min=0)
    #     W0 = np.asfortranarray(W0)
    #     W[:] = spams.fistaFlat(Y, X, W0, **param_alpha)
    #     W0[:] = W
    #     print(f'abs sum sol {np.abs(W0).sum()}, min max {W0.min(), W0.max()}, abs min max {np.abs(W0).min(), np.abs(W0).max()}, nonzero {np.sum(W0 !=0) / W0.size}')
    #     print(f'time {time() - tt}')

    Y = np.zeros(Y_full_shape, dtype=dtype, order='F')
    Y[:, train_idx] = best_recon.T
    # Y[:, train_idx] = best_freedom.T
    # best_sols, best_recon, best_lambdas, best_freedom
    out = col2im_nd(Y, block_size, orig_shape, overlap)

    # param_alpha['mode'] = 1
    # param_alpha['D'] = X

    return out

def frobenius_sort(bval, bvec, bdelta=None, base=None):

    def Cb(bval1, bval2, l=1):
        logbval1 = np.log(bval1, out=np.zeros_like(bval1), where=bval1>1e-5)
        logbval2 = np.log(bval2, out=np.zeros_like(bval2), where=bval2>1e-5)

        return np.exp(-(logbval1 - logbval2)**2 / (2 * l**2))

    def Ctheta(bvec1, bvec2):
        # print(np.inner(bvec1, bvec2))
        return np.arccos(np.inner(bvec1, bvec2)).squeeze()

    def distance(bval1, bvec1, bval2, bvec2):
        # print(bval1, bvec1, bval2, bvec2)
        # print(Ctheta(bvec1, bvec2))
        # print(Cb(bval1, bval2))
        return Ctheta(bvec1, bvec2) * Cb(bval1, bval2)

    if bdelta is None:
        bdelta = np.ones_like(bval)

    if base is None:
        # base_bvec = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        base_bvec = np.array([1, 0, 0])
        base_bval = np.mean(bval)
    else:
        base_bval, base_bvec = base

    bvec = bvec / np.linalg.norm(bvec, axis=0, keepdims=True)
    base_bvec = base_bvec / np.linalg.norm(base_bvec)
    base_bvec = base_bvec[None, :]

    assert base_bvec.ndim == 2
    assert len(bval) == bvec.shape[0]
    assert len(bval) == len(bdelta)

    # bmatrix = bval/3 * (np.eye(3) + bdelta * np.diag([-1, -1, 2]))

    # bxx = bval / 3 * (1 - bdelta)
    # byy = bval / 3 * (1 - bdelta)
    # bzz = bval / 3 * (1  + 2*bdelta)

    # bmatrix = np.diag([bxx, byy, bzz])
    # bmatrix = bval / 3 * (np.eye(3) + bdelta * np.diag([-1, -1, 2]))

    # N = len(bval)
    # bmatrix = np.zeros((N, 3, 3))
    # for i in range(N):
    #     bmatrix[i] = bval[i]**2 * np.outer(bvec[i], bvec[i])

    # bmatrix_vec = bval * np.array([bvec[:,0]**2, 2*bvec[:,0] * bvec[:,1], 2*bvec[:,0] * bvec[:,2],
    #                                bvec[:,1]**2, 2*bvec[:,1] * bvec[:,2], bvec[:,2]**2]).T

    # bmatrix = np.zeros([N, 3, 3])
    # for i in range(N):
    #     bmatrix[i, 0] = bmatrix_vec[i, :3]
    #     bmatrix[i, 1, 1:] = bmatrix_vec[i, 3:5]
    #     bmatrix[i, 2, 2] = bmatrix_vec[i, 5]
    #     bmatrix[i] = bmatrix[i] + bmatrix[i].T - np.eye(3) * np.diag(bmatrix[i])


    # print(np.abs(np.trace(bmatrix, axis1=-2, axis2=-1) - bval).max())
    # print(bval)
    # assert np.allclose(np.trace(bmatrix, axis1=-2, axis2=-1), bval)

    distance_all = distance(base_bval, base_bvec, bval, bvec)
    # print(distance_all)
    # print(distance_all.shape)
    # norm = np.linalg.norm(base - bmatrix, ord='fro', axis=0)
    sorter = np.argsort(distance_all)
    return sorter


def make_groups(bvals, ratio, b0_threshold=10, l=1):
    bvals = np.copy(bvals)
    bvals[bvals < b0_threshold] = 1e-15
    unique_bvals = np.unique(bvals)
    mean_bval = np.mean(bvals)
    ngroups = len(unique_bvals)

    weigths = np.exp(-(np.log(mean_bval) - np.log(unique_bvals))**2 / (2 * l**2))
    groups = np.zeros([ngroups, ngroups], dtype=bool)
    groups_var = [bvalue == unique_bvals for bvalue in bvals]
    # groups_var = np.array(groups_var, dtype=bool)
    each = np.ceil(np.sum(groups_var, axis=0) * ratio)
    groups_var = np.zeros([int(each.sum()), ngroups], dtype=bool)

    indexes = [0] + list(np.int16(each.cumsum()))
    for i, (gv1, gv2) in enumerate(pairwise(indexes)):
        # print(gv1, gv2, i)
        idx = np.index_exp[gv1:gv2, i]
        groups_var[idx] = 1

    groups = ssp.csc_array(groups)
    groups_var = ssp.csc_array(groups_var)

    graph = {'eta_g': weigths,
             'groups' : groups, # Are groups a subgroup of another group?
             'groups_var' : groups_var}  # Are groups sharing elements?
    return graph
