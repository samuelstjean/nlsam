from __future__ import division, print_function

import numpy as np
try:
    from nlsam.spams import spams
except ImportError:
    try:
        from spams import spams
    except ImportError:
        try:
            import spams
        except ImportError:
            try:
                from spams_python import spams
            except ImportError:
                raise ValueError("Couldn't find spams library")

from time import time
from itertools import repeat
from multiprocessing import Pool, cpu_count

from nlsam.utils import sparse_dot, im2col_nd, col2im_nd, padding

import scipy.sparse as ssp


def universal_worker(input_pair):
    """http://stackoverflow.com/a/24446525"""
    function, args = input_pair
    return function(*args)


def pool_args(function, *args):
    return zip(repeat(function), zip(*args))


def greedy_set_finder(sets):
    """Returns a list of subsets that spans the input sets with a greedy algorithm
    http://en.wikipedia.org/wiki/Set_cover_problem#Greedy_algorithm"""

    sets = [set(s) for s in sets]
    universe = set()

    for s in sets:
        universe = universe.union(s)
        sets

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


# def processer(data, mask, variance, block_size, overlap, param_alpha, param_D, downscaling, beta=0.5, gamma=1., dtype=np.float64, eps=1e-10, n_iter=10):
def processer(arglist):

    data, mask, variance, block_size, overlap, param_alpha, param_D, downscaling, beta, gamma, dtype, eps, n_iter = arglist

    orig_shape = data.shape
    var_array = variance[mask]
    mask = np.repeat(mask[..., None], orig_shape[-1], axis=-1)
    mask_array = im2col_nd(mask, block_size, overlap).T
    train_idx = np.sum(mask_array, axis=0) > mask_array.shape[0]/2

    X = im2col_nd(data, block_size, overlap).T

    # If mask is empty, return a bunch of zeros as blocks
    if not np.any(train_idx):
        return np.zeros_like(data)

    X_full_shape = X.shape
    X = np.asfortranarray(X[:, train_idx], dtype=dtype)
    W = np.ones((param_alpha['D'].shape[1], X.shape[1]), dtype=dtype, order='F')
    eps = np.finfo(dtype).eps
    param_alpha['mode'] = 1
    param_alpha['lambda1'] = np.asscalar(np.percentile(var_array, 95))
    param_alpha['L'] = int(0.5 * X.shape[0])

    X_old = np.ones_like(X)
    not_converged = np.ones(X.shape[1], dtype=np.bool)
    alpha_converged = np.zeros((param_alpha['D'].shape[1], X.shape[1]), dtype=dtype)

    for i in range(n_iter):

        if np.all(not_converged == 0):
            print("broke the loop", i, np.abs(X - X_old).max(), np.abs(X_old - X).min(), np.max(X), np.min(X), np.max(X) * 0.02)
            break

        param_alpha['W'] = W
        alpha = spams.lassoWeighted(np.asfortranarray(X[:, not_converged]), **param_alpha)
        X[:, not_converged] = sparse_dot(param_alpha['D'], alpha)

        # Convergence if max(X-X_old) < 2 % max(X)
        X_conv = np.max(np.abs(X[:, not_converged] - X_old[:, not_converged]), axis=0) > (np.max(X[:, not_converged], axis=0) * 0.02)

        x, y, idx = ssp.find(alpha)
        alpha_converged[:, not_converged][x, y] = idx
        not_converged[not_converged] = np.logical_or(X_conv, np.any(X[:, not_converged] < 0, axis=0))

        W = np.array(np.maximum(np.repeat(np.sum(np.abs(X[:, not_converged] - X_old[:, not_converged]), axis=0, keepdims=True), param_alpha['D'].shape[1], axis=0), eps), dtype=dtype, order='F', copy=False)
        X_old[:, not_converged] = X[:, not_converged]

    alpha = alpha_converged

    weigths = np.ones(X_full_shape[1], dtype=dtype, order='F')
    weigths[train_idx] = 1. / np.array((alpha != 0).sum(axis=0) + 1., dtype=dtype).squeeze()
    X2 = np.zeros(X_full_shape, dtype=dtype, order='F')
    X2[:, train_idx] = X

    return col2im_nd(X2, block_size, orig_shape, overlap, weigths)


def denoise(data, block_size, overlap, param_alpha, param_D, variance, n_iter=10, noise_std=None,
            batchsize=512, mask_data=None, mask_train=None, mask_noise=None,
            whitening=True, savename=None, dtype=np.float64, debug=False):

    # no overlapping blocks for training
    no_over = (0, 0, 0, 0)
    X = im2col_nd(data, block_size, no_over).T

    # Solving for D
    param_D['verbose'] = False
    param_D['posAlpha'] = True
    param_D['posD'] = True
    param_alpha['pos'] = True
    param_D['mode'] = 2
    param_D['lambda1'] = 1.2 / np.sqrt(np.prod(block_size))
    param_D['K'] = int(2*np.prod(block_size))

    if 'D' in param_alpha:
        print ("D is already supplied, \
               \nhot-starting dictionnary learning from D")
        param_D['D'] = param_alpha['D']
        param_D['iter'] = 150

        start = time()
        step = block_size[0]

        mask = mask_data
        mask_data = im2col_nd(np.repeat(mask_data[..., None], data.shape[-1], axis=-1), block_size, no_over).T
        train_idx = np.sum(mask_data, axis=0) > mask_data.shape[0]/2
        train_data = X[:, train_idx]
        train_data = np.asfortranarray(train_data[:, np.any(train_data != 0, axis=0)], dtype=dtype)
        train_data /= np.sqrt(np.sum(train_data**2, axis=0, keepdims=True))
        param_alpha['D'] = spams.trainDL(train_data, **param_D)
        param_alpha['D'] /= np.sqrt(np.sum(param_alpha['D']**2, axis=0, keepdims=True, dtype=dtype))
        param_D['D'] = param_alpha['D']

        print ("Training done : total time = ",
               time()-start, np.min(param_alpha['D']), np.max(param_alpha['D']))

    else:
        if 'K' not in param_D:
            print ('No dictionnary size specified. 256 atoms will be chosen.')
            param_D['K'] = 256

        param_D['iter'] = 150

        start = time()
        step = block_size[0]
        param_D['batchsize'] = 500 #train_data.shape[1]//10
        mask = mask_data
        mask_data = im2col_nd(np.repeat(mask_data[..., None], data.shape[-1], axis=-1), block_size, no_over).T
        train_idx = np.sum(mask_data, axis=0) > mask_data.shape[0]/2
        train_data = X[:, train_idx]
        train_data = np.asfortranarray(train_data[:, np.any(train_data != 0, axis=0)], dtype=dtype)
        train_data /= np.sqrt(np.sum(train_data**2, axis=0, keepdims=True), dtype=dtype)
        param_alpha['D'] = spams.trainDL(train_data, **param_D)
        param_alpha['D'] /= np.sqrt(np.sum(param_alpha['D']**2, axis=0, keepdims=True, dtype=dtype))

        print ("Training done : total time = ",
               time()-start, np.min(param_alpha['D']), np.max(param_alpha['D']))

    del train_data
    start = time()
    param_alpha['mode'] = 2
    param_alpha['lambda1'] = 1.2 / np.sqrt(np.prod(block_size))
    data = padding(data, block_size, overlap)

    n_cores = param_alpha['numThreads']
    param_alpha['numThreads'] = 1
    param_D['numThreads'] = 1

    print('Multiprocessing Stuff')
    time_multi = time()
    pool = Pool(processes=n_cores)
    print("cores", n_cores)
    downscaling = 1.
    beta = 0.5
    gamma = 1
    eps = 1e-10

    arglist = [(data[:, k:k+block_size[1], ...], mask[:, k:k+block_size[1]], variance[:, k:k+block_size[1]], block_size_subset, overlap_subset, param_alpha_subset, param_D_subset, downscaling_subset, beta_subset, gamma_subset, dtype_subset, eps_subset, n_iter_subset)
               for k, block_size_subset, overlap_subset, param_alpha_subset, param_D_subset, downscaling_subset, beta_subset, gamma_subset, dtype_subset, eps_subset, n_iter_subset
               in zip(range(data.shape[1] - block_size[1] + 1),
                      repeat(block_size),
                      repeat(overlap),
                      repeat(param_alpha),
                      repeat(param_D),
                      repeat(downscaling),
                      repeat(beta),
                      repeat(gamma),
                      repeat(dtype),
                      repeat(eps),
                      repeat(n_iter))]

    data_denoised = pool.map(processer, arglist)
    pool.close()
    pool.join()

    param_alpha['numThreads'] = n_cores
    param_D['numThreads'] = n_cores

    print('Multiprocessing done', time()-time_multi)

    # Put together the multiprocessed results
    data_subset = np.zeros_like(data)
    divider = np.zeros_like(data, dtype=np.int16)
    ones = np.ones_like(data_denoised[0], dtype=np.int16)

    for k in range(len(data_denoised)):
        data_subset[:, k:k+block_size[1], ...] += data_denoised[k]
        divider[:, k:k+block_size[1], ...] += ones

    data_subset /= divider
    return data_subset
