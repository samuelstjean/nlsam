from __future__ import division, print_function

import numpy as np
import warnings
from time import time

from itertools import repeat, izip
from multiprocessing import Pool, cpu_count

from nlsam.utils import sparse_dot, im2col_nd, col2im_nd, padding
from scipy.sparse import lil_matrix

warnings.simplefilter("ignore", category=FutureWarning)

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


def apply_weights(alpha, W):
    cx = alpha.tocoo()
    for i, j in izip(cx.row, cx.col):
        cx[i, j] /= W[i, j]

    return cx


def compute_weights(alpha, alpha_old, W, tau, eps):
    cx = alpha.tocoo()
    cy = alpha_old.tocoo()

    # Reset W values to eps
    idx = cy.nonzero()
    W[idx] = 1./eps
    # for i, j in izip(cy.row, cy.col):
    #     W[i, j] = 1./eps

    # Assign new weights
    idx = cx.nonzero()
    W[idx] = 1. / ((cx.data**tau) + eps)
    # for i, j, v in izip(cx.row, cx.col):
    #     W[i, j] =

    return W


def check_conv(alpha, alpha_old, eps=1e-5):
    x = alpha.tocoo()
    y = alpha_old.tocoo()
    # has_converged = np.zeros(alpha.shape[1], dtype=np.bool)

    return np.abs(x - y).max(axis=0) < eps

    # diff = (x - y).max(axis=0)
    # diff.data = np.abs(diff.data)
    # return diff < eps
    # for i, d in izip(diff.row, diff.data):
    #     has_converged[i] = d < eps

    # return has_converged

# def processer(data, mask, variance, block_size, overlap, param_alpha, param_D, dtype=np.float64, n_iter=10):
def processer(arglist):

    data, mask, variance, block_size, overlap, param_alpha, param_D, dtype, n_iter = arglist

    orig_shape = data.shape
    # mask = np.repeat(mask[..., None], orig_shape[-1], axis=-1)
    mask_array = im2col_nd(np.broadcast_to(mask[..., None], orig_shape), block_size, overlap)
    train_idx = np.sum(mask_array, axis=0) > mask_array.shape[0]/2

    # If mask is empty, return a bunch of zeros as blocks
    if not np.any(train_idx):
        return np.zeros_like(data)

    X = im2col_nd(data, block_size, overlap)
    var_mat = np.median(im2col_nd(variance[..., 0:orig_shape[-1]], block_size, overlap)[:, train_idx], axis=0).astype(dtype)

    X_full_shape = X.shape
    X = X[:, train_idx]

    # param_alpha['mode'] = 1
    param_alpha['L'] = int(0.5 * X.shape[0])

    D = param_alpha['D']

    alpha = lil_matrix((D.shape[1], X.shape[1]))
    W = np.ones(alpha.shape, dtype=dtype, order='F')

    DtD = np.dot(D.T, D)
    DtX = np.dot(D.T, X)
    DtXW = np.empty_like(DtX, order='F')

    alpha_old = np.ones(alpha.shape, dtype=dtype)
    has_converged = np.zeros(alpha.shape[1], dtype=np.bool)

    xi = np.random.randn(X.shape[0], X.shape[1]) * var_mat
    eps = np.max(np.abs(np.dot(D.T, xi)), axis=0)
    param_alpha['mode'] = 1
    param_alpha['pos'] = True
    gamma = 3
    tau = 1

    for _ in range(n_iter):
        not_converged = np.equal(has_converged, False)
        DtXW[:, not_converged] = DtX[:, not_converged] / W[:, not_converged]

        for i in range(alpha.shape[1]):
            if not has_converged[i]:

                param_alpha['lambda1'] = var_mat[i] * (X.shape[0] + gamma * np.sqrt(2 * X.shape[0]))
                DtDW = (1 / W[..., None, i]) * DtD * (1 / W[:, i])
                alpha[:, i:i+1] = spams.lasso(X[:, i:i+1], Q=np.asfortranarray(DtDW), q=DtXW[:, i:i+1], **param_alpha)

        arr = alpha.toarray()
        nonzero_ind = arr != 0
        arr[nonzero_ind] /= W[nonzero_ind]
        has_converged = np.max(np.abs(alpha_old - arr), axis=0) < 1e-5

        if np.all(has_converged):
            print(_, "break")
            break

        alpha_old = arr

        W[:] = 1. / (np.abs(alpha_old**tau) + eps)

    alpha = arr
    X[:] = sparse_dot(D, alpha)

    weigths = np.ones(X_full_shape[1], dtype=dtype, order='F')
    weigths[train_idx] = 1. / (np.sum(alpha != 0, axis=0) + 1.)

    X2 = np.zeros(X_full_shape, dtype=dtype, order='F')
    X2[:, train_idx] = X

    return col2im_nd(X2, block_size, orig_shape, overlap, weigths)


def denoise(data, block_size, overlap, param_alpha, param_D, variance, n_iter=10,
            mask=None, savename=None, dtype=np.float64, debug=False):

    # no overlapping blocks for training
    no_over = (0, 0, 0, 0)
    X = im2col_nd(data, block_size, no_over)

    # Solving for D
    param_alpha['pos'] = True
    param_alpha['mode'] = 2
    param_alpha['lambda1'] = 1.2 / np.sqrt(np.prod(block_size))

    param_D['verbose'] = False
    param_D['posAlpha'] = True
    param_D['posD'] = True
    param_D['mode'] = 2
    param_D['lambda1'] = 1.2 / np.sqrt(np.prod(block_size))
    param_D['K'] = int(2*np.prod(block_size))
    param_D['iter'] = 150
    param_D['batchsize'] = 500

    if 'D' in param_alpha:
        param_D['D'] = param_alpha['D']

    # mask_col = im2col_nd(np.repeat(mask[..., None], data.shape[-1], axis=-1), block_size, no_over)
    mask_col = im2col_nd(np.broadcast_to(mask[..., None], data.shape), block_size, no_over)
    train_idx = np.sum(mask_col, axis=0) > mask_col.shape[0]/2
    train_data = X[:, train_idx]
    train_data = np.asfortranarray(train_data[:, np.any(train_data != 0, axis=0)], dtype=dtype)
    train_data /= np.sqrt(np.sum(train_data**2, axis=0, keepdims=True), dtype=dtype)
    param_alpha['D'] = spams.trainDL(train_data, **param_D)
    param_alpha['D'] /= np.sqrt(np.sum(param_alpha['D']**2, axis=0, keepdims=True, dtype=dtype))
    param_D['D'] = param_alpha['D']

    del train_data

    # data = padding(data, block_size, overlap)

    n_cores = param_alpha['numThreads']
    param_alpha['numThreads'] = 1
    param_D['numThreads'] = 1

    # print('Multiprocessing Stuff')
    time_multi = time()
    pool = Pool(processes=n_cores)
    # print("cores", n_cores)

    arglist = [(data[:, :, k:k+block_size[2]], mask[:, :, k:k+block_size[2]], variance[:, :, k:k+block_size[2]], block_size_subset, overlap_subset, param_alpha_subset, param_D_subset, dtype_subset, n_iter_subset)
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

    param_alpha['numThreads'] = n_cores
    param_D['numThreads'] = n_cores

    print('Multiprocessing done in {0:.2f} mins.'.format((time() - time_multi) / 60.))

    # Put together the multiprocessed results
    data_subset = np.zeros_like(data)
    divider = np.zeros_like(data, dtype=np.int16)
    ones = np.ones_like(data_denoised[0], dtype=np.int16)

    for k in range(len(data_denoised)):
        data_subset[:, :, k:k+block_size[2]] += data_denoised[k]
        divider[:, :, k:k+block_size[2]] += ones

    data_subset /= divider
    return data_subset
