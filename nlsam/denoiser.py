import numpy as np
import logging
import scipy.linalg

from time import time
from itertools import cycle

from nlsam.utils import im2col_nd, col2im_nd
from nlsam.angular_tools import angular_neighbors, split_per_shell, greedy_set_finder
from autodmri.blocks import extract_patches

from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm
from scipy.linalg import solve_triangular, null_space, qr_delete, qr_insert, solve_banded, cholesky_banded, cho_solve_banded
import spams

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

        if subsample:
            for n in range(len(neighbors)):
                neighbors[n] = greedy_set_finder(neighbors[n])

        indexes = [x for shell in neighbors for x in shell]
    else:
        neighbors = angular_neighbors(sym_bvecs, angular_size - 1) % data.shape[-1]
        neighbors = neighbors[:data.shape[-1]]  # everything was doubled for symmetry

        full_indexes = [(dwi,) + tuple(neighbors[dwi]) for dwi in range(data.shape[-1]) if dwi in dwis]

        if subsample:
            indexes = greedy_set_finder(full_indexes)
        else:
            indexes = full_indexes

    # Full overlap for dictionary learning
    overlap = np.array(block_size, dtype=np.int16) - 1

    # If we have more b0s than indexes, then we have to add a few more blocks since
    # we won't do a full cycle. If we have more b0s than indexes after that, then it breaks.
    if num_b0s > len(indexes):
        the_rest = [rest for rest in full_indexes if rest not in indexes]
        indexes += the_rest[:(num_b0s - len(indexes))]

    if num_b0s > len(indexes):
        error = (f'Seems like you still have more b0s {num_b0s} than available blocks {len(indexes)},'
                 ' either average them or deactivate subsampling.')
        raise ValueError(error)

    b0_block_size = tuple(block_size[:-1]) + ((angular_size + 1,))
    data_denoised = np.zeros(data.shape, np.float32)
    divider = np.zeros(data.shape[-1])

    # Put all idx + b0 in this array in each iteration
    to_denoise = np.empty(data.shape[:-1] + (angular_size + 1,), dtype=dtype)

    for i, idx in enumerate(indexes, start=1):
        current_b0 = tuple((next(split_b0s_idx),))
        to_denoise[..., 0] = data[..., current_b0].squeeze()
        to_denoise[..., 1:] = data[..., idx]
        divider[list(current_b0 + idx)] += 1

        logger.info(f'Now denoising volumes {current_b0 + idx} / block {i} out of {len(indexes)}.')

        data_denoised[..., current_b0 + idx] += local_denoise(to_denoise,
                                                              b0_block_size,
                                                              overlap,
                                                              variance,
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
            b0s = np.mean(data_denoised[..., filled_b0s], axis=-1)
        else:
            b0s = data_denoised[..., filled_b0s]

        logger.info(f'Filling in b0s volumes {empty_b0s} from the average of b0s volumes {filled_b0s}.')

        data_denoised[..., empty_b0s] = b0s

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
    param_D['modeD'] = 0
    param_D['lambda1'] = 1.2 / np.sqrt(np.prod(block_size))
    param_D['K'] = int(2 * np.prod(block_size))
    param_D['iter'] = 150
    param_D['regul'] = 'fused-lasso'
    # param_D['batchsize'] = 500
    param_D['numThreads'] = n_cores

    if 'D' in param_alpha:
        param_D['D'] = param_alpha['D']

    mask_col = extract_patches(mask, block_size[:-1], (1, 1, 1), flatten=False)
    axis = tuple(range(mask_col.ndim//2, mask_col.ndim))
    train_idx = np.sum(mask_col, axis=axis).ravel() > (np.prod(block_size[:-1]) / 2)

    train_data = np.asfortranarray(X[:, train_idx])
    train_data /= np.sqrt(np.sum(train_data**2, axis=0, keepdims=True), dtype=dtype)

    param_alpha['D'] = spams.structTrainDL(train_data, **param_D)
    param_alpha['D'] /= np.sqrt(np.sum(param_alpha['D']**2, axis=0, keepdims=True, dtype=dtype))
    param_D['D'] = param_alpha['D']

    del train_idx, train_data, X, mask_col

    param_alpha['numThreads'] = n_cores
    param_D['numThreads'] = n_cores

    # slicer = [np.index_exp[:, :, k:k + block_size[2]] for k in range((data.shape[2] - block_size[2] + 1))]
    slicer = [np.index_exp[:, :, k:k + block_size[2]] for k in range(40,50)]

    if verbose:
        progress_slicer = tqdm(slicer) # This is because tqdm consumes the (generator) slicer, but we also need it later :/
    else:
        progress_slicer = slicer

    time_multi = time()

    data_denoised = Parallel(n_jobs=1)(delayed(processer)(data,
                                                                mask,
                                                                variance,
                                                                block_size,
                                                                overlap,
                                                                param_alpha,
                                                                param_D,
                                                                current_slice,
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


def processer(data, mask, variance, block_size, overlap, param_alpha, param_D, current_slice,
              dtype=np.float64, n_iter=10, gamma=3, tau=1, tolerance=1e-5):

    # Fetch the current slice for parallel processing since now the arrays are dumped and read from disk
    # instead of passed around as smaller slices by the function to 'increase performance'

    data = data[current_slice]
    mask = mask[current_slice]
    variance = variance[current_slice]

    orig_shape = data.shape
    mask_array = im2col_nd(mask, block_size[:-1], overlap[:-1])
    train_idx = np.sum(mask_array, axis=0) > (mask_array.shape[0] / 2)

    # If mask is empty, return a bunch of zeros as blocks
    if not np.any(train_idx):
        return np.zeros_like(data)

    Y = im2col_nd(data, block_size, overlap)
    Y_full_shape = Y.shape
    Y = Y[:, train_idx].astype(dtype)

    # param_alpha['L'] = int(0.5 * Y.shape[0])
    param_alpha['pos'] = True
    param_alpha['regul'] = 'fused-lasso'
    # param_alpha['admm'] = True
    param_alpha['loss'] = 'square'
    # param_alpha['verbose'] = True
    param_alpha['admm'] = False

    lambda_max = 1

    def soft_thresh(x, gamma):
        return np.sign(x) * np.max(0, np.abs(x) - gamma)

    maxlambda = np.linalg.norm(soft_thresh(X.T @ Y / Y.shape[1]))

    lambda_min = lambda_max / 100
    nlambas = 1
    lambda_path = np.logspace(lambda_max, lambda_min, nlambas)
    alphaw = 0.95

    X = param_alpha['D']
    W = np.zeros((X.shape[1], Y.shape[1]), dtype=dtype, order='F')
    W0 = np.zeros((X.shape[1], Y.shape[1]), dtype=dtype, order='F')

    # from scipy.optimize import lsq_linear
    from time import time
    tt = time()
    # W0 = lsq_linear(X, Y, bounds=(0, np.inf))['x']
    # print(f'time nnls {time() - tt}')
    W0 = np.linalg.lstsq(X, Y, rcond=None)[0]

    if param_alpha['pos']:
        W0.clip(min=0)

    del param_alpha['mode']
    del param_alpha['D']
    from time import time

    for ii, lbda in enumerate(lambda_path):
        tt = time()
        print(ii, f'current lambda {lbda}')
        param_alpha['lambda1'] = alphaw * lbda
        param_alpha['lambda2'] = (1 - alphaw) * lbda
        W0 = np.linalg.lstsq(X, Y, rcond=None)[0]
        W0.clip(min=0)
        W0 = np.asfortranarray(W0)
        W[:] = spams.fistaFlat(Y, X, W0, **param_alpha)
        W0[:] = W
        print(f'abs sum sol {np.abs(W0).sum()}, min max {W0.min(), W0.max()}, abs min max {np.abs(W0).min(), np.abs(W0).max()}, nonzero {np.sum(W0 !=0) / W0.size}')
        print(f'time {time() - tt}')

    Y = np.zeros(Y_full_shape, dtype=dtype, order='F')
    Y[:, train_idx] = X @ W
    out = col2im_nd(Y, block_size, orig_shape, overlap)

    param_alpha['mode'] = 1
    param_alpha['D'] = X

    return out

def frobenius_sort(bval, bvec, bdelta=None, base=(1,0,0)):
    if bdelta is None:
        bdelta = 1

    bmatrix = bval/3 * (np.eye(3) + bdelta * np.diag([-1, -1, 2]))


    assert np.trace(np.allclose(bmatrix, bval))

    norm = np.linalg.norm(base, bmatrix, ord='fro', axis=-1)
    sorted = np.argsort(norm)
    return sorted

# def diagonal_form(a, upper=1, lower=1):
#     """
#     a is a numpy square matrix
#     this function converts a square matrix to diagonal ordered form
#     returned matrix in ab shape which can be used directly for solve_banded
#     """
#     n = a.shape[1]
#     assert(np.all(a.shape ==(n,n)))

#     ab = np.zeros((2*n-1, n))

#     for i in range(n):
#         ab[i,(n-1)-i:] = np.diagonal(a,(n-1)-i)

#     for i in range(n-1):
#         ab[(2*n-2)-i,:i+1] = np.diagonal(a,i-(n-1))

#     mid_row_inx = int(ab.shape[0]/2)
#     upper_rows = [mid_row_inx - i for i in range(1, upper+1)]
#     upper_rows.reverse()
#     upper_rows.append(mid_row_inx)
#     lower_rows = [mid_row_inx + i for i in range(1, lower+1)]
#     keep_rows = upper_rows+lower_rows
#     ab = ab[keep_rows,:]

#     return ab

@njit
def _diagonal_banded(l_and_u, a):
    n = a.shape[1]
    (nlower, nupper) = l_and_u

    diagonal_ordered = np.empty((nlower + nupper + 1, n), dtype=a.dtype)
    for i in range(1, nupper + 1):
        for j in range(n - i):
            diagonal_ordered[nupper - i, i + j] = a[j, i + j]

    for i in range(n):
        diagonal_ordered[nupper, i] = a[i, i]

    for i in range(nlower):
        for j in range(n - i - 1):
            diagonal_ordered[nupper + 1 + i, j] = a[i + j + 1, j]

    return diagonal_ordered

def diagonal_form(a):
    n = a.shape[1]
    ab = np.zeros((3, n))

    diagu = np.diag(a, k=1)
    diag = np.diag(a)
    diagl = np.diag(a, k=-1)

    ab[0, 1:] = diagu
    ab[1, :] = diag
    ab[2, :-1] = diagl

    return ab

def diagonal_choform(a):
    n = a.shape[1]
    ab = np.zeros((2, n))

    diagu = np.diag(a, k=1)
    diag = np.diag(a)

    ab[0, 1:] = diagu
    ab[1, :] = diag

    return ab

from numba import njit
# from qpsolvers import solve_ls

# @njit(cache=True, parallel=True)
def path_stuff(X, y, penalty='fused', nlambdas=500, eps=1e-8, lmin=1e-5, l2=1e-6, l1=0, dtype=np.float32):
    if nlambdas <= 0 or eps <= 0 or lmin < 0 or l2 < 0 or l1 < 0:
        error = 'Parameter error'
        raise ValueError(error)

    nold = X.shape[0] # This is the original shape, which is modified later on if l2 > 0
    # N = y.shape[1]
    p = X.shape[1]

    mus = [None] * nlambdas
    lambdas = [None] * nlambdas
    betas = [None] * nlambdas
    dfs = [None] * nlambdas

    if penalty == 'fused':
        D = np.eye(p-1, p, k=1) - np.eye(p-1, p)
        C = D.T @ D - 2*np.eye(p) != 0
        # G = np.eye(p) + np.eye(p, k=1)
    elif penalty == 'sparse':
        if l1 <= 0:
            error = f'Penalty sparse requires l1 > 0, but it is set to l1 = {l1}'
            raise ValueError(error)

        D = np.eye(p-1, p, k=1) - np.eye(p-1, p)
        D = np.vstack((D, l1 * np.eye(p)))
    else:
        error = f'Penalty {penalty} is not implemented'
        raise ValueError(error)

    rank = np.linalg.matrix_rank(X)
    if l2 == 0 and rank < nold:
        error = f'System {X.shape} is undefined with rank = {rank} < n = {nold}, set l2 = {l2} larger than 0'
        raise ValueError(error)

    if l2 > 0:
        y = np.hstack((y, np.zeros(p)))
        X = np.vstack((X, np.sqrt(l2) * np.eye(p)))
        # D = np.vstack((D, np.zeros([p, 1])))

    n = X.shape[0]
    m = D.shape[0]
    B = np.zeros(m, dtype=bool)
    S = np.zeros(m, dtype=np.int8)
    muk = np.zeros(m)

    Xpinv = np.linalg.pinv(X)
    # L = np.linalg.cholesky(D @ D.T)
    # Special form if we have the fused penalty
    DDt_diag_banded = np.vstack([np.full(m, -1), np.full(m, 2)])
    DDt_diag_banded[0, 0] = 0
    L_banded = cholesky_banded(DDt_diag_banded, lower=False, check_finite=False)
    # DDt_banded = diagonal_form(D @ D.T)

    # PcolX = X @ Xpinv
    # PcolX = np.eye(n)
    # XtXpinv = np.linalg.pinv(X.T @ X)
    # P = X @ Xpinv
    # yold = y
    yproj = X @ Xpinv @ y
    Dproj = D @ Xpinv
    # L = D.T @ D
    # Q, R = np.linalg.qr(D)

    # step 1
    # DinvD = np.linalg.pinv(Dm @ Dm.T) @ Dm
    # Dts = D.T @ s
    # uk = DinvD @ (y - lambdak * Dts)

    # step 1
    # Q, R = np.linalg.qr(D.T, mode='reduced')

    # H = null_space(D)
    # H = np.mean(D != 0, axis=0, keepdims=True).T
    # H = np.ones((p, 1))
    # PXy = np.mean(X, axis=1, keepdims=True) @ y
    # XH = X.sum(axis=1, keepdims=True)
    # XHpinv = np.linalg.pinv(XH)
    # XH = X @ H

    # XH = np.zeros((n, 1))
    # for i in range(n):
    #     idx = np.nonzero(D[:, i])
    #     XH[i] = X[idx].mean()

    XH = X.sum(axis=1, keepdims=True)
    Xty = X.T @ y
    # v = Xty - X.T @ XH @ np.linalg.pinv(XH.T @ XH) @ XH.T @ y
    v = Xty - X.T @ XH @ np.linalg.pinv(XH) @ y
    # v = Xty @ (np.eye(n) - PXy)
    # v = Xty - X.T @ XH @ y
    # u0 = np.linalg.lstsq(D @ D.T, D @ v, rcond=None)[0]
    # u0 = solve_banded((1, 1), DDt_banded, D @ v, check_finite=False)
    u0 = cho_solve_banded((L_banded, False), D @ v, check_finite=False)
    # u0 = np.linalg.lstsq(R.T @ R, D @ v, rcond=None)[0]
    # u0 = D @ np.linalg.lstsq(L, v, rcond=None)[0]
    # u0 = D @ np.linalg.lstsq(R.T @ R, v, rcond=None)[0]
    # u0 = D @ np.linalg.lstsq(D.T @ D, v, rcond=None)[0]
    # u0 = D @ v
    # u0 = D @ solve_ls(R.T @ R, v, solver='cvxopt')

    # step 2
    DinvD = np.linalg.pinv(Dproj @ Dproj.T) @ Dproj
    Dts = np.zeros(Dproj.shape[1]) # first step B is empty
    a = DinvD @ yproj
    b = DinvD @ Dts
    # t = a / (b + np.sign(a))
    tp = a / (b + 1)
    tm = a / (b - 1)
    t = np.maximum(tp, tm)
    i0 = np.argmax(t)
    h0 = t[i0]

    # mat = D @ (np.eye(*Dm.shape) - Dm.T @ DinvD)
    # c = s @ (mat @ y)
    # d = s @ (mat @ Dts)
    # tl = c / d
    # tl[np.logical_and(c < 0, d < 0)] = 0
    # lk = np.max(tl)

    # lambdak = lbda
    B[i0] = True
    S[i0] = np.sign(u0[i0])
    lambdas[0] = h0
    mus[0] = u0
    dfs[0] = 0
    # first sol is usually empty?
    # betas[0] = a - h0 * b
    betas[0] = Xpinv @ (yproj - Dproj.T @ u0) # eq. 36

    k = 1
    updates = B[i0], S[i0]
    coord = i0

    # this is to compute a bunch of matrix transpose and pinv,
    # but because H is a colums of 1s, it has a very simple expression we just duplicate in columns later on
    XtXH = X.T @ XH
    XHtXH = XH.T @ XH
    ratio = XtXH / XHtXH

    while np.abs(lambdas[k-1]) > lmin and k < nlambdas:
        print(k, lambdas[k-1])

        # if updates[0]:
        #     # add line to QR
        #     Q, R = qr_insert(Q, R, D[coord], coord)
        #     pass
        # else:
        #     # remove line from QR
        #     Q, R = qr_delete(Q, R, coord)
        #     pass

        Db = D[B]
        Dm = D[~B]
        # Dpb = Dproj[B]
        # Dpm = Dproj[~B]
        s = S[B]
        Dts = Db.T @ s
        # DDt = Dm @ Dm.T
        # DDt_banded = diagonal_choform(DDt)
        DDt_banded = DDt_diag_banded[:, ~B]
        L_banded = cholesky_banded(DDt_banded, lower=False, check_finite=False)
        # H = np.ones((p, Dm.shape[1] - Dm.shape[0]))

        # DDt = Dm @ Dm.T
        # DtD = Dm.T @ Dm
        # DtD = R.T @ R

        # step 3a
        # H = null_space(Dm)
        # XHm = X@H
        # H = np.mean(Dm != 0, axis=0, keepdims=True)
        # H = np.ones(Dm.shape[0]) / (Dm.shape[1] - Dm.shape[0])
        # H = np.ones((Dm.shape[1], Dm.shape[1] - Dm.shape[0]))
        # H[np.nonzero(B)] = 1

        # XH = X @ H
        # Allocating is contiguous so this is faster than a view in the end
        null = Dm.shape[1] - Dm.shape[0]
        XHm = np.full((n, null), XH)
        # XtXHm = np.full((p, null), XtXH)
        HtDts = np.full(null, np.sum(Dts))
        # XHmpinv = np.full((null, null), 1 / XHtXH / null**2)
        # mid = X.T @ XHm @ np.linalg.pinv(XHm.T @ XHm)
        # mid = XtXHm @ XHmpinv
        mid = np.full((p, null), ratio / null)
        v = Xty - mid @ XHm.T @ y
        w = Dts - mid @ HtDts

        # a = Dm @ np.linalg.lstsq(DtD, v, rcond=None)[0]
        # b = Dm @ np.linalg.lstsq(DtD, w, rcond=None)[0]

        # a = solve_banded((1,1), DDt_banded, Dm @ v, check_finite=False)
        # b = solve_banded((1,1), DDt_banded, Dm @ w, check_finite=False)
        a = cho_solve_banded((L_banded, False), Dm @ v, check_finite=False)
        b = cho_solve_banded((L_banded, False), Dm @ w, check_finite=False)

        # tf.linalg.tridiagonal_solve(tf.constant(DDt),tf.constant(rhs),diagonals_format='matrix')
        # a = np.linalg.lstsq(DDt, Dm @ v, rcond=None)[0]
        # b = np.linalg.lstsq(DDt, Dm @ w, rcond=None)[0]

        # step 3b
        # hitting time
        tp = a / (b + 1)
        tm = a / (b - 1)
        t = np.maximum(tp, tm)
        # numerical issues fix
        t[t > lambdas[k-1].squeeze() + eps] = 0
        t[t > lambdas[k-1].squeeze()] = lambdas[k-1].squeeze()
        # t.clip(max=lambdas[k-1], out=t)

        ik = np.argmax(t)
        hk = t[ik]
        # print(f'{t}')

        if np.any(t > lambdas[k-1]):
            t[t > lambdas[k-1] + 1e-7] = 0
            t[t > lambdas[k-1]] = lambdas[k-1]

        # leaving time
        # mat = mid
        # mat = Dpb @ (In - Dpm.T @ np.linalg.pinv(Dpm @ Dpm.T) @ Dpm)
        # mat = Dpb @ (In - Dpm.T @ np.linalg.pinv(Dpm))
        # mat = Dpb @ (In - Dpm.T @ np.linalg.pinv(Dpm))
        # mat = Dpb @ (In - Dpm.T @ mid @ XH.T) @ H.T
        # c = s * (mat @ y)
        # d = s * (mat @ Dts)
        c = s * (Db @ (Xty - Dm.T @ a))
        d = s * (Db @ (Xty - Dm.T @ b))

        # tl = c / d
        # tl[~np.logical_and(c < 0, d < 0)] = 0
        tl = np.where((c < 0) & (d < 0), c/d, 0)
        # numerical issues fix
        tl[tl > lambdas[k-1].squeeze() + eps] = 0
        tl[tl > lambdas[k-1].squeeze()] = lambdas[k-1].squeeze()
        # tl.clip(max=lambdas[k-1], out=tl)

        ilk = np.argmax(tl)
        lk = tl[ilk]

        print(f'hk={hk}, lk={lk}')
        # update matrices and stuff for next step

        if hk > lk:
            coord = np.nonzero(~B)[0][ik]
            updates = True, np.sign(hk)
            lambdak = hk
            print(f'add {ik, coord}, B={B.sum()}')
        else:
            coord = np.nonzero(B)[0][ilk]
            updates = False, 0
            lambdak = lk
            print(f'remove {ilk, coord}, B={B.sum()}')

        muk[~B] = a - lambdak * b
        muk[B] = lambdak * s
        mus[k] = muk.copy()

        # update boundaries
        B[coord], S[coord] = updates
        # mus[k] = a - lambdak * b
        # betas[k] = XtXpinv @ (X.T @ yproj - Dpm.T @ mus[k])
        # betas[k] = Xpinv @ (yproj - Dpm.T @ mus[k])
        # betas[k] = Xpinv @ yproj - Dproj.T @ mus[k]
        # betas[k] = y - lambdak * Dpb.T @ s
        # betas[k] = XH @ (y - lambdak * Dpb.T @ s) # eq. 33
        betas[k] = Xpinv @ (yproj - Dproj.T @ muk) # eq. 36
        # betas[k] = Xpinv @ H @ (yproj - lambdak * Dpm.T @ s) # eq. 38
        lambdas[k] = lambdak
        dfs[k] = XH.shape[1]
        # Cp[k] = np.sum((y - X@betas[k])**2) - n * var + 2*var * dfs[k]
        k += 1

    # strip beta from the l2 penalty part if needed
    dtype = np.float64
    out = np.zeros([k, p], dtype=dtype)
    for i in range(k):
        out[i] = betas[i]

    return_lambdas = True
    if return_lambdas:
        return out, lambdas[:k]
    return out
