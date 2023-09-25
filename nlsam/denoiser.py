import numpy as np
import logging
# import scipy.linalg

from time import time
from itertools import cycle

from nlsam.utils import im2col_nd, col2im_nd
from nlsam.angular_tools import angular_neighbors, split_per_shell, greedy_set_finder
from autodmri.blocks import extract_patches

from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm
# from scipy.linalg import  null_space, cholesky_banded, cho_solve_banded
import scipy.linalg as la
import scipy.sparse as ssp
import spams
import qpsolvers
import numba

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

# @njit
# def _diagonal_banded(l_and_u, a):
#     n = a.shape[1]
#     (nlower, nupper) = l_and_u

#     diagonal_ordered = np.empty((nlower + nupper + 1, n), dtype=a.dtype)
#     for i in range(1, nupper + 1):
#         for j in range(n - i):
#             diagonal_ordered[nupper - i, i + j] = a[j, i + j]

#     for i in range(n):
#         diagonal_ordered[nupper, i] = a[i, i]

#     for i in range(nlower):
#         for j in range(n - i - 1):
#             diagonal_ordered[nupper + 1 + i, j] = a[i + j + 1, j]

#     return diagonal_ordered

# def diagonal_form(a):
#     n = a.shape[1]
#     ab = np.zeros((3, n))

#     diagu = np.diag(a, k=1)
#     diag = np.diag(a)
#     diagl = np.diag(a, k=-1)

#     ab[0, 1:] = diagu
#     ab[1, :] = diag
#     ab[2, :-1] = diagl

#     return ab


from numba.extending import get_cython_function_address
from numba import njit, types
import ctypes

_PTR = ctypes.POINTER

_dbl = ctypes.c_double
_int = ctypes.c_int

_ptr_dbl = _PTR(_dbl)
_ptr_int = _PTR(_int)

addr = get_cython_function_address('scipy.linalg.cython_lapack', 'dpbtrs')
functype = ctypes.CFUNCTYPE(None,
                            _ptr_int,  # UPLO
                            _ptr_int,  # N
                            _ptr_int,  # KD
                            _ptr_int,  # NRHS
                            _ptr_dbl,  # AB
                            _ptr_int,  # LDAB
                            _ptr_dbl,  # B
                            _ptr_int,  # LDB
                            _ptr_int)  # INFO
numba_dpbtrs = functype(addr)

addr = get_cython_function_address('scipy.linalg.cython_lapack', 'dpbtrf')
functype = ctypes.CFUNCTYPE(None,
                            _ptr_int,  # UPLO
                            _ptr_int,  # N
                            _ptr_int,  # KD
                            _ptr_dbl,  # AB
                            _ptr_int,  # LDAB
                            _ptr_int)  # INFO
numba_dpbtrf = functype(addr)
# fnty = types.FunctionType(types.int64(types.float64[:,:], types.float64[:,:], types.boolean, types.boolean))
# @njit(types.int64(fnty, types.UniTuple(types.int64, 2)), cache=True)


@njit(cache=True, parallel=True, fastmath=True, nogil=True)
def dpbtrs(cb, b, lower=False, overwrite_b=False):
    if lower:
        UPLO = 'L'
    else:
        UPLO = 'U'

    # if b.ndim == 1:
    #     b = np.atleast_2d(b).T

    if b.flags.f_contiguous:
        if not overwrite_b:
            b = np.copy(b)
    else:
        b = np.asfortranarray(b)

    cb = np.asfortranarray(cb)

    UPLO = np.array([ord(UPLO)], dtype=np.int32)
    NRHS = np.array(b.shape[1], dtype=np.int32)
    N = np.array(cb.shape[1], dtype=np.int32)
    KD = np.array(cb.shape[0] - 1, dtype=np.int32)
    AB = cb
    LDAB = np.array(cb.shape[0], dtype=np.int32)
    B = b
    LDB = np.array(b.shape[0], dtype=np.int32)
    INFO = np.array(0, dtype=np.int32)

    numba_dpbtrs( UPLO.ctypes,
                N.ctypes,
                KD.ctypes,
                NRHS.ctypes,
                AB.ctypes,
                LDAB.ctypes,
                B.ctypes,
                LDB.ctypes,
                INFO.ctypes)
    return B, INFO


@njit(cache=True, parallel=True, fastmath=True, nogil=True)
def dpbtrf(ab, lower=False, overwrite_ab=False):
    if lower:
        UPLO = 'L'
    else:
        UPLO = 'U'

    # if b.ndim == 1:
    #     b = np.atleast_2d(b).T

    if ab.flags.f_contiguous:
        if not overwrite_ab:
            ab = np.copy(ab)
    else:
        ab = np.asfortranarray(ab)

    UPLO = np.array([ord(UPLO)], dtype=np.int32)
    N = np.array(ab.shape[1], dtype=np.int32)
    KD = np.array(ab.shape[0] - 1, dtype=np.int32)
    AB = ab
    LDAB = np.array(ab.shape[0], dtype=np.int32)
    INFO = np.array(0, dtype=np.int32)

    numba_dpbtrf(UPLO.ctypes,
                N.ctypes,
                KD.ctypes,
                AB.ctypes,
                LDAB.ctypes,
                INFO.ctypes)
    return AB, INFO

@njit(cache=True, parallel=True, fastmath=True, nogil=True)
def backsolve(A, b, lower=False, transpose=0):
    alpha = 1.0
    out = la.blas.dtrsm(alpha, A, b, lower=lower, trans_a=transpose)
    # out = mydtrsm(alpha, A, b, lower=lower, trans_a=transpose)
    return out

@njit(cache=True, parallel=True, fastmath=True, nogil=True)
def forwardsolve(A, b, lower=True, transpose=0):
    return backsolve(A, b, lower=lower, transpose=transpose)

@njit(cache=True, parallel=True, fastmath=True, nogil=True)
def diagonal_choform(a):
    n = a.shape[1]
    ab = np.zeros((2, n))

    diagu = np.diag(a, k=1)
    diag = np.diag(a)

    ab[0, 1:] = diagu
    ab[1, :] = diag

    return ab

@njit(cache=True, parallel=True, fastmath=True, nogil=True)
def cholesky_banded(ab, lower=False, overwrite_ab=False):
    c, info = dpbtrf(ab, lower=lower, overwrite_ab=overwrite_ab)

    if info > 0:
        raise ValueError("%d-th leading minor not positive definite" % info)
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal pbtrf' % -info)
    return c

@njit(cache=True, parallel=True, fastmath=True, nogil=True)
def cho_solve_banded(cb_and_lower, b, overwrite_b=False):
    (cb, lower) = cb_and_lower

    x, info = dpbtrs(cb, b, lower=lower, overwrite_b=overwrite_b)

    if info > 0:
        raise ValueError("%dth leading minor not positive definite" % info)
    if info < 0:
        raise ValueError('illegal value in %dth argument of internal pbtrs' % -info)
    return x


# from numba import njit
# from qpsolvers import solve_ls

def path_stuff(X, y, penalty='fused', nsteps=500, eps=1e-8, lmin=0, l2=1e-6, l1=0, dtype=np.float32, return_lambdas=True):
    if nsteps <= 0 or eps < 0 or lmin < 0 or l2 < 0 or l1 < 0:
        error = 'Parameter error'
        raise ValueError(error)

    if y.ndim == 1:
        y = y[:, None]

    nold = X.shape[0] # This is the original shape, which is modified later on if l2 > 0
    N = y.shape[1]
    p = X.shape[1]

    if penalty == 'fused':
        D = np.eye(p-1, p, k=1) - np.eye(p-1, p)
    elif penalty == 'sparse':
        if l1 <= 0:
            error = f'Penalty sparse requires l1 > 0, but it is set to l1 = {l1}'
            raise ValueError(error)

        D = np.eye(p-1, p, k=1) - np.eye(p-1, p)
        D = np.vstack((D, l1 * np.eye(p)))
    else:
        error = f'Penalty {penalty} is not implemented'
        raise ValueError(error)

    # rank = np.linalg.matrix_rank(X)
    # if l2 == 0 and rank < nold:
    #     error = f'System {X.shape} is undefined with rank = {rank} < n = {nold}, set l2 = {l2} larger than 0'
    #     raise ValueError(error)

    if p >= nold and l2 == 0:
        l2 = 1e-6
        print(f'Adding a small ridge penalty l2={l2} as needed since X has more columns = {p} than rows = {nold}')

    if l2 > 0:
        y = np.vstack((y, np.zeros((p, N))))
        X = np.vstack((X, np.sqrt(l2) * np.eye(p)))

    n = X.shape[0]
    m = D.shape[0]
    B = np.zeros(m, dtype=bool)
    S = np.zeros(m, dtype=np.int8)
    muk = np.zeros(m)
    var = np.var(y, axis=0)

    all_mus = [None] * N
    all_lambdas = [None] * N
    all_betas = [None] * N
    all_dfs = [None] * N

    for i in range(N):
        all_mus[i] = [None] * nsteps
        all_lambdas[i] = [None] * nsteps
        all_betas[i] = [None] * nsteps
        all_dfs[i] = [None] * nsteps

    # Xpinv = np.linalg.pinv(X)
    # Special form if we have the fused penalty
    DDt_diag_banded = np.vstack([np.full(m, -1), np.full(m, 2)])
    DDt_diag_banded[0, 0] = 0
    L_banded = la.cholesky_banded(DDt_diag_banded, lower=False, check_finite=False)

    # yproj = X @ Xpinv @ y
    # Dproj = D @ Xpinv

    # step 1
    H = np.ones([p, 1])
    XH = X@H
    # Q, R = np.linalg.qr(XH)
    Xty = X.T @ y
    # b = scipy.linalg.solve_triangular(R.T, H.T @ Xty, lower=True, check_finite=False)
    # mid = scipy.linalg.solve_triangular(R, b, check_finite=False)

    rhs = np.linalg.lstsq(XH.T @ XH, H.T @ Xty, rcond=None)[0]
    v = Xty - X.T @ XH @ rhs
    u0 = la.cho_solve_banded((L_banded, False), D @ v, check_finite=False)

    # step 2
    t = np.abs(u0)
    i0 = np.argmax(t, axis=0)
    h0 = t[i0, range(N)]

    for i in range(N):
        all_betas[i][0] = H @ rhs[:, i]
        all_lambdas[i][0] = h0[i]
        all_mus[i][0] = u0[:, i]
        all_dfs[i][0] = 1

    # Q_all = Q.copy()
    # R_all = R.copy()
    Xty_all = Xty.copy()
    XtX = X.T @ X
    Xty = np.zeros(p)
    K = np.zeros(N, dtype=np.int16)

    for i in range(N):
        k = 0
        B[:] = 0
        S[:] = 0
        ilocal = i0[i]

        H = np.ones([p, 1])
        newH = np.ones((p, 1))
        newH[:ilocal + 1] = 0
        # XnewH = X @ newH
        # Q = Q_all.copy()
        # R = R_all.copy()
        Xty[:] = Xty_all[:, i]

        B[ilocal] = True
        S[ilocal] = np.sign(u0[ilocal, i])

        # Q, R = la.qr_insert(Q, R, XnewH, 1, which='col', check_finite=False)
        H = np.hstack((H, newH))
        XH = X @ H
        XtXH = XtX @ H

        mus = all_mus[i]
        lambdas = all_lambdas[i]
        betas = all_betas[i]
        dfs = all_dfs[i]
        # initvals = np.zeros((2, 2))

        while np.abs(lambdas[k]) > lmin and k < (nsteps - 1):
            # print(k, lambdas[k])

            Db = D[B]
            Dm = D[~B]
            s = S[B]
            Dts = Db.T @ s
            DDt = Dm @ Dm.T
            DDt_banded = diagonal_choform(DDt)
            L_banded = cholesky_banded(DDt_banded, check_finite=False)

            # Reset the QR to prevent accumulating errors during long runs
            # if k % 1 == 0:
            R = np.linalg.qr(XH, mode='r')

            rhs = np.vstack((H.T @ Xty, H.T @ Dts)).T
            # print(Q.shape, R.shape, rhs.shape)
            # b = la.solve_triangular(R.T, rhs, lower=True, check_finite=False)
            # A = la.solve_triangular(R, b, check_finite=False)
            # A = np.linalg.lstsq(R, np.linalg.lstsq(R.T, rhs, rcond=None)[0], rcond=None)[0]
            # A = np.linalg.lstsq(R.T @ R, rhs, rcond=None)[0]
            A = backsolve(R, forwardsolve(R, rhs, lower=False, transpose=True))
            # A1 = qpsolvers.solve_ls(XH.T @ XH, rhs[:, 0], solver='quadprog')
            # A2 = qpsolvers.solve_ls(XH.T @ XH, rhs[:, 1], solver='quadprog')

            # AA = XH.T @ XH
            # lhs = AA.T @ AA
            # A1 = qpsolvers.solve_qp(lhs, H.T @ Xty, solver='quadprog')
            # A2 = qpsolvers.solve_qp(lhs, H.T @ Dts, solver='quadprog')
            # A = np.linalg.solve(R, np.linalg.solve(R.T, rhs, rcond=None)[0], rcond=None)[0]

            # print(A1)
            # print(A2)
            A1 = A[:, 0]
            A2 = A[:, 1]

            # step 3a
            if Dm.shape[0] == 0: # Interior is empty
                hk = 0
                a = np.zeros(0)
                b = np.zeros(0)
            else:
                # XtXH = X.T @ Q @ R
                # XtXH = XtX @ H

                v = Xty - XtXH @ A1
                w = Dts - XtXH @ A2

                rhs = Dm @ np.vstack((v, w)).T
                A = cho_solve_banded((L_banded, False), rhs, check_finite=False)
                a = A[:, 0]
                b = A[:, 1]
                # a = la.cho_solve_banded((L_banded, False), Dm @ v, check_finite=False)
                # b = la.cho_solve_banded((L_banded, False), Dm @ w, check_finite=False)

                # step 3b
                # hitting time
                t = a / (b + np.sign(a) + eps) # prevent divide by 0
                # numerical issues fix
                t[t > lambdas[k].squeeze() + eps] = 0

                ik = np.argmax(t)
                hk = t[ik]

            # leaving time
            HA1 = H @ A1
            HA2 = H @ A2

            if Db.shape[0] == 0 or H.shape[0] == H.shape[1]: # Boundary is empty
                lk = 0
            else:
                c =  s * (Db @ HA1)
                d =  s * (Db @ HA2)

                tl = np.where((c < 0) & (d < 0), c/(d + eps), 0)
                # numerical issues fix
                tl[tl > lambdas[k].squeeze() + eps] = 0

                ilk = np.argmax(tl)
                lk = tl[ilk]

            # update matrices and stuff for next step
            if hk > lk: # variable enters the boundary
                coord = np.nonzero(~B)[0][ik]
                updates = True, np.sign(a[ik])
                lambdak = hk
                newH = np.ones(p)
                newH[:coord + 1] = 0
                newpos = np.searchsorted(np.nonzero(B)[0], coord)
                # Q, R = la.qr_insert(Q, R, X @ newH, newpos + 1, which='col', overwrite_qru=True, check_finite=False)
                H = np.insert(H, newpos + 1, newH, axis=1)
                XH = np.insert(XH, newpos + 1, X @ newH, axis=1)
                XH = X @ H
                XtXH = np.insert(XtXH, newpos + 1, XtX @ newH, axis=1)
                XtXH = XtX @ H
                XtXH = X.T @ XH
                # print(H.shape, R.shape, newpos, newH.shape, XH.shape)
                # print(f'add {ik, coord}, B={B.sum()}')
            elif lk > hk: # variable leaves the boundary
                coord = np.nonzero(B)[0][ilk]
                updates = False, 0
                lambdak = lk
                # Q, R = la.qr_delete(Q, R, ilk + 1, which='col', overwrite_qr=True, check_finite=False)
                H = np.delete(H, ilk + 1, axis=1)
                XH = np.delete(XH, ilk + 1, axis=1)
                XtXH = np.delete(XtXH, ilk + 1, axis=1)
                # print(f'remove {ilk, coord}, B={B.sum()}')
            elif (lk == 0) and (hk == 0): # end of the path, so everything stays as is
                lambdak = 0
                # print('end of the path reached')

            k += 1
            muk[~B] = a - lambdak * b
            muk[B] = lambdak * s
            mus[k] = muk.copy()

            # update boundaries
            B[coord], S[coord] = updates
            betas[k] = HA1 - lambdak * HA2 # eq. 38
            lambdas[k] = lambdak
            dfs[k] = H.shape[1] - 1

        K[i] = k

    all_l2_error = [None] * N
    all_Cps = [None] * N
    all_best_idx = np.zeros(N, dtype=np.int16)

    for i in range(N):
        k = K[i]
        print(i, K[i])
        lambdas = np.array(all_lambdas[i][:k+1])
        betas = np.array(all_betas[i][:k+1])
        dfs = np.array(all_dfs[i][:k+1])
        l2_error = np.sum((y[:, i] - betas @ X.T)**2, axis=1)
        Cps = l2_error - n * var[i] + 2 * var[i] * dfs
        best_idx = np.argmin(Cps)
        print(best_idx, Cps.shape)

        all_lambdas[i] = lambdas
        all_betas[i] = betas
        all_dfs[i] = dfs
        all_l2_error[i] = l2_error
        all_Cps[i] = Cps
        all_best_idx[i] = best_idx

    if return_lambdas:
        return all_betas, all_lambdas, all_Cps, all_best_idx
    return all_betas

def inner_path(X, y, Xty):