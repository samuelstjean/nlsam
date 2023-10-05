import numpy as np

from tqdm.autonotebook import tqdm
# from scipy.linalg import  null_space, cholesky_banded, cho_solve_banded
import scipy.linalg as la
import scipy.sparse as ssp
import qpsolvers
import numba

from numba.extending import get_cython_function_address
from numba import njit, types
from numba.core.extending import overload
import ctypes

_PTR = ctypes.POINTER

_dbl = ctypes.c_double
_int = ctypes.c_int

_ptr_dbl = _PTR(_dbl)
_ptr_int = _PTR(_int)

addr_dpbtrs = get_cython_function_address('scipy.linalg.cython_lapack', 'dpbtrs')
functype_dpbtrs = ctypes.CFUNCTYPE(None,
                            _ptr_int,  # UPLO
                            _ptr_int,  # N
                            _ptr_int,  # KD
                            _ptr_int,  # NRHS
                            _ptr_dbl,  # AB
                            _ptr_int,  # LDAB
                            _ptr_dbl,  # B
                            _ptr_int,  # LDB
                            _ptr_int)  # INFO
numba_dpbtrs = functype_dpbtrs(addr_dpbtrs)

addr_dtrsm = get_cython_function_address('scipy.linalg.cython_blas', 'dtrsm')
functype_dtrsm = ctypes.CFUNCTYPE(None,
                            _ptr_int,  # SIDE
                            _ptr_int,  # UPLO
                            _ptr_int,  # TRANSA
                            _ptr_int,  # DIAG
                            _ptr_int,  # M
                            _ptr_int,  # N
                            _ptr_dbl,  # ALPHA
                            _ptr_dbl,  # A
                            _ptr_int,  # LDA
                            _ptr_dbl,  # B
                            _ptr_int)  # LDB
numba_dtrsm = functype_dtrsm(addr_dtrsm)

addr_dgeqrf = get_cython_function_address('scipy.linalg.cython_blas', 'dgeqrf')
functype_dgeqrf = ctypes.CFUNCTYPE(None,
                            _ptr_int,  # M
                            _ptr_int,  # N
                            _ptr_dbl,  # A
                            _ptr_int,  # LDA
                            _ptr_dbl,  # TAU
                            _ptr_dbl,  # WORK
                            _ptr_int,  # LWORK
                            _ptr_int)  # INFO
numba_dgeqrf = functype_dgeqrf(addr_dgeqrf)

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
    NRHS = np.array(0, dtype=np.int32)
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
    from numba.extending import get_cython_function_address

    addr_dpbtrf = get_cython_function_address('scipy.linalg.cython_lapack', 'dpbtrf')
    functype_dpbtrf = ctypes.CFUNCTYPE(None,
                                _ptr_int,  # UPLO
                                _ptr_int,  # N
                                _ptr_int,  # KD
                                _ptr_dbl,  # AB
                                _ptr_int,  # LDAB
                                _ptr_int)  # INFO
    numba_dpbtrf = functype_dpbtrf(addr_dpbtrf)

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
def dtrsm(a, b, lower=False, overwrite_b=False, transpose=False):
    if lower:
        UPLO = 'L'
    else:
        UPLO = 'U'

    if transpose:
        TRANSA = 'T'
    else:
        TRANSA = 'N'

    SIDE = 'L'
    DIAG = 'N'

    # if b.ndim == 1:
    #     b = np.atleast_2d(b).T

    if b.flags.f_contiguous:
        if not overwrite_b:
            b = np.copy(b)
    else:
        b = np.asfortranarray(b)

    a = np.asfortranarray(a)

    # if b.ndim == 1:
    #     N = 0
    # else:
    #     N = b.shape[1]

    SIDE = np.array([ord(SIDE)], dtype=np.int32)
    UPLO = np.array([ord(UPLO)], dtype=np.int32)
    TRANSA = np.array([ord(TRANSA)], dtype=np.int32)
    DIAG = np.array([ord(DIAG)], dtype=np.int32)
    M = np.array(b.shape[0], dtype=np.int32)
    N = np.array(0, dtype=np.int32)
    ALPHA = np.array(1.0)
    A = a
    LDA = np.array(a.shape[0], dtype=np.int32)
    B = b
    LDB = np.array(b.shape[0], dtype=np.int32)

    numba_dtrsm(SIDE.ctypes,
                UPLO.ctypes,
                TRANSA.ctypes,
                DIAG.ctypes,
                M.ctypes,
                N.ctypes,
                ALPHA.ctypes,
                A.ctypes,
                LDA.ctypes,
                B.ctypes,
                LDB.ctypes)
    return B


@njit(cache=True, parallel=True, fastmath=True, nogil=True)
def backsolve(A, b, lower=False, transpose=False):
    out = dtrsm(A, b, lower=lower, transpose=transpose)
    return out

@njit(cache=True, parallel=True, fastmath=True, nogil=True)
def forwardsolve(A, b, lower=True, transpose=False):
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

@njit(cache=True)
def cholesky_banded(ab, lower=False, overwrite_ab=False):
    c, info = dpbtrf(ab, lower=lower, overwrite_ab=overwrite_ab)

    if info > 0:
        raise ValueError(f"{info}-th leading minor not positive definite")
    if info < 0:
        raise ValueError(f'illegal value in {-info}-th argument of internal pbtrf')
    return c

@njit(cache=True)
def cho_solve_banded(cb_and_lower, b, overwrite_b=False):
    (cb, lower) = cb_and_lower

    if b.ndim != 1:
        raise ValueError(f"b must be 1d, but is {b.ndim}")

    x, info = dpbtrs(cb, b, lower=lower, overwrite_b=overwrite_b)

    if info > 0:
        raise ValueError(f"{info}th leading minor not positive definite")
    if info < 0:
        raise ValueError(f'illegal value in {-info}th argument of internal pbtrs')
    return x


# # from https://github.com/numba/numba/blob/main/numba/np/linalg.py
# @overload(np.linalg.qr)
# def qr_impl(a, mode='r'):
#     # numba.np.linalg.ensure_lapack()

#     # numba.np.linalg._check_linalg_matrix(a, "qr")

#     # Need two functions, the first computes R, storing it in the upper
#     # triangle of A with the below diagonal part of A containing elementary
#     # reflectors needed to construct Q. The second turns the below diagonal
#     # entries of A into Q, storing Q in A (creates orthonormal columns from
#     # the elementary reflectors).

#     from numba.core import types
#     fatal_error_func = types.ExternalFunction("numba_fatal_error", types.intc())

#     def numba_ez_geqrf_impl(dtype):
#         sig = types.intc(
#             types.intp,             # kind
#             types.intp,             # m
#             types.intp,             # n
#             types.CPointer(dtype),  # a
#             types.intp,             # lda
#             types.CPointer(dtype),  # tau
#         )
#         return types.ExternalFunction("numba_ez_geqrf", sig)

#     dtype = np.float64
#     kind = np.int32(ord('d')) # Call the double version dgeqrf
#     numba_ez_geqrf = numba_ez_geqrf_impl(dtype)

#     def qr_impl(a, mode='r'):
#         n = a.shape[-1]
#         m = a.shape[-2]

#         if n == 0 or m == 0:
#             raise np.linalg.LinAlgError("Arrays cannot be empty")

#         # numba.np.linalg._check_finite_matrix(a)

#         # copy A as it will be destroyed
#         q = numba.np.linalg._copy_to_fortran_order(a)

#         minmn = min(m, n)
#         tau = np.empty((minmn), dtype=dtype)

#         ret = numba_ez_geqrf(
#             kind,  # kind
#             m,  # m
#             n,  # n
#             q.ctypes,  # a
#             m,  # lda
#             tau.ctypes  # tau
#         )
#         if ret < 0:
#             fatal_error_func()
#             assert 0   # unreachable

#         # pull out R, this is transposed because of Fortran
#         r = np.zeros((n, minmn), dtype=dtype).T

#         # the triangle in R
#         for i in range(minmn):
#             for j in range(i + 1):
#                 r[j, i] = q[j, i]

#         # and the possible square in R
#         for i in range(minmn, n):
#             for j in range(minmn):
#                 r[j, i] = q[j, i]

#         if mode == 'r':
#             return r

#         # Only mode R supported
#         fatal_error_func()

#     return qr_impl


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

    # all_mus = [None] * N
    # all_lambdas = [None] * N
    # all_betas = [None] * N
    # all_dfs = [None] * N

    # for i in range(N):
    #     all_mus[i] = [None] * nsteps
    #     all_lambdas[i] = [None] * nsteps
    #     all_betas[i] = [None] * nsteps
    #     all_dfs[i] = [None] * nsteps

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

    all_mus = [None] * N
    all_lambdas = [None] * N
    all_betas = [None] * N
    all_dfs = [None] * N

    for i in range(N):
        all_mus[i] = [None] * nsteps
        all_lambdas[i] = [None] * nsteps
        all_betas[i] = [None] * nsteps
        all_dfs[i] = [None] * nsteps

    for i in range(N):
        all_betas[i][0] = H @ rhs[:, i]
        all_lambdas[i][0] = h0[i]
        all_mus[i][0] = u0[:, i]
        all_dfs[i][0] = 1

    # Q_all = Q.copy()
    # R_all = R.copy()
    # Xty_all = Xty.copy()
    # XtX = X.T @ X
    # Xty = np.zeros(p)
    # K = np.zeros(N, dtype=np.int16)

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



@numba.njit(cache=True, nogil=True)
def inner_path(X, y, D, Xty, i0, u0, h0, lmin, nsteps, eps):
    N = y.shape[1]
    p = X.shape[1]
    n = X.shape[0]
    m = D.shape[0]

    XtX = X.T @ X

    for i in range(N):
        B = np.zeros(m, dtype=numba.boolean)
        # B = np.zeros(m, dtype=bool)
        S = np.zeros(m)
        Xty_local = np.zeros(p)
        H = np.ones((p, 1))
        newH = np.ones((p, 1))
        muk = np.zeros(m)

        k = 0
        i0_local = i0[i]
        newH[:i0_local + 1] = 0
        Xty_local[:] = Xty[:, i]

        B[i0_local] = True
        S[i0_local] = np.sign(u0[i0_local, i])

        H = np.hstack((H, newH))
        XH = X @ H
        XtXH = XtX @ H

        # stuff to return
        mus = np.zeros((nsteps, m))
        lambdas = np.zeros(nsteps)
        betas = np.zeros((nsteps, p))
        dfs = np.zeros(nsteps, dtype=np.int16)
        K = np.zeros(N, dtype=np.int16)

        lambdas[0] = h0[i]

        while np.abs(lambdas[k]) > lmin and k < (nsteps - 1):
            # print(k, lambdas[k])

            Db = D[B]
            Dm = D[~B]
            s = S[B]
            Dts = Db.T @ s
            DDt = Dm @ Dm.T
            DDt_banded = diagonal_choform(DDt)
            L_banded = cholesky_banded(DDt_banded)

            # Reset the QR to prevent accumulating errors during long runs
            # if k % 1 == 0:
            Q, R = np.linalg.qr(XH)
            # print(R.shape)
            # print(R)
            # rhs = H.T @ np.vstack((Xty, Dts)).T
            # rhs = np.zeros((H.shape[1], 2))
            # rhs[:, 0] = H.T @ Xty
            # rhs[:, 1] = H.T @ Dts
            A1 = backsolve(R, forwardsolve(R, H.T @ Xty_local, lower=False, transpose=True))
            A2 = backsolve(R, forwardsolve(R, H.T @ Dts, lower=False, transpose=True))
            # A1 = A[:, 0]
            # A2 = A[:, 1]

            # step 3a
            if Dm.shape[0] == 0: # Interior is empty
                hk = 0
                a = np.zeros(0)
                b = np.zeros(0)
            else:
                v = Xty_local - XtXH @ A1
                w = Dts - XtXH @ A2

                # rhs = Dm @ np.vstack((v, w)).T
                # rhs = np.zeros((Dm.shape[0], 2))
                # rhs[:, 0] = Dm @ v
                # rhs[:, 1] = Dm @ w
                # A = cho_solve_banded((L_banded, False), rhs)
                a = cho_solve_banded((L_banded, False), Dm @ v)
                b = cho_solve_banded((L_banded, False), Dm @ w)
                # a = A[:, 0]
                # b = A[:, 1]

                # step 3b
                # hitting time
                t = a / (b + np.sign(a) + eps) # prevent divide by 0
                # numerical issues fix
                t[t > lambdas[k] + eps] = 0

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
                tl[tl > lambdas[k] + eps] = 0

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
                # H = np.insert(H, newpos + 1, newH, axis=1)
                H = np.ones((H.shape[0], H.shape[1] + 1))
                # XH = np.insert(XH, newpos + 1, X @ newH, axis=1)
                XH = X @ H
                # XtXH = np.insert(XtXH, newpos + 1, XtX @ newH, axis=1)
                XtXH = XtX @ H
                XtXH = X.T @ XH
                # print(H.shape, R.shape, newpos, newH.shape, XH.shape)
                # print(f'add {ik, coord}, B={B.sum()}')
            elif lk > hk: # variable leaves the boundary
                coord = np.nonzero(B)[0][ilk]
                updates = False, 0
                lambdak = lk
                # Q, R = la.qr_delete(Q, R, ilk + 1, which='col', overwrite_qr=True, check_finite=False)
                H = np.ones((H.shape[0], H.shape[1] - 1))
                XH = X@H
                XTXH = XtX@H
                # H = np.delete(H, ilk + 1, axis=1)
                # XH = np.delete(XH, ilk + 1, axis=1)
                # XtXH = np.delete(XtXH, ilk + 1, axis=1)
                # print(f'remove {ilk, coord}, B={B.sum()}')
            elif (lk == 0) and (hk == 0): # end of the path, so everything stays as is
                lambdak = 0
                # print('end of the path reached')

            k += 1
            muk[~B] = a - lambdak * b
            muk[B] = lambdak * s
            # print(a.shape, b.shape, lambdak, s.shape, muk.shape)
            mus[k] = muk

            # update boundaries
            B[coord], S[coord] = updates
            betas[k] = HA1 - lambdak * HA2 # eq. 38
            lambdas[k] = lambdak
            dfs[k] = H.shape[1] - 1

        K[i] = k

@njit()
def callqr(a, mode):
    return np.linalg.qr(a, mode)