# cython: language_level=3, boundscheck=False, infer_types=True, initializedcheck=False, cdivision=True,  linetrace=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

import cython
import numpy as np
cimport numpy as np

from scipy.linalg.cython_blas cimport dtrsm as cython_dtrsm
from scipy.linalg.cython_lapack cimport dpbtrf as cython_dpbtrf
from scipy.linalg.cython_lapack cimport dpbtrs as cython_dpbtrs
from scipy.linalg.cython_lapack cimport dpotrf as cython_dpotrf
from scipy.linalg.cython_lapack cimport dpotrs as cython_dpotrs

import scipy.linalg as la
# import igraph as ig

from numpy.linalg._umath_linalg import qr_r_raw_m, qr_r_raw_n
from libc.math cimport sqrt
# from scipy.linalg._decomp_update import qr_insert, qr_delete

cdef inline double[::1] dpbtrs(double[::1,:] AB, double[::1] B, bint lower=False, bint overwrite_b=False):
    cdef:
        char UPLO
        int N = AB.shape[1]
        int KD = AB.shape[0] - 1
        int NRHS = 1
        int LDAB = AB.shape[0]
        int LDB = B.shape[0]
        int INFO

    if lower:
        UPLO = b'L'
    else:
        UPLO = b'U'

    cython_dpbtrs(&UPLO,
                &N,
                &KD,
                &NRHS,
                &AB[0,0],
                &LDAB,
                &B[0],
                &LDB,
                &INFO)

    if INFO > 0:
        raise ValueError(f"{INFO}th leading minor not positive definite")
    if INFO < 0:
        raise ValueError(f'illegal value in {-INFO}th argument of internal dpbtrs')

    return B


cdef inline double[::1] dpotrs(double[::1,:] A, double[::1] B, bint lower=False, bint overwrite_b=False):
    cdef:
        char UPLO
        int N = A.shape[0]
        int NRHS = 1
        int LDA = A.shape[0]
        int LDB = B.shape[0]
        int INFO

    if lower:
        UPLO = b'L'
    else:
        UPLO = b'U'

    cython_dpotrs(&UPLO,
                &N,
                &NRHS,
                &A[0,0],
                &LDA,
                &B[0],
                &LDB,
                &INFO)

    if INFO > 0:
        raise ValueError(f"{INFO}th leading minor not positive definite")
    if INFO < 0:
        raise ValueError(f'illegal value in {-INFO}th argument of internal dpotrs')

    return B


cdef inline double[::1,:] dpbtrf(double[::1,:] AB, bint lower=False, bint overwrite_ab=False):
    # abtemp = np.zeros((ab.shape[0], ab.shape[1]), order='F', dtype=np.float64)

    cdef:
        char UPLO
        int N = AB.shape[1]
        int KD = AB.shape[0] - 1
        int LDAB = AB.shape[0]
        int INFO = 0

        # double[:,:] AB = ab.copy_fortran()

    if lower:
        UPLO = b'L'
    else:
        UPLO = b'U'

    # if b.ndim == 1:
    #     b = np.atleast_2d(b).T

    # if ab.flags.f_contiguous:
    #     AB[:] = ab
    # else:
    #     AB[:] = np.asfortranarray(ab)

    # AB[:] = ab.copy_fortran()

    # UPLO = np.array([ord(UPLO)], dtype=np.int32)
    # N = np.array(ab.shape[1], dtype=np.int32)
    # KD = np.array(ab.shape[0] - 1, dtype=np.int32)
    # AB = ab
    # LDAB = np.array(ab.shape[0], dtype=np.int32)
    # INFO = np.array(0, dtype=np.int32)

    cython_dpbtrf(&UPLO,
                &N,
                &KD,
                &AB[0,0],
                &LDAB,
                &INFO)

    if INFO > 0:
        raise ValueError(f"{INFO}th leading minor not positive definite")
    if INFO < 0:
        raise ValueError(f'illegal value in {-INFO}th argument of internal dpbtrf')

    return AB


cdef inline double[::1,:] dpotrf(double[::1,:] A, bint lower=False, bint overwrite_a=False):
    cdef:
        char UPLO
        int N = A.shape[1]
        int LDA = A.shape[0]
        int INFO = 0

        # double[:,:] A = a.copy_fortran()

    if lower:
        UPLO = b'L'
    else:
        UPLO = b'U'

    # if b.ndim == 1:
    #     b = np.atleast_2d(b).T

    # if ab.flags.f_contiguous:
    #     AB[:] = ab
    # else:
    #     AB[:] = np.asfortranarray(ab)

    # AB[:] = ab.copy_fortran()

    # UPLO = np.array([ord(UPLO)], dtype=np.int32)
    # N = np.array(ab.shape[1], dtype=np.int32)
    # KD = np.array(ab.shape[0] - 1, dtype=np.int32)
    # AB = ab
    # LDAB = np.array(ab.shape[0], dtype=np.int32)
    # INFO = np.array(0, dtype=np.int32)

    cython_dpotrf(&UPLO,
                &N,
                &A[0,0],
                &LDA,
                &INFO)

    if INFO > 0:
        raise ValueError(f"{INFO}th leading minor not positive definite")
    if INFO < 0:
        raise ValueError(f'illegal value in {-INFO}th argument of internal pbtrs')

    return A


cdef inline double[:] dtrsm(double[::1,:] A, double[::1] B, bint lower=False, bint transpose=False) noexcept nogil:
    # atemp = np.zeros((a.shape[0], a.shape[1]), order='F', dtype=np.float64)
    # btemp = np.zeros(b.shape[0], order='F', dtype=np.float64)

    cdef:
        char SIDE, UPLO, TRANSA, DIAG
        int M = B.shape[0]
        int N = 0
        double ALPHA = 1.0
        int LDA = A.shape[0]
        int LDB = B.shape[0]

        # double[:,:] A = a.copy_fortran()
        # double[:] B = b.copy_fortran()

    if lower:
        UPLO = b'L'
    else:
        UPLO = b'U'

    if transpose:
        TRANSA = b'T'
    else:
        TRANSA = b'N'

    SIDE = b'L'
    DIAG = b'N'

    # if b.flags.f_contiguous:
    #     B[:] = b
    # else:
    #     B[:] = np.asfortranarray(b)

    # A[:] = np.asfortranarray(a)
    # A[:] = a.copy_fortran()
    # B[:] = b.copy_fortran()

    # if b.ndim == 1:
    #     N = 0
    # else:
    #     N = b.shape[1]

    # SIDE = np.array([ord(SIDE)], dtype=np.int32)
    # UPLO = np.array([ord(UPLO)], dtype=np.int32)
    # TRANSA = np.array([ord(TRANSA)], dtype=np.int32)
    # DIAG = np.array([ord(DIAG)], dtype=np.int32)
    # M = np.array(b.shape[0], dtype=np.int32)
    # N = np.array(0, dtype=np.int32)
    # ALPHA = np.array(1.0)
    # A = a
    # LDA = np.array(a.shape[0], dtype=np.int32)
    # B = b
    # LDB = np.array(b.shape[0], dtype=np.int32)

    cython_dtrsm(&SIDE,
                &UPLO,
                &TRANSA,
                &DIAG,
                &M,
                &N,
                &ALPHA,
                &A[0,0],
                &LDA,
                &B[0],
                &LDB)
    return B

# cdef inline void posQR(double[:, ::1] Q, double[:, ::1] R):
#     cdef int i, j, k
#     cdef int r0 = R.shape[0]
#     cdef int r1 = R.shape[1]
#     cdef int q0 = Q.shape[0]

#     for i in range(r0):
#         if R[i, i] < 0:
#             for j in range(q0):
#                 Q[j, i] = -Q[j, i]
#             for k in range(r1):
#                 R[i, k] = -R[i, k]


# cdef inline double[:] backsolve(double[:,:] A, double[:] b, bint lower=False, bint transpose=False):
#     out = dtrsm(A.copy_fortran(), b.copy_fortran(), lower=lower, transpose=transpose)
#     return np.asarray(out)

# cdef inline double[:] forwardsolve(double[:,:] A, double[:] b, bint lower=True, bint transpose=False):
#     return backsolve(A, b, lower=lower, transpose=transpose)

cdef inline double[:,::1] diagonal_choform(double[:,::1] a) noexcept:
    n = a.shape[1]
    ab = np.zeros((2, n))

    diagu = np.diagonal(a, offset=1)
    diag = np.diagonal(a)

    ab[0, 1:] = diagu
    ab[1, :] = diag

    return ab

cdef inline double[:,:] cholesky_banded(double[:,::1] ab, bint lower=False, bint overwrite_ab=False) noexcept:
    c = dpbtrf(ab.copy_fortran(), lower=lower, overwrite_ab=overwrite_ab)
    return c

cdef inline double[:] cho_solve_banded(double[:,:] cb, double[::1] b, bint lower=False, bint overwrite_b=False) noexcept:
    x = dpbtrs(cb.copy_fortran(), b.copy_fortran(), lower=lower, overwrite_b=overwrite_b)
    return x


cdef inline double[:] cho_solve(double[:,:] c, double[:] b, bint lower=False, bint overwrite_b=False) noexcept:
    x = dpotrs(c.copy_fortran(), b.copy_fortran(), lower=lower, overwrite_b=overwrite_b)
    return x


cdef inline double[:,:] cho_factor(double[:,::1] a, bint lower=False, bint overwrite_a=False) noexcept:
    c = dpotrf(a.copy_fortran(), lower=lower, overwrite_a=overwrite_a)
    return c


cdef inline double[:,:] qr(double[:,::1] a) noexcept:
    cdef:
        int m = a.shape[0]
        int n = a.shape[1]
        int max_mn = max(m, n)
        int min_mn = min(m, n)
        double[:,:] temp = a.copy()
        # double[:] tau = np.zeros(max_mn, dtype=np.float64)
        # double[:,:] r = np.zeros((min_mn, min_mn), dtype=np.float64)
        # double[:,:] out = np.zeros((max_mn, max_mn), dtype=np.float64)

    # out = np.zeros((max_mn, max_mn), dtype=np.float64)

    if m <= n:
        qr_r_raw_m(temp)
    else:
        qr_r_raw_n(temp)

    # return np.triu(temp[:min_mn, :])
    return temp[:min_mn, :]
    # print(temp.shape, m, n, max_mn, min_mn, r.shape, out.shape)

    # m, n = r.shape[:2]

    # if m < n:
    #     out[:m, :] = r
    #     return out
    # return r

@cython.profile(False)
cdef inline double cmean(double[:] a) noexcept nogil:
    cdef:
        int i = a.shape[0]
        int j
        double out = 0

    if i == 1:
        return a[0]

    for j in range(i):
        out += a[j]
    return out / i

@cython.profile(False)
cdef inline void cmean2(double[::1,:] a, double[:] out) noexcept nogil:
    cdef:
        int m = a.shape[0]
        int n = a.shape[1]
        int i, j

    if n == 1:
        out[:] = a[:, 0]
    else:
        for i in range(m):
            out[i] = cmean(a[i])
            # for j in range(n):
            #     out[i] += a[i, j]
            # out[i] /= n

cdef inline void make_diag_pos(double[:, :] Q, double[:, :] R) noexcept nogil:
    cdef:
        int k = Q.shape[0]
        int m = R.shape[0]
        int n = R.shape[1]
        # int n = Q.shape[1]
        int i, j

    for i in range(m):
        if R[i, i] < 0:
            for j in range(k):
                Q[j, i] = -Q[j, i]
            for j in range(i, m):
                R[i, j] = -R[i, j]



# cdef inline double[:,:] add_col_R(double[:,:] R, double[:,:] A, double[::1] a) noexcept nogil:
#     cdef:
#         int m = A.shape[0]
#         int n = A.shape[1]
#         int i, j, o, k
#         double sum1, sum2, gamma, norma. normu
#         double[:] u = np.zeros(m, dtype=np.float64)
#         double[:,:] out = np.zeros((m+1, m+1), dtype=np.float64)

#     for i in range(m):
#         for j in range(n):
#             k = i - j
#             if k > 0:
#                 for o in range(k):
#                     sum2 += u[o] * r[i,o]
#             else:
#                 sum2 = 0
#             sum1 += a[i,j] - a[j] - sum2
#         u[i] = 1.0 / r[i,i] * (sum1 - sum2)

#     norma = np.sum(a**2)
#     normu = np.sum(u**2)
#     gamma = sqrt(a - u)

#     out[:m, :m] = R
#     out[m+1, :m+1] = u
#     out[m+1, m+1] = gamma

#     return out

# def print(*a):
#     pass

def inner_path(np.ndarray[double, ndim=2] X,
               np.ndarray[double, ndim=2] y,
               np.ndarray[double, ndim=2] D,
               np.ndarray[double, ndim=2] Xty,
               np.ndarray[np.int16_t, ndim=1] i0,
               np.ndarray[np.int16_t, ndim=1] S0,
               np.ndarray[np.float32_t, ndim=1] h0,
               double lmin, int nsteps, double eps):

    cdef:
        int N = y.shape[1]
        int p = X.shape[1]
        int n = X.shape[0]
        int m = D.shape[0]
        int i, k, i0_local, splits, df, idx, r, cutoff, nextidx, coord
        double lambdak, lk, hk
        list new_HtXty, new_HtDts
        np.ndarray[double, ndim=1] HtXty
        np.ndarray[double, ndim=1] HtDts
        np.ndarray[double, ndim=2] newXH
        np.ndarray[double, ndim=2] Q
        np.ndarray[double, ndim=2] R
        np.ndarray[double, ndim=1] t
        np.ndarray[double, ndim=1] tl
        # np.ndarray[np.int64_t, ndim=1] indices

    # np.int16_t[:] K = np.zeros(N, dtype=np.int16)
    XtX = X.T @ X
    X = np.asfortranarray(X)
    # DtD = D.T @ D
    # np.fill_diagonal(DtD, 0)
    # DtD = DtD.astype(bool)

    B = np.zeros(m, dtype=bool)
    S = np.zeros(m, dtype=np.int16)
    Xty_local = np.zeros(p)
    # H0 = np.ones((p, 1)) / p
    # newH = np.ones((p, 1))
    muk = np.zeros(m)
    all_indexes = np.arange(p)

    # L = np.abs(D.T @ D)
    # np.fill_diagonal(L, 0)
    # graph = ig.Graph.Adjacency(L, mode='upper')
    # clusters = graph.connected_components()

    # stuff to return
    all_mus = np.zeros((N, nsteps, m))
    all_lambdas = np.zeros((N, nsteps))
    all_betas = np.zeros((N, nsteps, p))
    all_dfs = np.zeros((N, nsteps), dtype=np.int16)
    all_Ks = np.zeros(N, dtype=np.int16)

    # Xmean = X.mean(axis=1, keepdims=True)
    Xty_mean = Xty.mean(axis=0, keepdims=True)
    Q_orig, R_orig = np.linalg.qr(X.mean(axis=1, keepdims=True))
    # lmin = max(lmin, eps)

    for i in range(N):
        B[:] = 0
        S[:] = 0
        # newH[:] = 1
        muk[:] = 0

        k = 0
        i0_local = i0[i]
        # newH[:i0_local + 1] = 0
        Xty_local[:] = Xty[:, i]

        B[i0_local] = True
        S[i0_local] = S0[i]

        # H = np.append(H0, newH, axis=1)
        # XH = X @ H
        # DtD_local = DtD.copy()
        # DtD_local[i0_local, :] = 0
        # DtD_local[:, i0_local] = 0
        # DtD_local[i0_local, i0_local - 1] = 0
        # DtD_local[i0_local, i0_local + 1] = 0
        # DtD_local[i0_local - 1, i0_local] = 0
        # DtD_local[i0_local + 1, i0_local] = 0
        # XH = X[DtD_local].mean(axis=1, keepdims=True)

        # nonzeros = np.nonzero(DtD_local)

        # XH[:, 0] = np.mean(X[:, :i0_local], axis=1)
        # HtXty[0] = np.mean(Xty_local[:i0_local])

        # XH[:, 1] = np.mean(X[:, i0_local+1:], axis=1)
        # HtXty[1] = np.mean(Xty_local[i0_local+1:])

        # XtXH = XtX @ H

        mus = all_mus[i]
        lambdas = all_lambdas[i]
        betas = all_betas[i]
        dfs = all_dfs[i]
        lambdas[0] = h0[i]
        r = 1
        cutoff = i0_local

        # XH_prev = X.mean(axis=1, keepdims=True)

        nsplits_prev = 0
        HtXty = Xty_mean[:, i]
        HtDts = np.zeros(1)

        Q = Q_orig
        R = R_orig

        # print('initial cutoff', cutoff, np.nonzero(~B)[0])

        while lambdas[k] > lmin and k < (nsteps - 1):
            # print(k, lambdas[k])

            Db = D[B]
            Dm = D[~B]
            s = S[B]
            Dts = Db.T @ s

            consec = np.split(all_indexes, np.nonzero(B)[0] + 1)
            nsplits = len(consec)
            # nsplits = B.sum() + 1
            # print(nsplits, nsplits1)

            # # ################
            # XH_old = np.zeros((X.shape[0], nsplits))
            # # HtXty_old = np.zeros(nsplits)
            # # HtDts_old = np.zeros(nsplits)

            # iterator = enumerate(consec)
            # nextidx = 0

            # for idx, indices in iterator:
            #     # print(k, idx ,indices, cutoff, consec)
            #     if indices[-1] == cutoff: # new cutoff point in the graph
            #         cmean2(X[:, indices], XH_old[:, idx])
            #         # HtXty_old[idx] = cmean(Xty_local[indices])
            #         # HtDts_old[idx] = cmean(Dts[indices])

            #         # process the next group as it also changed if we are not at the end
            #         idx, indices = next(iterator, (None, None))
            #         if idx is not None:
            #             cmean2(X[:, indices], XH_old[:, idx])
            #             # HtXty_old[idx] = cmean(Xty_local[indices])
            #             # HtDts_old[idx] = cmean(Dts[indices])

            #             # we now have to copy the next index from the old array to the new array
            #             # go back one if we add a new column to the new array
            #             # skip one extra instead if we merge two columns in the new array
            #             if nsplits > XH_prev.shape[1]:
            #                 nextidx = -1
            #             else:
            #                 nextidx = 1

            #     else: # copy the old data if it didn't change
            #         XH_old[:, idx] = XH_prev[:, idx+nextidx]
            #         # HtXty_old[idx] = HtXty_prev[idx+nextidx]
            #         # HtDts_old[idx] = HtDts_prev[idx+nextidx]

            # # ################

            # Q, R = np.linalg.qr(XH_old)
            # XH_prev = XH_old
            # print('shapes', Q.shape, R.shape, XH_old.shape)
            # iterator = enumerate(consec)
            # for idx, indices in iterator:
            #     if indices[-1] == cutoff: # new cutoff point in the graph

            # raise ValueError()

            # print('consec', consec)
            all_indices = np.nonzero(B)[0]
            # print('indices', all_indices, cutoff)
            idx = np.nonzero(all_indices == cutoff)[0]
            # all_indices += 1
            # print(consec)
            # print(nsplits, nsplits_prev)
            # print(all_indices, idx, cutoff, all_indices.shape, all_indices.ndim)
            # raise ValueError()
            if nsplits > nsplits_prev:
                addcol = 2
                delcol = 1
                if idx == 0:
                    indices = slice(0, all_indices[idx] + 1)
                else:
                    indices = slice(all_indices[idx - 1] + 1, all_indices[idx] + 1)
                # slice on the right
                # if idx == 0:
                #     indices = slice(0, all_indices[idx] + 1)
                # # elif idx == len(all_indices) - 1:
                #     # indices = slice(all_indices[idx-1] + 1, m + 1)
                # else:
                #     indices = slice(all_indices[idx-1] + 1, all_indices[idx] + 1)
            else:
                addcol = 1
                delcol = 2
                # print('in delcol', addcol, delcol, idx, cutoff ,all_indices)
                # if all_indices[idx] == 1:
                    # indices = slice(0, all_indices[0])
                if idx == 0:
                    indices = slice(0, all_indices[0] + 1)
                else:
                    indices = slice(all_indices[idx - 1] + 1, all_indices[idx] + 1)
                # slice on the left
                # if idx == 0:
                    # indices = slice(0, all_indices[idx] + 1)
                # idx += 1

            # print(idx, all_indices, cutoff, len(all_indices))


            # print('current', indices, idx, all_indices)


                # _, indices = next(iterator)
                # print('new indices', indices)

            newXH = np.zeros((X.shape[0], addcol))

            cmean2(X[:, indices], newXH[:, 0])
            new_HtXty = [cmean(Xty_local[indices])]
            new_HtDts = [cmean(Dts[indices])]

            if addcol == 2:
                # print('previous set', indices)
                # _, indices = next(iterator)
                # slice on the right
                # if idx == 0:
                #     indices = slice(0, all_indices[idx] + 1)
                if idx == len(all_indices) - 1:
                    indices = slice(all_indices[idx] + 1, m + 1)
                else:
                    indices = slice(all_indices[idx] + 1, all_indices[idx+1] + 1)
                cmean2(X[:, indices], newXH[:, 1])
                new_HtXty.append(cmean(Xty_local[indices]))
                new_HtDts.append(cmean(Dts[indices]))

            if delcol == 2:
                todel = (idx, idx + 1)
                toins = idx
            else:
                todel = (idx,)
                toins = idx

            # print('newXH', newXH)
            # print('QR', Q@R)
            # print('new_HtXty', new_HtXty)
            # # print(new_HtDts)
            # print(cutoff, indices, all_indices, idx, consec, todel, toins)
            # print('QR update', k, idx ,indices, cutoff, addcol, delcol, consec)
            # print('QR update', Q.shape, R.shape, idx, delcol, cutoff)
            Q, R = la.qr_delete(Q, R, todel[0], p=delcol, which='col', check_finite=False)
            # print('QR update', Q.shape, R.shape)
            Q, R = la.qr_insert(Q, R, newXH, toins, which='col', overwrite_qru=False, check_finite=False)
            # print('QR update', Q.shape, R.shape)
            # print('QR updated', Q@R)

            # print('before del', todel, idx, addcol, delcol, HtXty.shape[0], HtDts.shape[0])
            # print(HtXty)
            HtXty = np.delete(HtXty, todel)
            HtDts = np.delete(HtDts, todel)

            HtXty = np.insert(HtXty, toins, new_HtXty)
            HtDts = np.insert(HtDts, toins, new_HtDts)
            # print('after del', todel, toins, idx, addcol, delcol, HtXty.shape[0], HtDts.shape[0])
            # print(HtXty, '\n')

            # If Q is square, the downdate will change to full mode instead of economic
            # so we strip the (possible) zeros here to make R square again
            if R.shape[1] < R.shape[0]:
                # print('resize', Q.shape, R.shape)
                R = R[:R.shape[1], :R.shape[1]]
                Q = Q[:, :R.shape[1]]
                # print('resize', Q.shape, R.shape)


            # diagneg = np.diagonal(R) < 0
            # Q[:, diagneg] *= -1
            # R[diagneg] *= -1

            # print(XH_prev)
            # print(XH_old)
            # print(Q@R)
            # XH_prev = XH_old
            # HtXty_prev = HtXty_old
            # HtDts_prev = HtDts_old
            nsplits_prev = nsplits

            # HtXty = HtXty_old
            # HtDts = HtDts_old

            # print('diff QR', np.abs(XH.T@XH - R.T@R).max(), np.abs(XH - Q@R).max(), np.shape(XH), np.shape(Q), R.shape)
            # print('Q', Q)
            # print('R', R)
            # R1 = qr(XH)
            # Q, R = np.linalg.qr(XH)
            # print('diff', np.abs(Q@R - XH).max(), np.abs(R - np.triu(R1)).max())

            # rhs = H.T @ np.vstack((Xty, Dts)).T
            # rhs = np.zeros((H.shape[1], 2))
            # rhs[:, 0] = H.T @ Xty
            # rhs[:, 1] = H.T @ Dts
            # A1 = backsolve(R, forwardsolve(R, H.T @ Xty_local, lower=False, transpose=True))
            # A2 = backsolve(R, forwardsolve(R, H.T @ Dts, lower=False, transpose=True))
            # A1 = np.linalg.lstsq(R, np.linalg.lstsq(R.T, H.T @ Xty_local, rcond=-1)[0], rcond=-1)[0]
            # A2 = np.linalg.lstsq(R, np.linalg.lstsq(R.T, H.T @ Dts, rcond=-1)[0], rcond=-1)[0]
            # A1 = A[:, 0]
            # A2 = A[:, 1]

            # chofac = cho_factor(XH.T @ XH)
            # A1new = cho_solve(chofac, H.T @ Xty_local)
            # A2new = cho_solve(chofac, H.T @ Dts)
            # chofac = la.cho_factor(XH.T @ XH)

            # chofac = np.asarray(R)

            # print('diff', np.abs(Q@R - XH).max(), np.abs(R - np.triu(R1)).max())

            # chofac = cho, False
            # chofac = cho

            # Q, R = np.linalg.qr(XH)
            # R = np.asarray(qr(XH))

            # print(Q)
            # print(R)
            # print(Q.flags, R.flags)
            make_diag_pos(Q, R)
            # print(Q)
            # print(R)
            # print(Q.flags, R.flags)
            # diagneg = np.diagonal(R) < 0
            # Q[:, diagneg] *= -1
            # R[diagneg] *= -1
            # print('R pos', np.diag(R), R.shape, XH.shape)
            # print('Ht', HtXty)
            # print('Ht', HtDts)
            # print('R.T@R', R.shape, R.T@R)
            A1 = cho_solve(R, HtXty)
            A2 = cho_solve(R, HtDts)
            # A1 = np.linalg.lstsq(R.T @ R, HtXty, rcond=-1)[0]
            # A2 = np.linalg.lstsq(R.T @ R, HtDts, rcond=-1)[0]
            # print('A1', np.asarray(A1))
            # print('A2', np.asarray(A2))
            # print('R', np.diag(R), R.shape)

            HA1 = np.zeros(p)
            HA2 = np.zeros(p)

            for idx, indices in enumerate(consec):
                HA1[indices] = A1[idx] / len(indices)
                HA2[indices] = A2[idx] / len(indices)

            # step 3a
            if r == m: # Interior is empty
                hk = 0
                a = np.zeros(0)
                b = np.zeros(0)
            else:
                DDt = Dm @ Dm.T
                DDt_banded = diagonal_choform(DDt)
                # XtXHA = X.T @ Q @ R @ np.vstack((A1, A2)).T
                XtXHA = XtX @ np.vstack((HA1, HA2)).T

                L_banded = cholesky_banded(DDt_banded)
                rhs = np.vstack((Xty_local, Dts)).T - XtXHA
                v = rhs[:, 0]
                w = rhs[:, 1]
                # v = Xty_local - XtXHA[:, 0]
                # w = Dts - XtXHA[:, 1]

                # XtXH1 = X.T @ Q @ R
                # v1 = Xty_local - XtXH @ A1
                # w1 = Dts - XtXH @ A2
                # print(np.abs(v1 - v).max(), np.abs(w1 - w).max(), np.abs(XtXH1 - XtXH).max())
                # rhs = Dm @ np.vstack((v, w)).T
                # rhs = np.zeros((Dm.shape[0], 2))
                # rhs[:, 0] = Dm @ v
                # rhs[:, 1] = Dm @ w
                # A = cho_solve_banded((L_banded, False), rhs)
                a = cho_solve_banded(L_banded, Dm @ v)
                b = cho_solve_banded(L_banded, Dm @ w)

                a = np.asarray(a)
                b = np.asarray(b)

                # a = A[:, 0]
                # b = A[:, 1]

                # step 3b
                # hitting time
                t = a / (b + np.sign(a))
                # numerical issues fix
                t[t > lambdas[k] + eps] = 0
                # t[t >= lambdas[k]] = 0
                ik = np.argmax(t)
                hk = t[ik]

                # print('t', t)
                # print('a', a)
                # print('b', b)
                # print('v', v)
                # print('w', w)

                if hk < lmin:
                    hk = 0

            # leaving time
            # print(H.shape, A1.shape, A2.shape, nsplits, k)
            # HA1 = H @ A1
            # HA2 = H @ A2


            if r == 0: # or HA1.shape[0] == HA1.shape[1]: # Boundary is empty
                lk = 0
            else:
                # print(s.shape, Db.shape, HA1.shape, HA2.shape)
                c = s * (Db @ HA1)
                d = s * (Db @ HA2)

                tl = np.where((c < 0) & (d < 0), c/(d + eps), 0)
                # tl = np.divide(c, d, where=(c < 0) & (d < 0), out=np.zeros_like(c))
                # numerical issues fix
                tl[tl > lambdas[k] + eps] = 0
                # tl[tl >= lambdas[k]] = 0

                ilk = np.argmax(tl)
                lk = tl[ilk]

                if lk < lmin:
                    lk = 0

                # print('c', c)
                # print('d', d)
                # print('tl', tl)

            # print('hk', hk)
            # print(t)
            # print('lk', lk)
            # print(tl)
            # update matrices and stuff for next step
            if hk > lk: # variable enters the boundary
                coord = np.nonzero(~B)[0][ik]
                update_B = True
                update_S = np.sign(a[ik])
                lambdak = hk
                # graph.delete_edges([(coord, coord + 1)])
                cutoff = coord
                r += 1
                # print('add cutoff', coord, cutoff, ik, np.nonzero(~B)[0])

                # newH[:] = 1
                # newH[:coord + 1] = 0
                # newpos = np.searchsorted(np.nonzero(B)[0], coord, side='right')
                # # print(np.abs(Q@R - XH).sum(), np.sum(np.diag(R) < 0))
                # # Q, R = np.linalg.qr(XH)
                # # XtXH = np.hstack((XtXH[:, :newpos], XtX @ newH, XtXH[:, newpos:]))
                # H = np.hstack((H[:, :newpos], newH, H[:, newpos:]))
                # XtXH = XtX @ H
                # # XH = X @ H
                # # print(XH.shape, X.copy().shape, H.shape, Q.shape, R.shape)
                # print('add', r, ik, coord)
                # Q, R = la.qr_insert(Q, R, X @ newH, newpos, which='col', overwrite_qru=True, check_finite=False)
                # # print(np.abs(Q@R - XH).sum(), np.sum(np.diag(R) < 0))
                # # print(np.abs(Q@R - XH).sum(), np.sum(np.diag(R) < 0))
                # # print(np.abs(Q1@R1 - XH).sum(), np.sum(np.diag(R1) < 0), XH.shape, Q1.shape, R1.shape)
            elif lk > hk: # variable leaves the boundary
                coord = np.nonzero(B)[0][ilk]
                update_B = False
                update_S = 0
                lambdak = lk
                # print('remove cutoff', coord, lk, ilk, np.nonzero(B)[0])

                # Next index is our cutoff on the right because columns will merge
                # if we are already on the last item then it's our cutoff
                if ilk == len(np.nonzero(B)[0]) - 1:
                    cutoff = np.nonzero(B)[0][ilk - 1]
                else:
                    cutoff = np.nonzero(B)[0][ilk + 1]
                # cutoff = coord
                # print('remove cutoff', coord, cutoff, ilk, np.nonzero(B)[0])
                # graph.add_edges([(coord, coord + 1)])
                r -= 1
                # print('remove', r, ilk, coord)
                # # print(ilk, coord, np.nonzero(B)[0])
                # # print(np.abs(Q@R - XH).sum(), np.sum(np.diag(R) < 0))
                # H = np.delete(H, ilk, axis=1)
                # # XH = np.delete(XH, ilk, axis=1)
                # XtXH = np.delete(XtXH, ilk, axis=1)
                # Q, R = la.qr_delete(Q, R, ilk, which='col', check_finite=False)
                # # print(np.abs(Q@R - XH).sum(), np.sum(np.diag(R) < 0))
                # # print(np.abs(Q@R - XH).sum(), np.sum(np.diag(R) < 0))

            elif (lk == 0) and (hk == 0): # end of the path, so everything stays as is
                # Can be triggered if lambda = 0 is the only and first solution
                lambdak = 0
                coord = 0
                update_B = False
                update_S = 0

            # Ensure diag(R) is always positive for cholesky by flipping signs of negative entries in QR
            # Q1 = Q.copy()
            # R1 = R.copy()
            # idx = np.diag(R) < 0
            # Q[:, idx] *= -1
            # R[idx] *= -1

            # posQR(Q1, R1)
            # print(np.abs(Q1 - Q).max(), np.abs(R1 - R).max(), )
            # print('QR diff',  np.abs(Q@R - XH).sum(), np.isfinite(Q@R).sum(), np.isnan(Q@R).sum(), np.shape(Q@R), hk, lk, lambdak)

            # clusters = graph.connected_components()

            k += 1
            muk[~B] = a - lambdak * b
            muk[B] = lambdak * s
            mus[k] = muk

            # update boundaries
            B[coord] = update_B
            S[coord] = update_S
            betas[k] = HA1 - lambdak * HA2 # eq. 38
            lambdas[k] = lambdak
            dfs[k] = nsplits

        all_Ks[i] = k + 1 # Because k=0 is added in the outside function

    return all_mus, all_lambdas, all_betas, all_dfs, all_Ks

def run(*args):
    inner_path(*args)

def this_function_causes_line_profiler_to_print_all_the_way_down_here():
    pass
