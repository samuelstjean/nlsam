# cython: language_level=3, boundscheck=False, infer_types=True, initializedcheck=False, cdivision=True
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
# from scipy.linalg._decomp_update import qr_insert, qr_delete

cdef inline double[::1] dpbtrs(double[::1,:] AB, double[::1] B, bint lower=False, bint overwrite_b=False):
    # btemp = np.zeros(b.shape[0], order='F', dtype=np.float64)
    # abtemp = np.zeros((cb.shape[0], cb.shape[1]), order='F', dtype=np.float64)

    cdef:
        char UPLO
        int N = AB.shape[1]
        int KD = AB.shape[0] - 1
        int NRHS = 1
        int LDAB = AB.shape[0]
        int LDB = B.shape[0]
        int INFO

        # double[:] B = b.copy_fortran()
        # double[:,:] AB = cb.copy_fortran()

    if lower:
        UPLO = b'L'
    else:
        UPLO = b'U'

    # if b.ndim == 1:
    #     b = np.atleast_2d(b).T

    # if b.flags.f_contiguous:
    #     if not overwrite_b:
    #         B[:] = np.copy(b)
    # else:
    #     B[:] = np.asfortranarray(b)

    # if b.flags.f_contiguous:
    #     B[:] = b
    # else:
    #     B[:] = np.asfortranarray(b)

    # AB[:] = np.asfortranarray(cb)

    # AB[:] = cb.copy_fortran()
    # B[:] = b.copy_fortran()



    # UPLO = np.array([ord(UPLO)], dtype=np.int32)
    # NRHS = np.array(0, dtype=np.int32)
    # N = np.array(cb.shape[1], dtype=np.int32)
    # KD = np.array(cb.shape[0] - 1, dtype=np.int32)
    # AB = cb
    # LDAB = np.array(cb.shape[0], dtype=np.int32)
    # B = b
    # LDB = np.array(b.shape[0], dtype=np.int32)
    # INFO = np.array(0, dtype=np.int32)

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
        raise ValueError(f'illegal value in {-INFO}th argument of internal pbtrs')

    return B


cdef inline double[::1] dpotrs(double[::1,:] A, double[::1] B, bint lower=False, bint overwrite_b=False):
    cdef:
        char UPLO
        int N = A.shape[1]
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
        raise ValueError(f'illegal value in {-INFO}th argument of internal pbtrs')

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
        raise ValueError(f'illegal value in {-INFO}th argument of internal pbtrs')

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


cdef inline double[:] cho_solve(double[:,:] c, double[::1] b, bint lower=False, bint overwrite_b=False) noexcept:
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
        int i0_local, k, splits, df, idx, r, cutoff, nextidx
        double lambdak, lk, hk
        np.ndarray[np.int64_t, ndim=1] indices

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

    # if lmin < eps:
    #     lmin = eps

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
        # K = all_Ks[i]
        lambdas[0] = h0[i]
        # df = 1
        r = 1
        cutoff = i0_local

        XH_prev = X.mean(axis=1, keepdims=True)
        HtXty_prev = Xty_local.mean(keepdims=True)
        HtDts_prev = np.zeros(1)

        while lambdas[k] > lmin and k < (nsteps - 1):
            # print(k, lambdas[k])

            Db = D[B]
            Dm = D[~B]
            s = S[B]
            Dts = Db.T @ s

            # if Dm.shape[0] > 0:
            #     H = la.null_space(Dm)
            #     XH = X@H
            #     HtXty = H.T @ Xty_local
            #     HtDts = H.T @ Dts
            # else:
            #     print('break off', i, k, lambdas[k])
            #     beta = np.linalg.lstsq(X.mean(axis=0, keepdims=True).T @ X.mean(axis=0, keepdims=True), Xty_local, rcond=-1)
            #     break

            consec = np.split(all_indexes, np.nonzero(B)[0] + 1)
            nsplits = len(consec)

            XH = np.zeros((X.shape[0], nsplits))
            HtXty = np.zeros(nsplits)
            HtDts = np.zeros(nsplits)

            iterator = enumerate(consec)
            nextidx = 0

            for idx, indices in iterator:
                # print(k, idx ,indices, cutoff, consec)
                if indices[-1] == cutoff: # new cutoff point in the graph
                    cmean2(X[:, indices], XH[:, idx])
                    HtXty[idx] = cmean(Xty_local[indices])
                    HtDts[idx] = cmean(Dts[indices])

                    # process the next group as it also changed if we are not at the end
                    idx, indices = next(iterator, (None, None))
                    if idx is not None:
                        cmean2(X[:, indices], XH[:, idx])
                        HtXty[idx] = cmean(Xty_local[indices])
                        HtDts[idx] = cmean(Dts[indices])

                        # we now have to copy the next index from the old array to the new array
                        # go back one if we add a new colum to the new array
                        # skip one extra instead if we merge two columns in the new array
                        if nsplits > XH_prev.shape[1]:
                            nextidx = -1
                        else:
                            nextidx = 1

                else: # copy the old data if it didn't change
                    XH[:, idx] = XH_prev[:, idx+nextidx]
                    HtXty[idx] = HtXty_prev[idx+nextidx]
                    HtDts[idx] = HtDts_prev[idx+nextidx]

            # print(XH_prev)
            # print(XH)
            XH_prev = XH
            HtXty_prev = HtXty
            HtDts_prev = HtDts

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
            R = np.asarray(qr(XH))
            diagneg = np.diagonal(R) < 0
            # Q[:, diagneg] *= -1
            R[diagneg] *= -1

            A1 = cho_solve(R, HtXty)
            A2 = cho_solve(R, HtDts)

            # try:
            #     H = la.null_space(Dm)

            #     A1b = np.linalg.lstsq((X@H).T @ (X@H), H.T @ Xty_local, rcond=-1)[0]
            #     A2b = np.linalg.lstsq((X@H).T @ (X@H), H.T @ Dts, rcond=-1)[0]

            #     print('diff', np.abs(A1 - A1b).max(), np.abs(H.T @ Xty_local - HtXty).max(), np.abs((X@H).T @ (X@H) - R.T@R).max())
            #     print('diff', np.abs(A2 - A2b).max(), np.abs(H.T @ Dts - HtDts).max())
            # except:
            #     print('end reached')

            # step 3a
            if r == m: # Interior is empty
                hk = 0
                a = np.zeros(0)
                b = np.zeros(0)
            else:
                DDt = Dm @ Dm.T
                DDt_banded = diagonal_choform(DDt)
                XtXHA = X.T @ XH @ np.vstack((A1, A2)).T

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

            HA1 = np.zeros(p)
            HA2 = np.zeros(p)

            for idx, indices in enumerate(consec):
                HA1[indices] = A1[idx] / len(indices)
                HA2[indices] = A2[idx] / len(indices)

            # print(consec)
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

                ilk = np.argmax(tl)
                lk = tl[ilk]

                if lk < lmin:
                    lk = 0

            # print(a,b)
            # print(c,d)
            # print(HA1, HA2)
            # print(N, k, r, hk, lk, XH.shape, H.shape)
            # raise ValueError

            # print('lambdas', hk, lk, B.sum(), i, k)
            # update matrices and stuff for next step
            if hk > lk: # variable enters the boundary
                coord = np.nonzero(~B)[0][ik]
                updates = True, np.sign(a[ik])
                lambdak = hk
                # graph.delete_edges([(coord, coord + 1)])
                cutoff = coord
                r += 1

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
                updates = False, 0
                lambdak = lk
                cutoff = np.nonzero(B)[0][ilk -1]
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
                updates = False, 0

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
            B[coord] = updates[0]
            S[coord] = updates[1]
            betas[k] = HA1 - lambdak * HA2 # eq. 38
            lambdas[k] = lambdak
            dfs[k] = nsplits

            # print('B', B)

        all_Ks[i] = k

    return all_mus, all_lambdas, all_betas, all_dfs, all_Ks

def run(*args):
    inner_path(*args)

def this_function_causes_line_profiler_to_print_all_the_way_down_here():
    pass