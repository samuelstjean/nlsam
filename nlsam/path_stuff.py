import numpy as np

# from tqdm.autonotebook import tqdm
# from scipy.linalg import  null_space, cholesky_banded, cho_solve_banded
import scipy.linalg as la
# import scipy.sparse as ssp
# import qpsolvers

from nlsam.inner_path import inner_path

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


def path_stuff(X, y, penalty='fused', nsteps=None, eps=1e-8, lmin=1e-5, l2=1e-6, l1=0, dtype=np.float32, return_lambdas=False):

    if nsteps is None:
        nsteps = int(3 * max(X.shape))

    if nsteps <= 0 or eps < 0 or lmin < 0 or l2 < 0 or l1 < 0:
        error = 'Parameter error'
        raise ValueError(error)

    if y.ndim == 1:
        y = y[:, None]

    var = np.var(y, axis=0)
    nold = X.shape[0] # This is the original shape, which is modified later on if l2 > 0
    N = y.shape[1]
    m = X.shape[0]
    p = X.shape[1]
    yold = y
    Xold = X

    if penalty == 'fused':
        D = np.eye(p-1, p, k=1) - np.eye(p-1, p)
    elif penalty == 'sparse':
        if l1 <= 0:
            error = f'Penalty sparse requires l1 > 0, but it is set to l1 = {l1}'
            raise ValueError(error)

        D = np.eye(p-1, p, k=1) - np.eye(p-1, p)
        D = np.vstack((D, l1 * np.eye(p)))
        m = D.shape[0]
        p = D.shape[1]
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

    # n = X.shape[0]
    d = D.shape[0]
    # B = np.zeros(m, dtype=bool)
    # S = np.zeros(m, dtype=np.int8)
    # muk = np.zeros(m)

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
    DDt_diag_banded = np.vstack([np.full(d, -1), np.full(d, 2)])
    DDt_diag_banded[0, 0] = 0
    L_banded = la.cholesky_banded(DDt_diag_banded, lower=False, check_finite=False)

    # yproj = X @ Xpinv @ y
    # Dproj = D @ Xpinv

    # step 1
    # print('in step 1')
    H = np.ones([p, 1]) / p
    # XH = X@H
    XH = X.mean(axis=1, keepdims=True)
    # Q, R = np.linalg.qr(XH)
    Xty = X.T @ y
    HtXty = Xty.mean(axis=0, keepdims=True)
    # b = scipy.linalg.solve_triangular(R.T, H.T @ Xty, lower=True, check_finite=False)
    # mid = scipy.linalg.solve_triangular(R, b, check_finite=False)

    # rhs = np.linalg.lstsq(XH.T @ XH, H.T @ Xty, rcond=None)[0]
    rhs = np.linalg.lstsq(XH.T @ XH, HtXty, rcond=None)[0]
    v = Xty - X.T @ XH @ rhs
    u0 = la.cho_solve_banded((L_banded, False), D @ v, check_finite=False)

    # step 2
    # print('in step 2')
    t = np.abs(u0)
    i0 = np.argmax(t, axis=0).astype(np.int16)
    # h0 = t[i0, range(N)].astype(np.float32)
    h0 = np.max(t, axis=0).astype(np.float32)
    S0 = np.sign(u0[i0, range(N)]).astype(np.int16)

    # all_mus = [None] * N
    # all_lambdas = [None] * N
    # all_betas = [None] * N
    # all_dfs = [None] * N

    # for i in range(N):
    #     all_mus[i] = [None] * nsteps
    #     all_lambdas[i] = [None] * nsteps
    #     all_betas[i] = [None] * nsteps
    #     all_dfs[i] = [None] * nsteps

    #     all_betas[i][0] = H @ rhs[:, i]
    #     all_lambdas[i][0] = h0[i]
    #     all_mus[i][0] = u0[:, i]
    #     all_dfs[i][0] = 1

    # Q_all = Q.copy()
    # R_all = R.copy()
    # Xty_all = Xty.copy()
    # XtX = X.T @ X
    # Xty = np.zeros(p)
    # K = np.zeros(N, dtype=np.int16)

    all_l2_error = [None] * N
    all_Cps = [None] * N
    all_AIC = [None] * N
    all_AICc = [None] * N
    all_Cps_v2 = [None] * N
    all_obj = [None] * N
    # all_poly = [None] * N
    all_best_idx = np.zeros(N, dtype=np.int16)
    # all_best_poly = np.zeros(N, dtype=np.int16)
    all_best_sem = np.zeros(N, dtype=np.int16)

    all_mus, all_lambdas, all_betas, all_dfs, all_K = inner_path(X, y, D, Xty, i0, S0, h0, lmin, nsteps, eps)
    # print(all_mus.shape, all_lambdas.shape, all_betas.shape, all_dfs.shape, all_K.shape)
    # print(u0.shape, np.shape(H @ rhs))
    all_mus[:, 0] = u0.T
    # all_lambdas[:, 0] = h0[i]
    all_betas[:, 0] = rhs.T @ H.T
    all_dfs[:, 0] = 1
    dtype = np.float32
    all_freedom = np.zeros((N, nsteps))

    # degree = 10

    for i in range(N):
        k = all_K[i]
        # print(i, all_K[i], all_dfs[i, :k][-1], all_dfs[i, :k+1][-1], all_dfs[i].max())
        lambdas = np.asarray(all_lambdas[i][:k], dtype=dtype).T
        betas = np.asarray(all_betas[i][:k], dtype=dtype).T
        dfs = np.asarray(all_dfs[i][:k], dtype=dtype)
        l2_error = np.sum((yold[:, i:i+1] - Xold @ betas)**2, axis=0, dtype=dtype)
        obj = l2_error + lambdas * np.abs(D @ betas).sum(axis=0, dtype=dtype) + l2 * np.sum(betas**2, axis=0, dtype=dtype)
        Cps = l2_error - m * var[i] + 2 * var[i] * dfs
        Cps_v2 = (l2_error + 2*dfs*var[i]) / m
        AIC = 2*p + m * np.log(l2_error / dfs)
        # AICc = AIC + (2*dfs**2 + 2*dfs) / (m - dfs - 1)
        AICc = AIC + (2*dfs * (dfs + 1)) / (m - dfs - 1)
        # K = (dfs * p) + p * (p + 1) / 2
        # AICc = AIC + (2*K * (dfs + 1 + p)) / (m - dfs - 1 - p)
        # AICc = AIC + 2 * (K * (K + dfs)) / (m*p - K - dfs)
        # AICc = AIC + 2 * (dfs * (dfs + 1)) / (m - dfs - 1)
        AICc[~np.isfinite(AICc)] = np.nan
        best_idx = np.argmin(Cps)
        # print('best idx', best_idx, Cps.shape, obj.shape)

        # all_lambdas[i] = lambdas
        # all_betas[i] = betas
        # all_dfs[i] = dfs
        all_l2_error[i] = l2_error
        all_Cps[i] = Cps
        all_AIC[i] = AIC
        all_AICc[i] = AICc
        all_Cps_v2[i] = Cps_v2
        all_obj[i] = obj
        all_best_idx[i] = best_idx
        all_freedom[i, :len(dfs)] = dfs

        sem = np.nanstd(AICc) / np.sqrt(len(AICc))
        argmin = np.nanargmin(AICc)
        absmin = np.abs(np.nanmin(AICc) - (AICc - sem))
        best_sem = np.nanargmin(absmin[:argmin+1]) # only allow simpler models on the left

        # position on the right if needed
        if np.nanmin(AICc) < (AICc[best_sem] - sem):
            best_sem += 1
        all_best_sem[i] = best_sem
        # poly = np.polyfit(range(len(AICc)), AICc, degree)
        # all_poly[i] = np.polyval(poly, range(len(AICc)))
        # best_poly = np.argmin(all_poly[i])
        # all_best_poly[i] = best_poly
        # print('best poly idx', best_sem, AICc.shape, obj.shape, '\n')


    # Truncate output to max elements
    maxK = np.max(all_K)
    all_betas = all_betas[:, :maxK]
    all_lambdas = all_lambdas[:, :maxK]
    all_freedom = all_freedom[:, :maxK]
    # best_sols = all_betas[:, all_best_idx]
    best_sols = all_betas[range(N), all_best_sem]
    best_recon = best_sols @ Xold.T
    best_lambdas = all_lambdas[range(N), all_best_sem]
    best_freedom = all_freedom[range(N), all_best_sem]
    # best_polysols = all_betas[:, best_poly]
    # best_polyrecon = Xold @ best_polysols

    # print('chosen lambdas', )
    # print('chosen dfs', all_freedom[range(N), all_best_sem])
    if return_lambdas:
        return all_betas, all_lambdas, all_Cps, all_best_idx, all_obj, best_sols, best_recon
    return best_sols, best_recon, best_lambdas, best_freedom



# def inner_path(X, y, D, Xty, i0, u0, h0, lmin, nsteps, eps):
#     N = y.shape[1]
#     p = X.shape[1]
#     n = X.shape[0]
#     m = D.shape[0]

#     XtX = X.T @ X

#     for i in range(N):
#         B = np.zeros(m, dtype=bool)
#         S = np.zeros(m)
#         Xty_local = np.zeros(p)
#         H = np.ones((p, 1))
#         newH = np.ones((p, 1))
#         muk = np.zeros(m)

#         k = 0
#         i0_local = i0[i]
#         newH[:i0_local + 1] = 0
#         Xty_local[:] = Xty[:, i]

#         B[i0_local] = True
#         S[i0_local] = np.sign(u0[i0_local, i])

#         H = np.hstack((H, newH))
#         XH = X @ H
#         XtXH = XtX @ H

#         # stuff to return
#         mus = np.zeros((nsteps, m))
#         lambdas = np.zeros(nsteps)
#         betas = np.zeros((nsteps, p))
#         dfs = np.zeros(nsteps, dtype=np.int16)
#         K = np.zeros(N, dtype=np.int16)

#         lambdas[0] = h0[i]

#         while np.abs(lambdas[k]) > lmin and k < (nsteps - 1):
#             # print(k, lambdas[k])

#             Db = D[B]
#             Dm = D[~B]
#             s = S[B]
#             Dts = Db.T @ s
#             DDt = Dm @ Dm.T
#             DDt_banded = diagonal_choform(DDt)
#             L_banded = cholesky_banded(DDt_banded)

#             # Reset the QR to prevent accumulating errors during long runs
#             # if k % 1 == 0:
#             Q, R = np.linalg.qr(XH)
#             # print(R.shape)
#             # print(R)
#             # rhs = H.T @ np.vstack((Xty, Dts)).T
#             # rhs = np.zeros((H.shape[1], 2))
#             # rhs[:, 0] = H.T @ Xty
#             # rhs[:, 1] = H.T @ Dts
#             A1 = backsolve(R, forwardsolve(R, H.T @ Xty_local, lower=False, transpose=True))
#             A2 = backsolve(R, forwardsolve(R, H.T @ Dts, lower=False, transpose=True))
#             # A1 = A[:, 0]
#             # A2 = A[:, 1]

#             # step 3a
#             if Dm.shape[0] == 0: # Interior is empty
#                 hk = 0
#                 a = np.zeros(0)
#                 b = np.zeros(0)
#             else:
#                 v = Xty_local - XtXH @ A1
#                 w = Dts - XtXH @ A2

#                 # rhs = Dm @ np.vstack((v, w)).T
#                 # rhs = np.zeros((Dm.shape[0], 2))
#                 # rhs[:, 0] = Dm @ v
#                 # rhs[:, 1] = Dm @ w
#                 # A = cho_solve_banded((L_banded, False), rhs)
#                 a = cho_solve_banded((L_banded, False), Dm @ v)
#                 b = cho_solve_banded((L_banded, False), Dm @ w)
#                 # a = A[:, 0]
#                 # b = A[:, 1]

#                 # step 3b
#                 # hitting time
#                 t = a / (b + np.sign(a) + eps) # prevent divide by 0
#                 # numerical issues fix
#                 t[t > lambdas[k] + eps] = 0

#                 ik = np.argmax(t)
#                 hk = t[ik]

#             # leaving time
#             HA1 = H @ A1
#             HA2 = H @ A2

#             if Db.shape[0] == 0 or H.shape[0] == H.shape[1]: # Boundary is empty
#                 lk = 0
#             else:
#                 c =  s * (Db @ HA1)
#                 d =  s * (Db @ HA2)

#                 tl = np.where((c < 0) & (d < 0), c/(d + eps), 0)
#                 # numerical issues fix
#                 tl[tl > lambdas[k] + eps] = 0

#                 ilk = np.argmax(tl)
#                 lk = tl[ilk]

#             # update matrices and stuff for next step
#             if hk > lk: # variable enters the boundary
#                 coord = np.nonzero(~B)[0][ik]
#                 updates = True, np.sign(a[ik])
#                 lambdak = hk
#                 newH = np.ones(p)
#                 newH[:coord + 1] = 0
#                 newpos = np.searchsorted(np.nonzero(B)[0], coord)
#                 # Q, R = la.qr_insert(Q, R, X @ newH, newpos + 1, which='col', overwrite_qru=True, check_finite=False)
#                 # H = np.insert(H, newpos + 1, newH, axis=1)
#                 # H = np.ones((H.shape[0], H.shape[1] + 1))
#                 XH = np.insert(XH, newpos + 1, X @ newH, axis=1)
#                 XH = X @ H
#                 XtXH = np.insert(XtXH, newpos + 1, XtX @ newH, axis=1)
#                 XtXH = XtX @ H
#                 XtXH = X.T @ XH
#                 # print(H.shape, R.shape, newpos, newH.shape, XH.shape)
#                 # print(f'add {ik, coord}, B={B.sum()}')
#             elif lk > hk: # variable leaves the boundary
#                 coord = np.nonzero(B)[0][ilk]
#                 updates = False, 0
#                 lambdak = lk
#                 # Q, R = la.qr_delete(Q, R, ilk + 1, which='col', overwrite_qr=True, check_finite=False)
#                 # H = np.ones((H.shape[0], H.shape[1] - 1))
#                 XH = X@H
#                 XTXH = XtX@H
#                 H = np.delete(H, ilk + 1, axis=1)
#                 XH = np.delete(XH, ilk + 1, axis=1)
#                 XtXH = np.delete(XtXH, ilk + 1, axis=1)
#                 # print(f'remove {ilk, coord}, B={B.sum()}')
#             elif (lk == 0) and (hk == 0): # end of the path, so everything stays as is
#                 lambdak = 0
#                 # print('end of the path reached')

#             k += 1
#             muk[~B] = a - lambdak * b
#             muk[B] = lambdak * s
#             # print(a.shape, b.shape, lambdak, s.shape, muk.shape)
#             mus[k] = muk

#             # update boundaries
#             B[coord], S[coord] = updates
#             betas[k] = HA1 - lambdak * HA2 # eq. 38
#             lambdas[k] = lambdak
#             dfs[k] = H.shape[1] - 1

#         K[i] = k
