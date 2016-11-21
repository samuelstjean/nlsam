import numpy as np
from _glmnet import elnet, solns #,modval, uncomp, , fishnet
# import _glmnet
from scipy.sparse import csr_matrix
from scipy.optimize import nnls
from scipy.linalg import lstsq

# from glmnet import glmnet

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def elastic_net_path(X, y, rho, ols=False, nlam=100, **kwargs):
    """return full path for ElasticNet"""

    # n_lambdas, intercepts, coefs, indices, nin, _, lambdas, _, jerr \
    #     = elastic_net(X, y, rho, **kwargs)
    # from time import time
    # t=time()
    lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr = elastic_net(X, y, rho, **kwargs)

    nobs, nx = X.shape

    a0 = a0[:lmu]
    ca = ca[:nx, :lmu]
    ia = ia[:nx]
    nin = nin[:lmu]
    rsq = rsq[:lmu]
    alm = alm[:lmu]

    ninmax = max(nin)

    if ninmax == 0:
        return np.zeros((nobs, lmu), dtype=np.float64), np.zeros([nx, lmu], dtype=np.float64)

    ca = ca[:ninmax]
    df = np.sum(ca != 0, axis=0)
    ja = ia[:ninmax] - 1    # ia is 1-indexed in fortran
    # ja = ia - 1    # ia is 1-indexed in fortran
    oja = np.argsort(ja)
    ja1 = ja[oja]
    beta = np.zeros([nx, lmu], dtype=np.float64)
    # print(beta.shape, ca.shape, ja1.shape, ca.shape, ia.shape, ja.shape, nx, lmu)
    # 1/0
    # print(ja1)
    # print(oja)
    # 1/0
    beta[ja1] = ca[oja]

        # predict = np.dot(D, fit['beta']) + fit['a0']
        # # print(fit['a0'], 'a0glmnet')
        # alphas[:, :fit['beta'].shape[1]] = fit['beta']
        # alphas[:, fit['beta'].shape[1]:] = 0.

        # Xhat[:, :predict.shape[1]] = predict
        # Xhat[:, predict.shape[1]:] = 0.



    predict = np.zeros((X.shape[0], nlam), dtype=np.float64)
    # predict[:, :lmu] = np.dot(X, beta[:, :lmu]) + a0
    predict[:, :lmu] = np.dot(X, beta) + a0
    # print(X.shape, predict.shape, beta.shape, a0.shape)
    out = np.zeros([nx, nlam], dtype=np.float64)
    out[:, :lmu] = beta
    return predict, out

    # print('time2 was', time()-t)
    # lmu=-1
    # print(lmu, 'lmu', nin)#, a0.sum(), ca, ia, nin, rsq, alm, nlp, jerr)
    # print(np.sum(X),np.sum(y),'sum1')
    # clip the stuff
    # lmu = n_lambdas
    # nvars = X.shape[1]
    # nx = X.shape[1] + 1 # nx was set before
    # a0 = a0[:lmu]
    # ca = ca[:nx, :lmu]
    # ia = ia[:nx]
    # nin = nin[:lmu]
    # rsq = rsq[:lmu]
    # alm = alm[:lmu]

    # if len(alm) > 2:
    #     lambda_0 = np.exp(2 * np.log(alm[1]) - np.log(alm[2]))
    #     alm[0] = lambda_0
    # # print(a0, 'a0sam')
    # ninmax = max(nin)

    # beta = solns(X.shape[1], ca, ia, nin)
    # beta = np.zeros([n_lambdas, X.shape[1]], dtype=np.float64)
    # print(beta.shape)
    # beta = solns(intercepts, indices, nin, beta)
    # # if ninmax == 0:
    # #     return np.zeros((X.shape[0], lmu), dtype=np.float64), np.zeros([nvars, lmu], dtype=np.float64)

    # ca = ca[:ninmax, :]
    # # df = np.sum(ca != 0, axis=0)
    # ja = ia[:ninmax] - 1    # ia is 1-indexed in fortran
    # oja = np.argsort(ja)
    # ja1 = ja[oja]
    # # print(ia, ja, oja, ja1)
    # beta = np.zeros([nvars, lmu], dtype=np.float64)
    # beta[ja1, :] = ca[oja, :]
    # # print(np.sum(beta), 'beta', oja, ja1, ca, nx, lmu)
    # predict = np.zeros((X.shape[0], beta.shape[1]), dtype=np.float64)
    # predict[:] = np.dot(X, beta) + intercepts
    # return predict, beta

    # indices -= 1
    # print(indices, len(indices), coefs.shape[1], nin)
    # print(nin)
    nlam = coefs.shape[1]
    reordered_coefs = np.zeros((X.shape[1], nlam), dtype=np.float64)
    # reordered_coefs2 = np.zeros((X.shape[1], nlam), dtype=np.float64)
    # predict = np.zeros((X.shape[0], nlam), dtype=np.float32)
    # print(coefs.shape, nin, coefs.shape, indices.shape)
    # reordered_coefs = coefs[:nin]
    # ind = indices[:nin] - 1
    for i in range(nlam):
        nval = nin[i]
        # Ordering from fortran starts at 1, so fix it to 0 for python
        ind = indices[:nval] - 1
        reordered_coefs[ind, i] = coefs[:nval, i]
        # reordered_coefs = coefs[:nval, i:i+1]
    # X = X[:, ind]#.squeeze()
    # ind = indices[:nval] - 1
    # # reordered_coefs2[indices[:nin]] = coefs[:nin]
    # # print(np.sum(np.abs(reordered_coefs - reordered_coefs2)))
    # # print(reordered_coefs, reordered_coefs2)
    # # ols = True
    # if ols:
    #     predict = np.zeros((X.shape[0], nlam))

    #     # print(reordered_coefs.shape, X.shape, y.shape)

    #     for i in range(nlam):
    #         idx = reordered_coefs[:, i] != 0
    #         if np.sum(idx) > 0:
    #             sol, _ = nnls(X[:, idx], y.ravel())
    #             # sol = lstsq(X[:, idx], y.ravel())[0]
    #             predict[:, i] = np.dot(X[:, idx], sol) + intercepts[i]
    # else:
    #     predict = np.dot(X, reordered_coefs) + intercepts
    #     # idx = reordered_coefs != 0
    #     # p2 = np.dot(X[:, idx], reordered_coefs[idx]) + intercepts
    #     # print(np.sum(np.abs(predict-p2)))
    # # print(np.sum(reordered_coefs != 0, axis=1), lambdas, coefs.shape[1], n_lambdas)
    # print(X.shape, reordered_coefs.shape, nlam, intercepts.shape)
    # 1/0
    # predict = np.dot(X, reordered_coefs) + intercepts
    # predict = np.zeros((X.shape[0], n_lambdas))
    predict = np.dot(X, reordered_coefs) + intercepts
    # predict = np.dot(X[:, ind], coefs) + intercepts
    # print(predict.shape, reordered_coefs.shape, 'predict')
    # print(np.sum(predict, axis=0), np.sum(predict, axis=0).shape)
    # print(predict[0])
    return predict, reordered_coefs


def select_best_path(X, y, beta, mu, variance=None, criterion='aic'):
    '''See https://arxiv.org/pdf/0712.0881.pdf p. 9 eq. 2.15 and 2.16

    With regards to my notation :
    X is D, the regressor/dictionary matrix
    y is X, the measured signal we wish to reconstruct
    beta is alpha, the coefficients
    mu is the denoised reconstruction X_hat = D * alpha
    '''

    # y.shape(y.shape[0])
    y = y.ravel()
    n = y.shape[0]
    p = X.shape[1]

    if criterion == 'aic':
        w = 2
    elif criterion == 'bic':
        w = np.log(n)
    elif criterion == 'ric':
        w = 2 * np.log(p)
    else:
        raise ValueError('Criterion {} is not supported!'.format(criterion))

    # mu = np.empty((X.shape[0],) + intercepts.shape, dtype=np.float32)
    # print(mu.shape, X.shape, y.shape, beta.shape, intercepts.shape, variance.shape)
    # print(X.shape, beta.shape, intercepts.shape)
    # print(np.dot(X, beta[0]).shape, intercepts[0].shape, mu[0].shape, (np.dot(X, beta[0]) + intercepts[0]).shape)
    # for i in range(beta.shape[0]):
    #     # print(i)
    #     mu[:, i] = np.dot(X, beta[i]) + intercepts[i]

    # print(X.shape, y.shape, beta.shape, mu.shape)
    # print(n,p)
    mse = np.mean((y[..., None] - mu)**2, axis=0) #, dtype=np.float32)
    df_mu = np.sum(beta != 0, axis=0) #, dtype=np.int32)
    # print(mse.shape, df_mu.shape, mu.shape, X.shape, y.shape)
    # 1/0
    # print(mu.shape, df_mu.shape, beta.shape, variance[...,None].shape, sse.shape, y.shape)
    # variance=None
    # variance=15**2
    # Use mse = SSE/n estimate for sample variance - we assume normally distributed
    # residuals though for the log-likelihood function...
    if variance is None:
        criterion = n * np.log(mse) + df_mu * w
        # s2 = sse / n
        # log_L = np.log(1 / np.sqrt(2 * np.pi * s2)) * n - sse / (2 * s2)
        # criterion = w * df_mu - 2 * log_L
    else:
        criterion = (mse / variance) + (w * df_mu / n)

    # We don't want empty models
    criterion[df_mu == 0] = 1e300
    best_idx = np.argmin(criterion, axis=0)

    # We can only estimate sigma squared after selecting the best model
    if n > df_mu[best_idx]:
        estimated_variance = np.sum((y - mu[:, best_idx])**2) / (n - df_mu[best_idx])
    else:
        estimated_variance = 0

    return mu[:, best_idx], beta[:, best_idx], best_idx


def lasso_path(X, y, ols=False, nlam=100, **kwargs):
    """return full path for Lasso"""

    # fit = glmnet(x=X.copy(), y=y.flatten(), family='gaussian', alpha=1., nlambda=nlambda)
    # from glmnet import ElasticNet
    # m = ElasticNet()
    # # print(X.shape, y.ravel().shape)
    # y = y.ravel()
    # m.fit(X, y)
    # return m.predict(X)[:, None]#, m.coef_path_
    return elastic_net_path(X, y, rho=1.0, ols=ols, nlam=nlam, **kwargs)


def elastic_net(X, y, rho, pos=True, thr=1e-4, weights=None, vp=None, copy=True,
                standardize=True, nlam=100, maxit=1e5, fit_intercept=True, **kwargs):
    """
    Raw-output wrapper for elastic net linear regression.
    X is D
    y is X
    rho for lasso/elastic net tradeoff
    """

    # print(thr)
    # print(pos, standardize, fit_intercept)
    if copy:
        # X/y is overwritten in the fortran function at every loop, so we must copy it each time
        X = np.array(X, copy=True, dtype=np.float64, order='F')
        y = np.array(y, copy=True, dtype=np.float64, order='F').ravel()

    # X = np.copy(X)
    # y = np.copy(y)
    # if y.ndim != 2:
    #     y.shape = (y.shape + (1,))
    # print(X.shape)
    # nx = X.shape[1]

    # # Flags determining overwrite behavior
    # overwrite_pred_ok = False
    # overwrite_targ_ok = False

    # thr = 1.0e-4   # Minimum change in largest coefficient
    # weights = None          # Relative weighting per observation case
    # vp = None               # Relative penalties per predictor (0 = no penalty)
    # isd = True              # Standardize input variables before proceeding?
    jd = np.zeros(1)        # X to exclude altogether from fitting
    ulam = None             # User-specified lambda values
    # flmin = 0.001  # Fraction of largest lambda at which to stop
    # nlam = 100    # The (maximum) number of lambdas to try.
    # maxit = 1000

    box_constraints = np.zeros((2, X.shape[1]), order='F')
    box_constraints[1] = 9.9e35 # this is a large number in fortran

    if not pos:
        box_constraints[0] = -9.9e35

    # for keyword in kwargs:
    #     # if keyword == 'overwrite_pred_ok':
    #     #     overwrite_pred_ok = kwargs[keyword]
    #     # elif keyword == 'overwrite_targ_ok':
    #     #     overwrite_targ_ok = kwargs[keyword]
    #     # if keyword == 'threshold':
    #     #     thr = kwargs[keyword]
    #     # elif keyword == 'weights':
    #     #     weights = np.asarray(kwargs[keyword]).copy()
    #     # elif keyword == 'penalties':
    #     #     vp = kwargs[keyword].copy()
    #     # elif keyword == 'standardize':
    #     #     isd = bool(kwargs[keyword])
    #     if keyword == 'exclude':
    #         # Add one since Fortran indices start at 1
    #         exclude = (np.asarray(kwargs[keyword]) + 1).tolist()
    #         jd = np.array([len(exclude)] + exclude)
    #     elif keyword == 'lambdas':
    #         if 'flmin' in kwargs:
    #             raise ValueError("Can't specify both lambdas & flmin keywords")
    #         ulam = np.asarray(kwargs[keyword])
    #         flmin = 2.  # Pass flmin > 1.0 indicating to use the user-supplied.
    #         nlam = len(ulam)
    #     elif keyword == 'flmin':
    #         flmin = kwargs[keyword]
    #         ulam = None
    #     elif keyword == 'nlam':
    #         if 'lambdas' in kwargs:
    #             raise ValueError("Can't specify both lambdas & nlam keywords")
    #         nlam = kwargs[keyword]
    #     else:
    #         raise ValueError("Unknown keyword argument '%s'" % keyword)

    # # If X is a Fortran contiguous array, it will be overwritten.
    # # Decide whether we want this. If it's not Fortran contiguous it will
    # # be copied into that form anyway so there's no chance of overwriting.
    # if np.isfortran(X):
    #     if not overwrite_pred_ok:
    #         # Might as well make it F-ordered to avoid ANOTHER copy.
    #         X = X.copy(order='F')
    # ulam = 0.001
    # nlam=None
    # y being a 1-dimensional array will usually be overwritten
    # with the standardized version unless we take steps to copy it.
    # if not overwrite_targ_ok:
    #     y = y.copy()

    # Uniform weighting if no weights are specified.
    if weights is None:
        weights = np.ones(X.shape[0], order='F')
    else:
        weights = np.array(weights, copy=True, order='F')

    # Uniform penalties if none were specified.
    if vp is None:
        vp = np.ones(X.shape[1], order='F')
    else:
        vp = vp.copy()

    # Call the Fortran wrapper.
    nx = X.shape[1] #+ 1
    # ulam = 0.1
    # nlam=None
    # Trick from official wrapper
    if X.shape[0] < X.shape[1]:
        flmin = 0.01
        # algo_flag = 2
    else:
        flmin = 1e-4
        # algo_flag = 1

    # ulam=None
    # print(X.shape, y.shape, weights.shape, jd,vp.shape, box_constraints.shape, nx, flmin, ulam, thr)
    # print(thr, len([algo_flag, rho, X, y, weights, jd, vp, box_constraints, nx, flmin, ulam, thr]))
    # print(len([rho, X, y, weights, jd, vp, box_constraints, nx, flmin, ulam, thr]))
    # print(([rho, X, y, weights, jd, vp, box_constraints, nx, flmin, ulam, thr]))
    # print(np.sum(X),np.sum(y),'sum1')
    # print(np.sum(X), np.sum(y),np.sum(weights),np.sum(jd),np.sum(vp),np.sum(box_constraints),np.sum(nx),np.sum(flmin),ulam,thr)
    # print(rho, X, y, weights, jd, vp, box_constraints, nx, flmin, ulam, thr,
              # nlam, standardize, maxit, fit_intercept)
    # print(X.shape, y.shape, weights.shape, jd.shape, vp.shape, box_constraints.shape, nx, flmin, ulam, thr)

    lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr = elnet(rho,
                                                      X,
                                                      y,
                                                      weights,
                                                      jd,
                                                      vp,
                                                      box_constraints,
                                                      nx,
                                                      flmin,
                                                      ulam,
                                                      thr,
                                                      nlam=nlam,
                                                      isd=standardize,
                                                      maxit=maxit,
                                                      intr=fit_intercept)
    # lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr = \
    #     elnet(algo_flag, rho, X, y, weights, jd, vp, box_constraints, nx, flmin, ulam, thr, nx-1,
    #           nlam, standardize, fit_intercept, maxit)

    return lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr


# This part stolen from
# https://github.com/ceholden/glmnet-python/blob/master/glmnet/utils.py


# def IC_path(X, y, coefs, intercepts, criterion='aic'):
#     """ Return AIC, BIC, or AICc for sets of estimated coefficients

#     Args:
#         X (np.ndarray): 2D (n_obs x n_features) design matrix
#         y (np.ndarray): 1D dependent variable
#         coefs (np.ndarray): 1 or 2D array of coefficients estimated from
#             GLMNET using one or more ``lambdas`` (n_coef x n_lambdas)
#         intercepts (np.ndarray): 1 or 2D array of intercepts from
#             GLMNET using one or more ``lambdas`` (n_lambdas)
#         criterion (str): AIC (Akaike Information Criterion), BIC (Bayesian
#             Information Criterion), or AICc (Akaike Information Criterion
#             corrected for finite sample sizes)

#     Returns:
#         np.ndarray: information criterion as 1D array (n_lambdas)

#     Note: AIC and BIC calculations taken from scikit-learn's LarsCV

#     """
#     # coefs = np.atleast_2d(coefs)

#     n_samples = y.shape[0]

#     criterion = criterion.lower()
#     if criterion == 'aic' or criterion == 'aicc':
#         K = 2
#     elif criterion == 'bic':
#         K = np.log(n_samples)
#     else:
#         raise ValueError('Criterion must be either AIC, BIC, or AICc')

#     residuals = y[:, np.newaxis] - (np.dot(X, coefs) + intercepts)
#     mse = np.mean(residuals**2, axis=0)
#     # df = np.zeros(coefs.shape[1], dtype=np.int16)
#     df = np.sum(coefs != 0, axis=-1)

#     # for k, coef in enumerate(coefs.T):
#     #     mask = np.abs(coef) > np.finfo(coef.dtype).eps
#     #     if not np.any(mask):
#     #         continue
#     #     df[k] = np.sum(mask)

#     with np.errstate(divide='ignore'):
#         criterion_ = n_samples * np.log(mse) + K * df
#         if criterion == 'aicc':
#             criterion_ += (2 * df * (df + 1)) / (n_samples - df - 1)

#     return criterion_
