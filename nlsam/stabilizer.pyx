#cython: wraparound=False, cdivision=True, boundscheck=False

cimport cython

from itertools import repeat
from libc.math cimport sqrt, exp, fabs, M_PI, isnan
from multiprocessing import Pool, cpu_count

import numpy as np
cimport numpy as np

from dipy.core.ndindex import ndindex
from scipy.special import erfinv
# from scipy.special._ufuncs cimport ndtri
# cdef extern from "scipy.special.cephes.h" nogil:
#     double ndtri (double y)

from nibabel.optpkg import optional_package
cython_gsl, have_cython_gsl, _ = optional_package("cython_gsl")

if not have_cython_gsl:
    raise ValueError('Cannot find gsl package (required for hyp1f1), \n'
        'try pip install cythongsl and sudo apt-get install libgsl0-dev libgsl0ldbl')

from cython_gsl cimport gsl_sf_hyperg_1F1

@cython.wraparound(True)
def stabilisation(data, m_hat, mask, sigma, N, n_cores=None):

      # Check all dims are ok
      if (data.shape != sigma.shape):
          raise ValueError('data shape {} is not compatible with sigma shape {}'.format(data.shape, sigma.shape))

      if (data.shape[:-1] != mask.shape):
          raise ValueError('data shape {} is not compatible with mask shape {}'.format(data.shape, mask.shape))

      if (data.shape != m_hat.shape):
          raise ValueError('data shape {} is not compatible with m_hat shape {}'.format(data.shape, m_hat.shape))

      pool = Pool(processes=n_cores)
      arglist = [(data[..., idx, :],
                  m_hat[..., idx, :],
                  mask[..., idx],
                  sigma[..., idx, :],
                  N_vox)
                 for idx, N_vox in zip(range(data.shape[-2]), repeat(N))]

      data_out = pool.map(_multiprocess_stabilisation, arglist)
      pool.close()
      pool.join()

      data_stabilized = np.empty(data.shape, dtype=np.float32)

      for idx in range(len(data_out)):
          data_stabilized[..., idx, :] = data_out[idx]

      return data_stabilized


def _multiprocess_stabilisation(arglist):
    """Helper function for multiprocessing the stabilization part."""

    data = arglist[0].astype(np.float64)
    m_hat = arglist[1].astype(np.float64)
    mask = arglist[2].astype(np.bool)
    sigma = arglist[3].astype(np.float64)
    N = arglist[4]

    out = np.zeros(data.shape, dtype=np.float32)

    for idx in ndindex(data.shape):
        if sigma[idx] > 0 and mask[idx]:
            eta = fixed_point_finder(m_hat[idx], sigma[idx], N)
            out[idx] = chi_to_gauss(data[idx], eta, sigma[idx], N)

    return out


cdef double hyp1f1(double a, int b, double x) nogil:
    """Wrapper for 1F1 hypergeometric series function
    http://en.wikipedia.org/wiki/Confluent_hypergeometric_function"""
    return gsl_sf_hyperg_1F1(a, b, x)


cdef double _inv_cdf_gauss(double y, double eta, double sigma) nogil:
    """Helper function for chi_to_gauss. Returns the gaussian distributed value
    associated to a given probability. See p. 4 of [1] eq. 13.

    Input
    -------
    y : double
        Probability of observing the desired value in the normal
        distribution N(eta, sigma**2)
    eta :
        Mean of the normal distribution N(eta, sigma**2)
    sigma : double
        Standard deviation of the normal distribution N(eta, sigma**2)

    return
    --------
        Value associated to probability y given a normal distribution N(eta, sigma**2)
    """
    with gil:
        return eta + sigma * sqrt(2) * erfinv(2*y - 1)


# cdef double erfinv(double y) nogil:
#     '''Same as scipy.special.erfinv, but with nogil'''
#     return ndtri((y + 1) / 2.0) / sqrt(2)


cdef double chi_to_gauss(double m, double eta, double sigma, int N,
                          double alpha=0.0001) nogil:
    """Maps the noisy signal intensity from a Rician/Non central chi distribution
    to its gaussian counterpart. See p. 4 of [1] eq. 12.

    Input
    --------
    m : double
        The noisy, Rician/Non central chi distributed value
    eta : double
        The underlying signal intensity estimated value
    sigma : double
        The gaussian noise estimated standard deviation
    N : int
        Number of coils of the acquision (N=1 for Rician noise)
    alpha : double
        Confidence interval for the cumulative distribution function.
        Clips the cdf to alpha/2 <= cdf <= 1-alpha/2

    Return
    --------
        double : The noisy gaussian distributed signal intensity

    Reference
    -----------
    [1]. Koay CG, Ozarslan E and Basser PJ.
    A signal transformational framework for breaking the noise floor
    and its applications in MRI.
    Journal of Magnetic Resonance 2009; 197: 108-119.
    """
    cdef double cdf

    with nogil:
        cdf = 1. - _marcumq_cython(eta/sigma, m/sigma, N)

        # clip cdf between alpha/2 and 1-alpha/2
        if cdf < alpha/2:
            cdf = alpha/2
        elif cdf > 1 - alpha/2:
            cdf = 1 - alpha/2

        return _inv_cdf_gauss(cdf, eta, sigma)


cdef double multifactorial(int N, int k=1) nogil:
    """Returns the multifactorial of order k of N.
    https://en.wikipedia.org/wiki/Factorial#Multifactorials

    N : int
        Number to compute the factorial of
    k : int
        Order of the factorial, default k=1

    return : double
        Return type is double, because multifactorial(21) > 2**64.
        Same as scipy.special.factorialk, but in a nogil clause.
    """
    if N == 0:
        return 1.

    elif N < (k + 1):
        return N

    return N * multifactorial(N - k, k)


cdef double _marcumq_cython(double a, double b, int M, double eps=1e-8,
                            int max_iter=10000) nogil:
    """Computes the generalized Marcum Q function of order M.
    http://en.wikipedia.org/wiki/Marcum_Q-function

    a : double, eta/sigma
    b : double, m/sigma
    M : int, order of the function (Number of coils, N=1 for Rician noise)

    return : double
        Value of the function, always between 0 and 1 since it's a pdf.
    """
    cdef:
        double a2 = 0.5 * a**2
        double b2 = 0.5 * b**2
        double d = exp(-a2)
        double h = exp(-a2)
        double f = (b2**M) * exp(-b2) / multifactorial(M)
        double f_err = exp(-b2)
        double errbnd = 1. - f_err
        double  S = f * h
        double temp = 0.
        int k = 1
        int j = errbnd > 4*eps

    if fabs(a) < eps:

        for k in range(M):
            temp += b**(2*k) / (2**k * multifactorial(k))

        return exp(-b**2/2) * temp

    elif fabs(b) < eps:
        return 1.

    while j or k <= M:

        d *= a2 / k
        h += d
        f *= b2 / (k + M)
        S += f * h

        f_err *= b2 / k
        errbnd -= f_err

        j = errbnd > 4*eps
        k += 1

        if k > max_iter:
            break

    return 1. - S


cdef double fixed_point_finder(double m_hat, double sigma, int N,
                                int max_iter=100, double eps=1e-4) nogil:
    """Fixed point formula for finding eta. Table 1 p. 11 of [1]

    Input
    --------
    m_hat : double
        Initial value for the estimation of eta
    sigma : double
        Gaussian standard deviation of the noise
    N : int
        Number of coils of the acquision (N=1 for Rician noise)
    max_iter : int, default=100
        Maximum number of iterations before breaking from the loop
    eps : double, default = 1e-4
        Criterion for reaching convergence between two subsequent estimates

    Return
    -------
    t1 : double
        Estimation of the underlying signal value
    """
    cdef:
        double delta, m, t0, t1
        int cond = True
        int n_iter = 0

    with nogil:
        # If m_hat is below the noise floor, return 0 instead of negatives
        # as per Bai 2014
        if m_hat < sqrt(0.5 * M_PI) * sigma:
            return 0

        delta = _beta(N) * sigma - m_hat

        if fabs(delta) < 1e-15:
            return 0

        m = m_hat

        t0 = m
        t1 = _fixed_point_k(t0, m, sigma, N)

        while cond:

            t0 = t1
            t1 = _fixed_point_k(t0, m, sigma, N)
            n_iter += 1
            cond = fabs(t1 - t0) > eps

            if n_iter > max_iter:
                break

        if t1 < 0 or isnan(t1): # Should not happen unless numerically unstable
            t1 = 0

        return t1


cdef double _beta(int N) nogil:
    """Helper function for _xi, see p. 3 [1] just after eq. 8."""
    cdef:
        double factorialN_1 = multifactorial(N - 1)
        double factorial2N_1 = multifactorial(2*N - 1, 2)

    return sqrt(0.5 * M_PI) * (factorial2N_1 / (2**(N-1) * factorialN_1))


cdef double _fixed_point_g(double eta, double m, double sigma, int N) nogil:
    """Helper function for _fixed_point_k, see p. 3 [1] eq. 11."""
    return sqrt(m**2 + (_xi(eta, sigma, N) - 2*N) * sigma**2)


cdef double _fixed_point_k(double eta, double m, double sigma, int N) nogil:
    """Helper function for fixed_point_finder, see p. 11 [1] eq. D2."""
    cdef:
        double fpg, num, denom
        double eta2sigma = -eta**2/(2*sigma**2)

    fpg = _fixed_point_g(eta, m, sigma, N)
    num = fpg * (fpg - eta)

    denom = eta * (1 - ((_beta(N)**2)/(2*N)) *
                   hyp1f1(-0.5, N, eta2sigma) *
                   hyp1f1(0.5, N+1, eta2sigma)) - fpg

    return eta - num / denom


def corrected_sigma(eta, sigma, mask, N, n_cores=None):
    """Compute the local corrected standard deviation for the adaptive nonlocal
    means according to the correction factor xi.

    Input
    --------
    eta : double
        Signal intensity
    sigma : double
        Noise magnitude standard deviation
    mask : ndarray
        Compute only the corrected sigma value inside the mask.
    N : int
        Number of coils of the acquisition (N=1 for Rician noise)
    n_cores : int
        Number of cpu cores to use for parallel computations, default : all of them

    Return
    --------
    sigma, ndarray
        Corrected sigma value, where sigma_gaussian = sigma / sqrt(xi)
    """
    pool = Pool(processes=n_cores)
    arglist = [(eta_vox, sigma_vox, mask_vox, N_vox)
               for eta_vox, sigma_vox, mask_vox, N_vox
               in zip(eta, sigma, mask, repeat(N))]
    sigma = pool.map(_corrected_sigma_parallel, arglist)
    pool.close()
    pool.join()

    return np.asarray(sigma).reshape(eta.shape).astype(np.float32)


def _corrected_sigma_parallel(arglist):
    """Helper function for corrected_sigma to multiprocess the correction
    factor xi."""

    eta, sigma, mask, N = arglist
    out = np.zeros(eta.shape, dtype=np.float32)

    for idx in ndindex(out.shape):
        if sigma[idx] > 0 and mask[idx]:
            out[idx] = _corrected_sigma(eta[idx], sigma[idx], N)

    return out


cdef double _corrected_sigma(double eta, double sigma, int N)  nogil:
    """Compute the local corrected standard deviation for the adaptive nonlocal
    means according to the correction factor xi.

    Input
    -------
    eta : double
        Signal intensity
    sigma : double
        Noise magnitude standard deviation
    N : int
        Number of coils of the acquisition (N=1 for Rician noise)
    mask : ndarray
        Compute only the corrected sigma value inside the mask.

    Return
    -------
    ndarray
        Corrected sigma value, where sigma_gaussian = sigma / sqrt(xi)
    """
    return sigma / sqrt(_xi(eta, sigma, N))


cdef double _xi(double eta, double sigma, int N) nogil:
    """Standard deviation scaling factor formula, see p. 3 of [1], eq. 10.

    Input
    -------
    eta : double
        Signal intensity
    sigma : double
        Noise magnitude standard deviation
    N : int
        Number of coils of the acquisition (N=1 for Rician noise)

    Return
    --------
    double
        The correction factor xi, where sigma_gaussian**2 = sigma**2 / xi
    """

    if fabs(sigma) < 1e-15:
        return 1.

    h1f1 = hyp1f1(-0.5, N, -eta**2/(2*sigma**2))
    return 2*N + eta**2/sigma**2 -(_beta(N) * h1f1)**2


# Test for cython functions
def _test_marcumq_cython(a, b, M, eps=1e-7, max_iter=10000):
    return _marcumq_cython(a, b, M, eps, max_iter)


def _test_beta(N):
    return _beta(N)


def _test_fixed_point_g(eta, m, sigma, N):
    return _fixed_point_g(eta, m, sigma, N)


def _test_fixed_point_k(eta, m, sigma, N):
    return _fixed_point_k(eta, m, sigma, N)


def _test_xi(eta, sigma, N):
    return _xi(eta, sigma, N)


def _test_multifactorial(N, k=1):
    return multifactorial(N, k)


def _test_inv_cdf_gauss(y, eta, sigma):
    return _inv_cdf_gauss(y, eta, sigma)


def _test_chi_to_gauss(m, eta, sigma, N):
    return chi_to_gauss(m, eta, sigma, N)


def _test_erfinv(y):
    return erfinv(y)


def _test_fixed_point_finder(m_hat, sigma, N):
    return fixed_point_finder(m_hat, sigma, N)
