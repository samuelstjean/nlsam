#cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
cimport cython

from libc.math cimport sqrt, exp, fabs, M_PI

from nlsam.multiprocess import multiprocesser
from scipy.special.cython_special cimport ndtri, ive, gamma, chndtr, gammainc, chdtr

# this is our special R wrapped marcum q function
# from numba import vectorize, jit
from rvlib._rmath_ffi.lib import pnchisq as pnchisq_R
# @jit(nopython=True, nogil=True)
# def pnchisq(q, df, ncp):
#     return pnchisq_R(q, df, ncp, False, False)


# libc.math isnan does not work on windows, it is called _isnan, so we use this one instead
cdef extern from "numpy/npy_math.h" nogil:
    bint npy_isnan(double x)

cdef extern from "hyp_1f1.h" nogil:
    double gsl_sf_hyperg_1F1(double a, double b, double x)


# These def are used to call the code from the external portions
def chi_to_gauss(m, eta, sigma, N):
    return _chi_to_gauss(m, eta, sigma, N)


def fixed_point_finder(m_hat, sigma, N, clip_eta=True):
    return _fixed_point_finder(m_hat, sigma, N, clip_eta)


def root_finder(r, N, max_iter=500, eps=1e-6):
    return _root_finder(r, N, max_iter, eps)


cdef double hyp1f1(double a, double b, double x) nogil:
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

    Note
    -----
    The original paper mentions to use erfinv, but we have directly that
    ndtri(y) = sqrt(2) * erfinv(2*y - 1)
    """
    return eta + sigma * ndtri(y)


cdef double _chi_to_gauss(double m, double eta, double sigma, double N,
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
        Number of coils of the acquisition (N=1 for Rician noise)
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


# cdef double multifactorial(double N, int k=1) nogil:
#     """Returns the multifactorial of order k of N.
#     https://en.wikipedia.org/wiki/Factorial#Multifactorials

#     N : int
#         Number to compute the factorial of
#     k : int
#         Order of the factorial, default k=1

#     return : double
#         Return type is double, because multifactorial(21) > 2**64.
#         Same as scipy.special.factorialk, but in a nogil clause.
#     """
#     if N == 0:
#         return 1.

#     elif N < (k + 1):
#         return N

#     return N * multifactorial(N - k, k)


# cdef double factorial(double x) nogil:
#     return gamma(x + 1)


cdef double _marcumq_cython(double a, double b, double M, double eps=1e-8) nogil:
    """Computes the generalized Marcum Q function of order M.
    http://en.wikipedia.org/wiki/Marcum_Q-function

    a : double, eta/sigma
    b : double, m/sigma
    M : double, order of the function (Number of coils, N=1 for Rician noise)

    return : double
        Value of the function, always between 0 and 1 since it's a pdf.

    Notes
    ------
    We actually use chndtr which is the cdf of a chi square variable with a few
    change of arguments. The relation is

    nchi2_pdf(x, k, lbda) = 1 - Marcum(sqrt(lbda), sqrt(x), k/2)

    and therefore

    Marcum(a, b, M) = 1 - nchi2_pdf(b**2, a**2, 2*M)

    or in our notation

    nchi_pdf = 1 - Marcum(a**2, b**2, 2*M)
    """
    cdef:
        double k = 2 * M
        double lbda = a**2
        double x = b**2
        double out

    # if k < 80:
    #     out = 1. - chndtr(x, k, lbda)
    # else:
    #     with gil:
    #         out = 1. - pnchisq(x, k, lbda)

    if fabs(b) < eps:
        return 1.

    if fabs(M) < eps:
        k = eps

    # a, b and M can not be negative, so we hardcode their probability to 0
    if (a < 0) or (b < 0) or (M < 0):
        return 1.

    # if fabs(a) < eps:
    #     for i in range(int(M)):
    #         temp += b**(2*i) / (2**i * gamma(i+1.))

    #     return exp(-b**2/2) * temp

    with gil:
        out = pnchisq_R(x, k, lbda, False, False)

    return out
    # return 1. - chndtr(x, k, lbda)


# cdef double _marcumq_cython(double a, double b, double M) nogil:
#     cdef:
#         double k = 2 * M
#         double lbda = a**2
#         double x = b**2

#         double h = 1 - 2/3 * (k+lbda) * (k+3*lbda) / (k + 2*lbda)**2
#         double p = (k+2*lbda) / (k + lbda)**2
#         double m = (h - 1) * (1 - 3*h)

#         double num = (x / (k + lbda))**h - (1 + h*p * (h - 1 - 0.5 * (2 - h) * m * p))
#         double denom = h * sqrt(2 * p) * (1 + 0.5 * m * p)

#     return 1. - ndtri(num / denom)


# Stolen from octave signal marcumq
# cdef double _marcumq_cython(double a, double b, double M, double eps=1e-7) nogil:
#     """Computes the generalized Marcum Q function of order M.
#     http://en.wikipedia.org/wiki/Marcum_Q-function

#     a : double, eta/sigma
#     b : double, m/sigma
#     M : int, order of the function (Number of coils, N=1 for Rician noise)

#     return : double
#         Value of the function, always between 0 and 1 since it's a pdf.
#     """
#     cdef:
#         bint cond = True
#         int s, c, k
#         double S, x, d, t
#         double temp = 0.
#         double z = a * b
#         int M_int = int(round(M))

#     if fabs(b) < eps:
#         return 1.

#     if fabs(a) < eps:
#         for k in range(M_int):
#             temp += b**(2*k) / (2**k * gamma(k+1.))

#         return exp(-b**2/2) * temp

#     if a < b:
#         s = 1
#         c = 0
#         x = a / b
#         d = x
#         S = ive(0, z)

#         for k in range(1, M_int):
#             S += (d + 1/d) * ive(k, z)
#             d *= x

#         k = M_int
#     else:
#         s = -1
#         c = 1
#         x = b / a
#         k = M_int
#         d = x**M_int
#         S = 0

#     while cond:
#         t = d * ive(k, z)
#         S += t
#         d *= x
#         k += 1

#         cond = fabs(t/S) > eps

#     return c + s * exp(-0.5 * (a-b)**2) * S


# cdef double _marcumq_cython(double a, double b, double M, double eps=1e-7) nogil:
#     cdef:
#         bint cond = True
#         int s, c, k
#         double S, x, d, t
#         double temp = 0.
#         double z = a * b
#         double a2 = a**2 / 2
#         double b2 = b**2 / 2

#     if fabs(b) < eps:
#         return 1.

#     # if fabs(a) < eps:
#     #     for k in range(M):
#     #         temp += b**(2*k) / (2**k * gamma(k + 1))

#     #     return exp(-b**2/2) * temp

#     while cond:
#         t = (gammainc(M+k, b2) / gamma(M)) / gamma(k + 1.)
#         S += t * a2**k
#         k += 1

#         cond = fabs(t/S) > eps

#     return 1 - exp(-a2) * S


cdef double _fixed_point_finder(double m_hat, double sigma, double N, bint clip_eta=True,
                                int max_iter=100, double eps=1e-6) nogil:
    """Fixed point formula for finding eta. Table 1 p. 11 of [1]

    Input
    --------
    m_hat : double
        Initial value for the estimation of eta
    sigma : double
        Gaussian standard deviation of the noise
    N : int
        Number of coils of the acquisition (N=1 for Rician noise)
    max_iter : int, default=100
        Maximum number of iterations before breaking from the loop
    eps : double, default = 1e-4
        Criterion for reaching convergence between two subsequent estimates
    clip_eta : bool, default True
        If True, eta is clipped to 0 when below the noise floor (Bai 2014).
        If False, a new starting point m_hat is used and yields a negative eta value,
        which ensures symmetry of the normal distribution near 0 (Koay 2009).

        Having eta at zero is coherent with magnitude values being >= 0,
        but allowing negative eta is in line with the original framework
        and allows averaging of normally distributed values.

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
        if clip_eta and (m_hat < sqrt(0.5 * M_PI) * sigma):
            return 0

        delta = _beta(N) * sigma - m_hat

        if fabs(delta) < 1e-15:
            return 0

        if delta > 0:
            m = _beta(N) * sigma + delta
        else:
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

        if npy_isnan(t1): # Should not happen unless numerically unstable
            t1 = 0

        if delta > 0 and not clip_eta:
            return -t1
        else:
            return t1


# cdef double _beta(double N) nogil:
#     """Helper function for _xi, see p. 3 [1] just after eq. 8."""
#     cdef:
#         double factorialN_1 = multifactorial(N - 1)
#         double factorial2N_1 = multifactorial(2*N - 1, 2)

#     return sqrt(0.5 * M_PI) * (factorial2N_1 / (2**(N-1) * factorialN_1))


cdef double _beta(double N) nogil:
    """Helper function for _xi, see p. 3 [1] just after eq. 8.
    Generalized version for non integer N"""
    return sqrt(2) * gamma(N + 0.5) / (gamma(N))


# cdef double _fixed_point_g(double eta, double m, double sigma, double N) nogil:
#     """Helper function for _fixed_point_k, see p. 3 [1] eq. 11."""
#     return sqrt(m**2 + (_xi(eta, sigma, N) - 2*N) * sigma**2)


cdef double _fixed_point_k(double eta, double m, double sigma, double N) nogil:
    """Helper function for fixed_point_finder, see p. 11 [1] eq. D2."""
    cdef:
        double fpg, num, denom
        double eta2sigma = -eta**2/(2*sigma**2)

    fpg = sqrt(m**2 + (_xi(eta, sigma, N) - 2*N) * sigma**2)
    num = fpg * (fpg - eta)

    denom = eta * (1 - ((_beta(N)**2)/(2*N)) *
                   hyp1f1(-0.5, N, eta2sigma) *
                   hyp1f1(0.5, N+1, eta2sigma)) - fpg

    return eta - num / denom


cdef double _fixed_point_k_v2(double eta, double m, double sigma, double N) nogil:
    """Helper function for fixed_point_finder, see p. 11 [1] eq. D3.

    This is a secret equation scheme which gives rise to a different fixed point iteration
    and is only here for completion purposes, as it a replacement for eq. D2.
    Consider this as a secret bonus for looking at the code since we currently do not use it ;)"""
    cdef:
        double num, denom
        double eta2sigma = -eta**2/(2*sigma**2)
        double beta_N = _beta(N)

    num = 2 * N * sigma * (m - beta_N * sigma * hyp1f1(-0.5, N, eta2sigma))
    denom = beta_N * eta * hyp1f1(0.5, N+1, eta2sigma)

    return eta + num / denom

cdef double _xi(double eta, double sigma, double N) nogil:
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

    cdef double h1f1, out

    if fabs(sigma) < 1e-15 or (eta / sigma) > 1e8:
        return 1.

    h1f1 = hyp1f1(-0.5, N, -eta**2/(2*sigma**2))
    out = 2*N + eta**2/sigma**2 - (_beta(N) * h1f1)**2

    # Ridiculou SNR > 1e15 returns nonsense high numbers (positive or negative in the order of 1e30).
    # due to floating point precision issues, but the function is bounded between 0 and 1.
    # It starts to accumulate error around SNR ~ 1e4 though,
    # so we clip it to 1 to stay on the safe side.

    if fabs(out) > 1:
        out = 1.

    return out


# Helper function for the root finding loop
cdef inline double k(double theta, double N, double r) nogil:
    cdef:
        # Again fake SNR value for xi
        double eta = theta
        double sigma = 1.
        double g, h1f1m, h1f1p, num, denom

    g = sqrt(_xi(eta, sigma, N) * (1 + r**2) - 2*N)
    h1f1m = hyp1f1(-0.5, N, -theta**2/2)
    h1f1p = hyp1f1(0.5, N+1, -theta**2/2)

    num = g * (g - theta)
    denom = theta * (1 + r**2) * (1 - _beta(N)**2/(2*N) * h1f1m * h1f1p) - g

    return theta - num / denom


cdef double _root_finder(double r, double N, int max_iter, double eps) nogil:

    cdef:
        bint cond
        double lower_bound = sqrt((2*N / _xi(0, 1, N)) - 1)

        # This is our fake SNR value for xi
        double eta = r
        double sigma = 1.


    if r < lower_bound:
        return 0

    t0 = r - lower_bound
    t1 = k(t0, N, r)

    for _ in range(max_iter):

        cond = fabs(t1 - t0) < eps

        t0 = t1
        t1 = k(t0, N, r)

        if cond:
            break

    return t1


# Test for cython functions
def _test_marcumq_cython(a, b, M):
    return _marcumq_cython(a, b, M)


def _test_beta(N):
    return _beta(N)


# def _test_fixed_point_g(eta, m, sigma, N):
#     return _fixed_point_g(eta, m, sigma, N)


def _test_fixed_point_k(eta, m, sigma, N):
    return _fixed_point_k(eta, m, sigma, N)


def _test_xi(eta, sigma, N):
    return _xi(eta, sigma, N)


# def _test_multifactorial(N, k=1):
#     return multifactorial(N, k)


def _test_inv_cdf_gauss(y, eta, sigma):
    return _inv_cdf_gauss(y, eta, sigma)


def _test_chi_to_gauss(m, eta, sigma, N):
    return chi_to_gauss(m, eta, sigma, N)


def _test_fixed_point_finder(m_hat, sigma, N, clip_eta=True):
    return fixed_point_finder(m_hat, sigma, N, clip_eta)
