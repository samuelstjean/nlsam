#cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
cimport cython

from libc.math cimport sqrt, exp, fabs, M_PI

from nlsam.multiprocess import multiprocesser
from scipy.special.cython_special cimport ndtri, ive

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

    Note
    -----
    The original paper mentions to use erfinv, but we have directly that
    ndtri(y) = sqrt(2) * erfinv(2*y - 1)
    """
    return eta + sigma * ndtri(y)


cdef double _chi_to_gauss(double m, double eta, double sigma, int N,
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


# Stolen from octave signal marcumq
cdef double _marcumq_cython(double a, double b, int M, double eps=1e-7) nogil:
    """Computes the generalized Marcum Q function of order M.
    http://en.wikipedia.org/wiki/Marcum_Q-function

    a : double, eta/sigma
    b : double, m/sigma
    M : int, order of the function (Number of coils, N=1 for Rician noise)

    return : double
        Value of the function, always between 0 and 1 since it's a pdf.
    """
    cdef:
        bint cond = True
        int s, c, k
        double S, x, d, t
        double temp = 0.
        double z = a * b

    if fabs(b) < eps:
        return 1.

    if fabs(a) < eps:
        for k in range(M):
            temp += b**(2*k) / (2**k * multifactorial(k))

        return exp(-b**2/2) * temp

    if a < b:
        s = 1
        c = 0
        x = a / b
        d = x
        S = ive(0, z)

        for k in range(1, M):
            S += (d + 1/d) * ive(k, z)
            d *= x

        k = M
    else:
        s = -1
        c = 1
        x = b / a
        k = M
        d = x**M
        S = 0

    while cond:
        t = d * ive(k, z)
        S += t
        d *= x
        k += 1

        cond = fabs(t/S) > eps

    return c + s * exp(-0.5 * (a-b)**2) * S


cdef double _fixed_point_finder(double m_hat, double sigma, int N, bint clip_eta=True,
                                int max_iter=100, double eps=1e-4) nogil:
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


cdef double _fixed_point_k_v2(double eta, double m, double sigma, int N) nogil:
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
    return 2*N + eta**2/sigma**2 - (_beta(N) * h1f1)**2


# Test for cython functions
def _test_marcumq_cython(a, b, M):
    return _marcumq_cython(a, b, M)


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


def _test_fixed_point_finder(m_hat, sigma, N, clip_eta=True):
    return fixed_point_finder(m_hat, sigma, N, clip_eta)
