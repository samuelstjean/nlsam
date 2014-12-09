from __future__ import division

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt, exp

from scipy.special import erfinv

from dipy.utils.optpkg import optional_package
cython_gsl, have_cython_gsl, _ = optional_package("cython_gsl")

if not have_cython_gsl:
    raise ValueError('cannot find gsl package (required for hyp1f1), \
        try pip install cythongsl and sudo apt-get install libgsl0-dev libgsl0ldbl')

from cython_gsl cimport gsl_sf_hyperg_1F1

DEF sqrt2   = 1.4142135623730951
DEF sqrtpi2 = 1.2533141373155001


@cython.cdivision(True)
cdef double hyp1f1(double a, int b, double x):
    """Wrapper for hypergeometric series function
    http://en.wikipedia.org/wiki/Confluent_hypergeometric_function"""
    return gsl_sf_hyperg_1F1(a, b, x)


@cython.cdivision(True)
cdef double _inv_cdf_gauss(double y, double eta, double sigma):
    """Helper function for _chi_to_gauss. Returns the gaussian distributed value
    associated to a given probability. See p. 4 of [1] eq. 13.

    y : float
        Probability of observing the desired value in the normal distribution N(eta, sigma**2)
    eta :
        Mean of the normal distribution N(eta, sigma**2)
    sigma : float
        Standard deviation of the normal distribution N(eta, sigma**2)

    return :
        Value associated to probability y given a normal distribution N(eta, sigma**2)
    """
    return eta + sigma * sqrt2 * erfinv(2*y - 1)


def _chi_to_gauss(m, eta, sigma, N, alpha=1e-7):
    """Maps the noisy signal intensity from a Rician/Non central chi distribution
    to it's gaussian counterpart. See p. 4 of [1] eq. 12.

    m : float
        The noisy, Rician/Non central chi distributed value
    eta : float
        The underlying signal intensity estimated value
    sigma : float
        The gaussian noise estimated standard deviation
    N : int
        Number of coils of the acquision (N=1 for Rician noise)
    alpha : float
        Confidence interval for the cumulative distribution function.
        clips the cdf to alpha/2 <= cdf <= 1-alpha/2

    return
        float : The noisy gaussian distributed signal intensity

    Reference:
    [1]. Koay CG, Ozarslan E and Basser PJ.
    A signal transformational framework for breaking the noise floor
    and its applications in MRI.
    Journal of Magnetic Resonance 2009; 197: 108-119.
    """

    cdf = 1 - _marcumq_cython(eta/sigma, m/sigma, N)
    cdf = np.clip(cdf, alpha/2, 1 - alpha/2)
    return _inv_cdf_gauss(cdf, eta, sigma)


@cython.cdivision(True)
cdef double factorial(int N, int k=1):
    """Returns the multifactorial of order k of N.
    https://en.wikipedia.org/wiki/Factorial#Multifactorials

    N : int
        Number to compute the factorial of
    k : int,
        Order of the factorial, default k=1

    return : double
        Return type is double, because factorial(21) > 2**64
        Same as scipy.special.factorialk
    """
    cdef:
        double factorial = 1.
        int i

    for i in range(N, 0, -k):
        factorial *= i

    return factorial


@cython.cdivision(True)
cdef double _marcumq_cython(double a, double b, int M, double eps=1e-8, int max_iter=10000):
    """Computes the generalized Marcum Q function of order M.
    http://en.wikipedia.org/wiki/Marcum_Q-function

    a : float, eta/sigma
    b : float, m/sigma
    M : int, order of the function (Number of coils, N=1 for Rician noise)

    return : float
        Value of the function, always between 0 and 1 since it's a pdf.
    """
    cdef:
        double a2, b2, d, h, f, f_err, errbnd, S, factorial_M
        double temp = 0.
        int i, j, k

    if abs(a) < eps:

        for k in range(M):
            temp += b**(2*k) / (2**k * factorial(k))

        return exp(-b**2/2) * temp

    elif abs(b) < eps:
        return 1.

    else:

        a2 = 0.5 * a**2
        b2 = 0.5 * b**2.
        d = exp(-a2)
        h = exp(-a2)

        factorial_M = factorial(M)
        f = (b2**M) * exp(-b2) / factorial_M
        f_err = exp(-b2)
        errbnd = 1. - f_err

        k = 1
        S = f * h
        j = (errbnd > 4*eps)

        while j or k <= M:

            d *= a2 / k
            h += d
            f *= b2 / (k + M)
            S += (f * h)

            f_err *= b2 / k
            errbnd -= f_err

            j = (errbnd > 4*eps)
            k += 1

            if (k > max_iter):
                break

        return 1. - S


def fixed_point_finder(m_hat, sigma, N, max_iter=100, eps=1e-4):
    """Fixed point formula for finding eta. Table 1 p. 11 of [1].
    This simply wraps the cython function _fixed_point_finder

    m_hat : float
        initial value for the estimation of eta.
    sigma : float
        Gaussian standard deviation of the noise.
    N : int
        Number of coils of the acquision (N=1 for Rician noise).
    max_iter : int, default=100
        maximum number of iterations before breaking from the loop.
    eps : float, default = 1e-4
        Criterion for reaching convergence between two subsequent estimates of eta.

    return
    t1 : float
        Estimation of the underlying signal value

    Reference:
    [1]. Koay CG, Ozarslan E and Basser PJ.
    A signal transformational framework for breaking the noise floor
    and its applications in MRI.
    Journal of Magnetic Resonance 2009; 197: 108-119.
    """
    return _fixed_point_finder(m_hat, sigma, N, max_iter, eps)


@cython.cdivision(True)
cdef double _fixed_point_finder(double m_hat, double sigma, int N, int max_iter=100, double eps=1e-4):
    """Fixed point formula for finding eta. Table 1 p. 11 of [1]

    m_hat : float
        initial value for the estimation of eta
    sigma : float
        Gaussian standard deviation of the noise
    N : int
        Number of coils of the acquision (N=1 for Rician noise)
    max_iter : int, default=100
        maximum number of iterations before breaking from the loop
    eps : float, default = 1e-4
        Criterion for reaching convergence between two subsequent estimates

    return
    t1 : float
        Estimation of the underlying signal value
    """

    cdef:
        double delta, m, t0, t1
        int cond, n_iter

    delta = _beta(N) * sigma - m_hat

    if delta == 0:
        return 0
    elif delta > 0:
        m = _beta(N) * sigma + delta
    else:
        m = m_hat

    t0 = m
    t1 = _fixed_point_k(t0, m, sigma, N)
    cond = True
    n_iter = 0

    while cond:

        t0 = t1
        t1 = _fixed_point_k(t0, m, sigma, N)
        n_iter += 1
        cond = abs(t1 - t0) > eps

        if n_iter > max_iter:
            break

    if delta > 0:
        return -t1

    return t1


@cython.cdivision(True)
cdef double _beta(int N):
    """Helper function for _xi, see p. 3 [1] just after eq. 8."""

    cdef:
        double factorialN_1, factorial2N_1

    factorialN_1 = factorial(N-1)
    factorial2N_1 = factorial(2*N-1, 2)

    return sqrtpi2 * (factorial2N_1 / (2**(N-1) * factorialN_1))
    # return np.sqrt(np.pi/2) * (factorial2(2*N-1)/(2**(N-1) * factorial(N-1)))


@cython.cdivision(True)
cdef double _fixed_point_g(double eta, double m, double sigma, int N):
    """Helper function for _fixed_point_k, see p. 3 [1] eq. 11."""
    return sqrt(m**2 + (_xi(eta, sigma, N) - 2*N) * sigma**2)


@cython.cdivision(True)
cdef double _fixed_point_k(double eta, double m, double sigma, int N):
    """Helper function for _fixed_point_finder, see p. 11 [1] eq. D2."""

    cdef:
        double fpg, num, denom
        double eta2sigma = -eta**2/(2*sigma**2)

    fpg = _fixed_point_g(eta, m, sigma, N)
    num = fpg * (fpg - eta)

    denom = eta * (1 - ((_beta(N)**2)/(2*N)) *
                   hyp1f1(-0.5, N, eta2sigma) *
                   hyp1f1(0.5, N+1, eta2sigma)) - fpg

    return eta - num / denom


def corrected_sigma(eta, sigma, N):
    """Compute the local corrected standard deviation for the adaptive nonlocal means
        according to the correction factor xi.

    eta : float
        Signal intensity
    sigma : float
        Noise magnitude standard deviation
    N : int
        Number of coils of the acquisition (N=1 for Rician noise)

    return :
        Corrected sigma value, where sigma_gaussian = sigma / xi
    """

    return sigma / _xi(eta, sigma, N)


@cython.cdivision(True)
cdef double _xi(double eta, double sigma, int N):
    """Standard deviation scaling factor formula, see p. 3 of [1], eq. 10.

    eta : float
        Signal intensity
    sigma : float
        Noise magnitude standard deviation
    N : int
        Number of coils of the acquisition (N=1 for Rician noise)

    return :
        the correction factor xi, where sigma_gaussian = sigma / xi
    """

    return 2*N + eta**2/sigma**2 - (_beta(N) * hyp1f1(-0.5, N, -eta**2/(2*sigma**2)))**2


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


def _test_factorial(N, k=1):
    return factorial(N, k)


def _test_inv_cdf_gauss(y, eta, sigma):
    return _inv_cdf_gauss(y, eta, sigma)
