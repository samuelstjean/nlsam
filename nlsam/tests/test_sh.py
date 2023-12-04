import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_equal, assert_raises

from nlsam.stabilizer import _test_xi as xi
from nlsam.sh import cart2sphere, sph_harm_ind_list, real_sh_descoteaux_from_index, smooth_pinv, sh_smooth

theta = np.array([1.57079633, 1.57079633, 0.55357436, 0.55357436, 1.01722197,
       1.01722197, 1.04719755, 1.04719755, 0.        , 1.25663706,
       1.25663706, 0.62831853, 0.62831853, 0.62831853, 0.62831853,
       1.57079633, 1.04719755, 1.25663706, 1.04719755, 1.25663706,
       1.57079633, 1.30788606, 0.79252919, 1.30788606, 0.79252919,
       0.27678718, 0.27678718, 1.40947549, 1.12199202, 1.13147493,
       1.40947549, 1.12199202, 1.13147493, 0.53027408, 0.80407115,
       0.81180364, 0.31415927, 0.9424778 , 0.53027408, 0.80407115,
       0.81180364, 0.31415927, 0.9424778 , 0.55357436, 0.53027408,
       0.80407115, 0.31415927, 0.53027408, 0.80407115, 0.31415927,
       0.55357436, 1.57079633, 1.57079633, 1.30480532, 0.79252919,
       1.30788606, 1.01722197, 0.81180364, 1.30480532, 1.12199202,
       1.40947549, 0.9424778 , 1.13147493, 1.30480532, 0.79252919,
       1.30788606, 1.01722197, 0.81180364, 1.30480532, 1.12199202,
       1.40947549, 0.9424778 , 1.13147493, 1.29400915, 1.29400915,
       1.40761322, 1.40761322, 1.57079633, 1.40761322, 1.57079633,
       1.40761322])

phi = np.array([ 5.53574359e-01,  2.58801829e+00,  0.00000000e+00, -3.14159265e+00,
        1.57079633e+00, -1.57079633e+00,  3.64863828e-01,  2.77672883e+00,
       -3.14159265e+00,  1.01722197e+00,  2.12437069e+00,  1.01722197e+00,
        2.12437069e+00, -1.01722197e+00, -2.12437069e+00, -1.22464680e-16,
       -2.77672883e+00, -2.12437069e+00, -3.64863828e-01, -1.01722197e+00,
        1.57079633e+00,  4.66013568e-01,  2.27508776e-01,  2.67557909e+00,
        2.91408388e+00, -7.19829328e-17, -3.14159265e+00,  7.79476318e-01,
        1.27817929e+00,  7.06870652e-01,  2.36211634e+00,  1.86341337e+00,
        2.43472200e+00,  5.39671437e-01,  1.34587633e+00,  6.26394387e-01,
        1.01722197e+00,  1.01722197e+00,  2.60192122e+00,  1.79571633e+00,
        2.51519827e+00,  2.12437069e+00,  2.12437069e+00,  1.57079633e+00,
       -5.39671437e-01, -1.34587633e+00, -1.01722197e+00, -2.60192122e+00,
       -1.79571633e+00, -2.12437069e+00, -1.57079633e+00,  2.86480547e+00,
        2.76787179e-01,  2.97240525e+00, -2.91408388e+00, -2.67557909e+00,
       -3.14159265e+00, -2.51519827e+00, -2.97240525e+00, -1.86341337e+00,
       -2.36211634e+00, -2.12437069e+00, -2.43472200e+00,  1.69187399e-01,
       -2.27508776e-01, -4.66013568e-01, -3.43077782e-17, -6.26394387e-01,
       -1.69187399e-01, -1.27817929e+00, -7.79476318e-01, -1.01722197e+00,
       -7.06870652e-01, -1.57079633e+00,  1.57079633e+00, -1.84045732e+00,
       -1.30113533e+00,  1.01722197e+00,  1.30113533e+00,  2.12437069e+00,
        1.84045732e+00])


def test_sph_harm_ind_list():
    m_list, n_list = sph_harm_ind_list(8)
    assert_equal(m_list.shape, n_list.shape)
    assert_equal(m_list.shape, (45,))
    assert(np.all(np.abs(m_list) <= n_list))
    assert_array_equal(n_list % 2, 0)
    assert_raises(ValueError, sph_harm_ind_list, 1)

    # Test for a full basis
    m_list, n_list = sph_harm_ind_list(8, True)
    assert_equal(m_list.shape, n_list.shape)
    # There are (sh_order + 1) * (sh_order + 1) coefficients
    assert_equal(m_list.shape, (81,))
    assert(np.all(np.abs(m_list) <= n_list))


def test_smooth_pinv():
    m, n = sph_harm_ind_list(4)
    B = real_sh_descoteaux_from_index(
    m, n, theta[:, None], phi[:, None])

    L = np.zeros(len(m))
    C = smooth_pinv(B, L)
    D = np.dot(np.linalg.inv(np.dot(B.T, B)), B.T)
    assert_array_almost_equal(C, D)

    L = n * (n + 1) * .05
    C = smooth_pinv(B, L)
    L = np.diag(L)
    D = np.dot(np.linalg.inv(np.dot(B.T, B) + L * L), B.T)

    assert_array_almost_equal(C, D)

    L = np.arange(len(n)) * .05
    C = smooth_pinv(B, L)
    L = np.diag(L)
    D = np.dot(np.linalg.inv(np.dot(B.T, B) + L * L), B.T)
    assert_array_almost_equal(C, D)


def test_real_sh_descoteaux_from_index():
    # Tests derived from tables in
    # http://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    # where real spherical harmonic $Y^m_n$ is defined to be:
    #    Real($Y^m_n$) * sqrt(2) if m > 0
    #    $Y^m_n$                 if m == 0
    #    Imag($Y^m_n$) * sqrt(2) if m < 0

    rsh = real_sh_descoteaux_from_index
    pi = np.pi
    sqrt = np.sqrt
    sin = np.sin
    cos = np.cos

    assert_array_almost_equal(rsh(0, 0, 0, 0),
                                0.5 / sqrt(pi))
    assert_array_almost_equal(rsh(-2, 2, pi / 5, pi / 3),
                                0.25 * sqrt(15. / (2. * pi)) *
                                (sin(pi / 5.)) ** 2. * cos(0 + 2. * pi / 3) *
                                sqrt(2))
    assert_array_almost_equal(rsh(2, 2, pi / 5, pi / 3),
                                -1 * 0.25 * sqrt(15. / (2. * pi)) *
                                (sin(pi / 5.)) ** 2. * sin(0 - 2. * pi / 3) *
                                sqrt(2))
    assert_array_almost_equal(rsh(-2, 2, pi / 2, pi),
                                0.25 * sqrt(15 / (2. * pi)) *
                                cos(2. * pi) * sin(pi / 2.) ** 2. * sqrt(2))
    assert_array_almost_equal(rsh(2, 4, pi / 3., pi / 4.),
                                -1 * (3. / 8.) * sqrt(5. / (2. * pi)) *
                                sin(0 - 2. * pi / 4.) *
                                sin(pi / 3.) ** 2. *
                                (7. * cos(pi / 3.) ** 2. - 1) * sqrt(2))
    assert_array_almost_equal(rsh(-4, 4, pi / 6., pi / 8.),
                                (3. / 16.) * sqrt(35. / (2. * pi)) *
                                cos(0 + 4. * pi / 8.) * sin(pi / 6.) ** 4. *
                                sqrt(2))
    assert_array_almost_equal(rsh(4, 4, pi / 6., pi / 8.),
                                -1 * (3. / 16.) * sqrt(35. / (2. * pi)) *
                                sin(0 - 4. * pi / 8.) * sin(pi / 6.) ** 4. *
                                sqrt(2))

    aa = np.ones((3, 1, 1, 1))
    bb = np.ones((1, 4, 1, 1))
    cc = np.ones((1, 1, 5, 1))
    dd = np.ones((1, 1, 1, 6))

    assert_equal(rsh(aa, bb, cc, dd).shape, (3, 4, 5, 6))


def test_smooth_pinv():
    m, n = sph_harm_ind_list(4)

    B = real_sh_descoteaux_from_index(
    m, n, theta[:, None], phi[:, None])

    L = np.zeros(len(m))
    C = smooth_pinv(B, L)
    D = np.dot(np.linalg.inv(np.dot(B.T, B)), B.T)
    assert_array_almost_equal(C, D)

    L = n * (n + 1) * .05
    C = smooth_pinv(B, L)
    L = np.diag(L)
    D = np.dot(np.linalg.inv(np.dot(B.T, B) + L * L), B.T)

    assert_array_almost_equal(C, D)

    L = np.arange(len(n)) * .05
    C = smooth_pinv(B, L)
    L = np.diag(L)
    D = np.dot(np.linalg.inv(np.dot(B.T, B) + L * L), B.T)
    assert_array_almost_equal(C, D)
