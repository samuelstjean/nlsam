import numpy as np
from numpy.testing import assert_array_less, assert_equal, assert_array_equal, assert_raises, assert_allclose, assert_array_almost_equal_nulp

from nlsam.stabilizer import _test_xi as xi
from nlsam.smoothing import local_standard_deviation, real_sh_descoteaux_from_index, sph_harm_ind_list, smooth_pinv


def test_local_standard_deviation():

    # SNR = 20
    mean = 100
    std = 5
    shape = (30, 30, 30, 3)

    for N in [1, 4, 8, 12]:
        noise = 0
        for _ in range(N):
            noise += np.random.normal(mean, std, shape)**2 + np.random.normal(mean, std, shape)**2

        noise = np.sqrt(noise)
        corrected_std = local_standard_deviation(noise) / np.sqrt(xi(mean, std, N))

        # everything less than 10% error of real value?
        assert_array_less(np.abs(std - corrected_std.mean()) / std, 0.1)

    # This estimation has a harder time at low SNR, high coils value, probably due
    # to how the synthetic noise field is computed
    # SNR = 5
    mean = 250
    std = 50
    shape = (30, 30, 30, 3)

    for N in [1, 4, 8, 12]:
        noise = 0
        for _ in range(N):
            noise += np.random.normal(mean, std, shape)**2 + np.random.normal(mean, std, shape)**2

        noise = np.sqrt(noise)
        corrected_std = local_standard_deviation(noise) / np.sqrt(xi(mean, std, N))

        # everything less than 10% error of real value?
        assert_array_less(np.abs(std - corrected_std.mean()) / std, 0.1)


# Copyright (c) 2008-2024, dipy developers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.

#     * Neither the name of the dipy developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def test_sph_harm_ind_list():
    m_list, l_list = sph_harm_ind_list(8)
    assert_equal(m_list.shape, l_list.shape)
    assert_equal(m_list.shape, (45,))
    assert (np.all(np.abs(m_list) <= l_list))
    assert_array_equal(l_list % 2, 0)
    assert_raises(ValueError, sph_harm_ind_list, 1)


def test_real_sh_descoteaux_from_index():
    # Tests derived from tables in
    # https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    # where real spherical harmonic $Y_l^m$ is defined to be:
    #    Real($Y_l^m$) * sqrt(2) if m > 0
    #    $Y_l^m$                 if m == 0
    #    Imag($Y_l^m$) * sqrt(2) if m < 0

    rsh = real_sh_descoteaux_from_index
    pi = np.pi
    sqrt = np.sqrt
    sin = np.sin
    cos = np.cos

    assert_allclose(rsh(0, 0, 0, 0), 0.5 / sqrt(pi))
    assert_allclose(
        rsh(-2, 2, pi / 5, pi / 3),
        0.25
        * sqrt(15.0 / (2.0 * pi))
        * (sin(pi / 5.0)) ** 2.0
        * cos(0 + 2.0 * pi / 3)
        * sqrt(2),
    )
    assert_allclose(
        rsh(2, 2, pi / 5, pi / 3),
        -1
        * 0.25
        * sqrt(15.0 / (2.0 * pi))
        * (sin(pi / 5.0)) ** 2.0
        * sin(0 - 2.0 * pi / 3)
        * sqrt(2),
    )
    assert_allclose(
        rsh(-2, 2, pi / 2, pi),
        0.25
        * sqrt(15 / (2.0 * pi))
        * cos(2.0 * pi)
        * sin(pi / 2.0) ** 2.0
        * sqrt(2),
    )
    assert_allclose(
        rsh(2, 4, pi / 3.0, pi / 4.0),
        -1
        * (3.0 / 8.0)
        * sqrt(5.0 / (2.0 * pi))
        * sin(0 - 2.0 * pi / 4.0)
        * sin(pi / 3.0) ** 2.0
        * (7.0 * cos(pi / 3.0) ** 2.0 - 1)
        * sqrt(2),
    )
    assert_allclose(
        rsh(-4, 4, pi / 6.0, pi / 8.0),
        (3.0 / 16.0)
        * sqrt(35.0 / (2.0 * pi))
        * cos(0 + 4.0 * pi / 8.0)
        * sin(pi / 6.0) ** 4.0
        * sqrt(2),
    )
    assert_allclose(
        rsh(4, 4, pi / 6.0, pi / 8.0),
        -1
        * (3.0 / 16.0)
        * sqrt(35.0 / (2.0 * pi))
        * sin(0 - 4.0 * pi / 8.0)
        * sin(pi / 6.0) ** 4.0
        * sqrt(2),
    )

    aa = np.ones((3, 1, 1, 1))
    bb = np.ones((1, 4, 1, 1))
    cc = np.ones((1, 1, 5, 1))
    dd = np.ones((1, 1, 1, 6))

    assert_equal(rsh(aa, bb, cc, dd).shape, (3, 4, 5, 6))


def test_smooth_pinv():
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

    m_values, l_values = sph_harm_ind_list(4)

    B = real_sh_descoteaux_from_index(m_values, l_values, theta[:, None], phi[:, None])
    BtB = B.T @ B

    L = np.zeros(len(m_values))
    C = smooth_pinv(B, L)
    D = np.linalg.pinv(BtB) @ B.T
    assert_allclose(C, D, atol=1e-15)

    L = l_values * (l_values + 1) * 0.05
    C = smooth_pinv(B, L)
    L = np.diag(L)
    D = np.linalg.pinv(BtB + L * L) @ B.T
    assert_allclose(C, D, atol=1e-15)

    L = np.arange(len(l_values)) * 0.05
    C = smooth_pinv(B, L)
    L = np.diag(L)
    D = np.linalg.pinv(BtB + L * L) @ B.T
    assert_allclose(C, D, atol=1e-15)
