#! /usr/bin/env python

from __future__ import division, print_function

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_less, run_module_suite

from nlsam._stabilizer import _test_xi as xi
from nlsam.smoothing import (sh_smooth,
                             local_standard_deviation,
                             local_piesno)

from dipy.core.gradients import gradient_table
from dipy.sims.voxel import multi_tensor
from dipy.data import get_sphere


def test_sh_smooth():
    sphere = get_sphere('repulsion724')
    fractions = [50, 50]
    angles = [(0, 0), (60, 0)]
    gtab = gradient_table(np.ones(724) * 1000, sphere.vertices)
    mevals = np.array([[0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]])
    signal, sticks = multi_tensor(gtab, mevals, S0=1, angles=angles,
                                  fractions=fractions, snr=50)

    assert_almost_equal(signal, sh_smooth(signal, gtab), decimal=1)


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


def test_local_piesno():

    std = 50
    shape = (30, 30, 30, 3)

    for N in [1, 4, 8, 12]:
        noise = 0
        for _ in range(N):
            noise += np.random.normal(0, std, shape)**2 + np.random.normal(0, std, shape)**2

        noise = np.sqrt(noise)

        # everything less than 3% error of real value on average?
        assert_array_less(np.abs(std - local_piesno(noise, N, return_mask=False).mean()) / std, 0.03)


if __name__ == "__main__":
    run_module_suite()
