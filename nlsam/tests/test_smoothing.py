#! /usr/bin/env python

from __future__ import division, print_function

import numpy as np
from numpy.testing import assert_array_less, run_module_suite

from nlsam.bias_correction import xi
from nlsam.smoothing import local_standard_deviation


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


if __name__ == "__main__":
    run_module_suite()
