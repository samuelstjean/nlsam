#! /usr/bin/env python

from __future__ import division, print_function

import numpy as np
from numpy.testing import assert_almost_equal, run_module_suite, assert_equal

from nlsam.denoiser import reject_from_training, greedy_set_finder


def test_greedy_set_finder():
    sets = ((1, 2, 3),
            (2, 3, 4),
            (3, 4, 5),
            (1, 3, 5))

    min_set = ((1, 2, 3),
               (3, 4, 5))

    assert_equal(min_set, greedy_set_finder(sets))


def test_reject_from_training():
    indexes = np.array([[1,2,3],
                        [4,5,6],
                        [7,8,9,],
                        [10,10,10],
                        [7,8,9],
                        [1,4,5]])

    idx_reordered = np.array([[7,8,9,],
                              [10,10,10],
                              [7,8,9],
                              [1,2,3],
                              [4,5,6],
                              [1,4,5]])
    rejection = (1, 3, 4)
    rej_bool = [False, False, False, True, True, True]

    idx, rej = reject_from_training(indexes, rejection)

    assert_equal(idx, idx_reordered)
    assert_equal(rej, rej_bool)


if __name__ == "__main__":
    run_module_suite()
