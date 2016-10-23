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
    indexes = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                        [10, 10, 10],
                        [11, 11, 11],
                        [1, 4, 5]])

    # put anything with volume 1, 3 or 4 at the end
    rejection = (1, 3, 4)

    # index set 0, 1 and 5 are thus problematics
    idx_reordered = np.take(indexes, (2, 3, 4, 0, 1, 5), axis=0)
    rej_bool = [False, False, False, True, True, True]

    idx, rej = reject_from_training(indexes, rejection)

    assert_equal(idx, idx_reordered)
    assert_equal(rej, rej_bool)


if __name__ == "__main__":
    run_module_suite()
