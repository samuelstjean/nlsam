#! /usr/bin/env python

from numpy.testing import run_module_suite, assert_equal
from nlsam.denoiser import greedy_set_finder

def test_greedy_set_finder():
    sets = ((1, 2, 3),
            (2, 3, 4),
            (3, 4, 5),
            (1, 3, 5))

    min_set = ((1, 2, 3),
               (3, 4, 5))

    assert_equal(min_set, greedy_set_finder(sets))


if __name__ == "__main__":
    run_module_suite()
