#! /usr/bin/env python

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, run_module_suite

from nlsam.angular_tools import angular_neighbors, _angle, greedy_set_finder


def test_angular_neighbors():

    vectors = [[0, 0, 1],
               [0, 0, 3],
               [1, 2, 3],
               [-1, -2, -3]]
    neighbors = angular_neighbors(vectors, 2)
    true_neighbors = np.array([[1, 2],
                               [0, 2],
                               [0, 1],
                               [0, 1]])

    assert_equal(neighbors, true_neighbors)


def test_angle():

    vec = [[-np.pi, 0, 0],
           [0,      0, 0],
           [0,      0, np.pi / 2]]
    angles = _angle(vec)
    correct = [[0,             np.pi / 2, np.pi / 2],
               [np.pi / 2,     np.pi / 2, np.pi / 2],
               [np.pi / 2,     np.pi / 2, 0]]

    assert_almost_equal(angles, correct)


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
