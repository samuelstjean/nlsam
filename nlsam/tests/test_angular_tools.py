#! /usr/bin/env python

import numpy as np
from numpy.testing import assert_allclose, assert_equal, run_module_suite

from nlsam.angular_tools import angular_neighbors, _angle


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

    assert_allclose(angles, correct)


if __name__ == "__main__":
    run_module_suite()
