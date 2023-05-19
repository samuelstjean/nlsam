#! /usr/bin/env python

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, run_module_suite

from nlsam.angular_tools import angular_neighbors, _angle, greedy_set_finder, split_shell


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


def test_split_shell():
    bvecs = np.array([[-0.923,0.254,-0.289],
                        [-0.371,-0.786,0.494],
                        [0.106,-0.563,-0.819],
                        [0.190,0.953,0.237],
                        [0.430,-0.713,0.554],
                        [0.723,-0.573,-0.388],
                        [-0.335,-0.160,0.929],
                        [0.592,0.220,0.775],
                        [0.898,0.410,-0.161],
                        [-0.184,0.800,-0.571],
                        [-0.261,0.059,0.964],
                        [-0.852,0.483,0.203],
                        [-0.406,-0.405,-0.819],
                        [0.974,0.225,0.033],
                        [0.281,0.960,-0.011],
                        [-0.899,-0.247,0.361],
                        [-0.436,0.840,-0.323],
                        [0.102,0.251,-0.963],
                        [-0.753,-0.183,-0.632],
                        [-0.550,0.680,0.484],
                        [-0.179,-0.874,-0.452],
                        [0.784,0.587,0.202],
                        [0.309,-0.420,0.854],
                        [-0.697,-0.007,0.717],
                        [-0.442,0.266,-0.856],
                        [0.793,0.178,-0.583],
                        [0.318,-0.929,-0.189],
                        [-0.478,-0.754,0.450],
                        [0.872,-0.377,0.314],
                        [0.508,0.690,0.516]])
    bvals = np.array([5 * [0], 5 * [995], 5 * [1000], 5 * [1995], 5 * [2000], 5 * [3000]]).ravel()
    dwis = np.arange(len(bvals))[bvals > 0]
    angular_size = 5
    true_idx = list(range(5,15)), list(range(15,25)), list(range(25,30))

    idx = split_shell(bvals, bvecs, angular_size, dwis, is_symmetric=False, bval_threshold=25)

    for n, ii in enumerate(idx):
        assert_equal(np.unique(ii), true_idx[n])

if __name__ == "__main__":
    run_module_suite()
