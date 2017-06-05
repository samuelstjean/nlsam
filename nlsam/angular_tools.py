from __future__ import division

import numpy as np


def angular_neighbors(vec, n):
    """
    Returns the indices of the n closest neighbors (excluding the vector itself)
    given an array of m points with x, y and z coordinates.

    Input : A m x 3 array, with m being the number of points, one per line.
    Each column has x, y and z coordinates for each vector.

    Output : A m x n array. Each line has the n indices of
    the closest n neighbors amongst the m input vectors.
    """

    # Sort the values and only keep the n closest neighbors.
    # The first angle is always 0, since _angle always
    # computes the angle between the vector and itself.

    return np.argsort(_angle(vec))[:, 1:n + 1]


def _angle(vec):
    """
    Inner function that finds the angle between all vectors of the input.
    The diagonal is the angle between each vector and itself, thus 0 everytime.
    It should not be called as is, since it serves mainly as a shortcut for other functions.
    """

    vec = np.array(vec)

    if vec.shape[1] != 3:
        raise ValueError("Input must be of shape N x 3. Current shape is {}".format(vec.shape))

    # Each vector is normalized to unit norm. We then replace
    # null norm vectors by 0 for sorting purposes.
    # Now each vector will have a angle of pi/2 with the null vector.

    vec = vec / np.sqrt(np.sum(vec**2, axis=1, keepdims=True))
    vec[np.isnan(vec)] = 0

    angle = [np.arccos(np.dot(vec, v[:, None]).clip(-1, 1)) for v in vec]

    return np.squeeze(angle)
