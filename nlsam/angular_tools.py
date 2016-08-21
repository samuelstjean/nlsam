from __future__ import division

import numpy as np
from dipy.core.sphere import Sphere


def sphere_neighbors(vec, sphere, n):
    """
    Returns the indices of the n closest neighbors of vec
    on the sphere given an array of m points with x, y and z coordinates.
    sphere can also be a dipy sphere object. It will be converted to the
    right format internally.

    You can also supply a n x 3 array as vec, but be aware that the closest
    vector is considered to be the first one. The remaining n-1 x 3 vectors
    will be appended to sphere.

    Input : vec : A 1 x 3 array organized as x, y and z coordinates.
            sphere : A m x 3 array organized as x, y and z coordinates.
                     Can also be a dipy sphere object.

    Output : 1D array giving the indices of the n closest points on the sphere
    to vec.
    """

    # Promote vec to 1D array if needed
    if len(vec.shape) == 1:
        vec = vec[None, :]

    # Check is sphere is a Sphere object and convert it if needed
    if isinstance(sphere, Sphere):
        sphere = np.array((sphere.x, sphere.y, sphere.z)).T

    return angular_neighbors(np.concatenate((vec, sphere), axis=0), n)[0]


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

    return np.argsort(_angle(vec))[:, 1:n+1]


def _angle(vec):
    """
    Inner function that finds the angle between all vectors of the input.
    The diagonal is the angle between each vector and itself, thus 0 everytime.
    It should not be called as is, since it serves mainly as a shortcut for other functions.
    """

    if vec.shape[1] != 3:
        raise ValueError("Input must be of shape N x 3. Current shape is", vec.shape)

    # Each vector is normalized to unit norm. We then replace
    # null norm vectors by 0 for sorting purposes.
    # Now each vector will have a angle of pi/2 with the null vector.

    vec = vec / np.sqrt(np.sum(vec**2, axis=1, keepdims=True))
    vec[np.isnan(np.sum(vec, axis=1))] = 0

    angle = np.zeros((vec.shape[0], vec.shape[0]))

    # Find list of angle values for each vector in the input array.
    for i in range(angle.shape[0]):

        # The dot product must be clipped between -1 and 1. Since we have unit
        # norm vectors that's the theoretical limit. Rounding error can produce
        # nan on colinear vectors (the diagonal of angle) without this fix.
        angle[i] = np.squeeze(np.arccos(np.clip(np.dot(vec, vec[i, None].T), -1, 1)))

    return angle
