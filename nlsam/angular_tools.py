import numpy as np


def angular_neighbors(vec, n):
    """
    Returns the indices of the n closest neighbors (excluding the vector itself)
    given an array of m points with x, y and z coordinates.

    Input : A m x 3 array, with m being the number of points, one per line.
    Each column has x, y and z coordinates for each vector.

    Output : A m x n array. Each line has the n indices of
    the closest n neighbors amongst the m input vectors.

    Note : Symmetries are not considered here so a vector and its opposite sign
    counterpart will be considered far apart, even though in dMRI we consider
    (x, y, z) and -(x, y, z) to be practically identical.
    """

    # Sort the values and only keep the n closest neighbors.
    # The first angle is always 0, since _angle always
    # computes the angle between the vector and itself.
    # Therefore we pick the rest of n+1 vectors and exclude the index
    # itself if it was picked, which can happen if we have N repetition of dwis
    # but want n < N angular neighbors
    arr = np.argsort(_angle(vec))[:, :n+1]

    # We only want n elements - either we remove an index and return the remainder
    # or we don't and only return the n first indexes.
    output = np.zeros((arr.shape[0], n), dtype=np.int32)
    for i in range(arr.shape[0]):
        cond = i != arr[i]
        output[i] = arr[i, cond][:n]

    return output


def _angle(vec):
    """
    Inner function that finds the angle between all vectors of the input.
    The diagonal is the angle between each vector and itself, thus 0 everytime.
    It should not be called as is, since it serves mainly as a shortcut for other functions.

    arccos(0) = pi/2, so b0s are always far from everyone in this formulation.
    """

    vec = np.array(vec)

    if vec.shape[1] != 3:
        raise ValueError(f"Input must be of shape N x 3. Current shape is {vec.shape}")

    # Each vector is normalized to unit norm. We then replace
    # null norm vectors by 0 for sorting purposes.
    # Now each vector will have a angle of pi/2 with the null vector.
    with np.errstate(divide='ignore', invalid='ignore'):
        vec = vec / np.sqrt(np.sum(vec**2, axis=1, keepdims=True))
        vec[np.isnan(vec)] = 0

    angle = [np.arccos(np.dot(vec, v).clip(-1, 1)) for v in vec]

    return np.array(angle)


def greedy_set_finder(sets):
    """Returns a list of subsets that spans the input sets with a greedy algorithm
    http://en.wikipedia.org/wiki/Set_cover_problem#Greedy_algorithm"""

    sets = [set(s) for s in sets]
    universe = set()

    for s in sets:
        universe = universe.union(s)

    output = []

    while len(universe) != 0:

        max_intersect = 0

        for i, s in enumerate(sets):

            n_intersect = len(s.intersection(universe))

            if n_intersect > max_intersect:
                max_intersect = n_intersect
                element = i

        output.append(tuple(sets[element]))
        universe = universe.difference(sets[element])

    return output


def split_per_shell(bvals, bvecs, angular_size, dwis, is_symmetric=False, bval_threshold=25):
    '''Process each shell separately for finding the valid angular neighbors.
    Returns a list of indexes for each shell separately
    '''
    # Round similar bvals together for identifying similar shells
    rounded_bvals = np.zeros_like(bvals)
    sorted_bvals = np.sort(np.unique(bvals))

    for unique_bval in sorted_bvals:
        idx = np.abs(unique_bval - bvals) < bval_threshold
        rounded_bvals[idx] = unique_bval

    non_bzeros = np.sort(np.unique(rounded_bvals))[1:]
    neighbors = [None] * len(non_bzeros)
    bvecs_idx = np.arange(bvecs.shape[0])

    for shell, unique_bval in enumerate(non_bzeros):
        shell_bvecs = bvecs[unique_bval == rounded_bvals]
        nbvecs = shell_bvecs.shape[0]

        if is_symmetric:
            sym_bvecs = shell_bvecs
        else:
            sym_bvecs = np.vstack((shell_bvecs, -shell_bvecs))

        current_shell = angular_neighbors(sym_bvecs, angular_size - 1) % nbvecs
        current_shell = current_shell[:nbvecs]

        # convert to per shell indexes
        positions = np.arange(nbvecs)
        new_positions = bvecs_idx[unique_bval == rounded_bvals]

        # this magically works for who knows why
        # https://stackoverflow.com/questions/13572448/replace-values-of-a-numpy-index-array-with-values-of-a-list?answertab=votes#tab-top
        index = np.digitize(current_shell.ravel(), positions, right=True)
        current_shell = new_positions[index].reshape(current_shell.shape)

        current_shell = [(dwi,) + tuple(current_shell[pos]) for pos, dwi in enumerate(new_positions) if dwi in dwis]
        neighbors[shell] = current_shell
    return neighbors
