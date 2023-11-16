# cython: wraparound=False, cdivision=True, boundscheck=False, language_level=3, embedsignature=True, infer_types=True

import numpy as np

cdef void _im2col3D(double[:,:,::1] A, double[::1,:] R, int[:] size) noexcept nogil:

    cdef:
        Py_ssize_t k = 0, l = 0
        Py_ssize_t a, b, c, m, n, o
        Py_ssize_t x = A.shape[0], y = A.shape[1], z = A.shape[2]
        Py_ssize_t s0 = size[0], s1 = size[1], s2 = size[2]

    # with nogil:
    for a in range(x - s0 + 1):
        for b in range(y - s1 + 1):
            for c in range(z - s2 + 1):

                l = 0

                for m in range(s0):
                    for n in range(s1):
                        for o in range(s2):
                            R[l, k] = A[a+m, b+n, c+o]
                            l += 1
                k += 1


cdef void _im2col3D_overlap(double[:,:,::1] A, double[::1,:] R, int[:] size, int[:] overlap):

    cdef:
        Py_ssize_t k = 0, l = 0
        Py_ssize_t a, b, c, m, n, o
        Py_ssize_t x = A.shape[0], y = A.shape[1], z = A.shape[2]
        Py_ssize_t s0 = size[0], s1 = size[1], s2 = size[2]
        Py_ssize_t over0 = overlap[0], over1 = overlap[1], over2 = overlap[2]

       # with nogil:
    for a in range(0, x - over0, s0 - over0):
        for b in range(0, y - over1, s1 - over1):
            for c in range(0, z - over2, s2 - over2):

                l = 0

                for m in range(s0):
                    for n in range(s1):
                        for o in range(s2):
                            R[l, k] = A[a+m, b+n, c+o]
                            l += 1

                k += 1


cdef void _im2col4D(double[:,:,:,::1] A, double[::1,:] R, int[:] size) noexcept nogil:

    cdef:
        Py_ssize_t k = 0, l = 0
        Py_ssize_t a, b, c, d, m, n, o, p
        Py_ssize_t x = A.shape[0], y = A.shape[1], z = A.shape[2], t = A.shape[3]
        Py_ssize_t s0 = size[0], s1 = size[1], s2 = size[2]

    # with nogil:
    for a in range(x - s0 + 1):
        for b in range(y - s1 + 1):
            for c in range(z - s2 + 1):

                l = 0

                for m in range(s0):
                    for n in range(s1):
                        for o in range(s2):
                            for p in range(t):

                                R[l, k] = A[a+m, b+n, c+o, p]
                                l += 1

                k += 1


def im2col_nd(A, block_shape, overlap):
    """
    Returns a 2d array of shape flat(block_shape) by A.shape/block_shape made
    from blocks of a nd array.
    """

    block_shape = np.array(block_shape, dtype=np.int32)
    overlap = np.array(overlap, dtype=np.int32)

    if (overlap < 0).any() or (block_shape < overlap).any():
        raise ValueError('Invalid overlap value, it must lie between 0 and min(block_size)-1', overlap, block_shape)

    A = padding(A, block_shape, overlap)

    if len(A.shape) != len(block_shape):
        raise ValueError("Number of dimensions mismatch!", A.shape, block_shape)

    dim0 = np.prod(block_shape)
    dim1 = np.prod(A.shape - block_shape + 1)
    dtype = np.float64

    A = np.array(A, dtype=dtype)
    R = np.zeros((dim0, dim1), dtype=dtype, order='F')

    # if A is zeros, R will also be zeros
    if not np.any(A):
        return R

    if len(A.shape) == 3:
        if np.sum(block_shape - overlap) > len(A.shape):
            _im2col3D_overlap(A, R, block_shape, overlap)
        else:
            _im2col3D(A, R, block_shape)
    elif len(A.shape) == 4:
            _im2col4D(A, R, block_shape)
    else:
        raise ValueError("3D or 4D supported only!", A.shape)

    return R


cdef void _col2im3D_overlap(double[:,:,::1] A, double[:,:,::1] div, double[::1,:] R, double[:] weights, int[:] block_shape, int[:] overlap):
    cdef:
        Py_ssize_t k = 0, l = 0
        Py_ssize_t a, b, c, m, n, o
        Py_ssize_t x = A.shape[0], y = A.shape[1], z = A.shape[2]
        Py_ssize_t s0 = block_shape[0], s1 = block_shape[1], s2 = block_shape[2]
        Py_ssize_t over0 = overlap[0], over1 = overlap[1], over2 = overlap[2]

    # with nogil:
    for a in range(0, x - over0, s0 - over0):
        for b in range(0, y - over1, s1 - over1):
            for c in range(0, z - over2, s2 - over2):

                l = 0

                for m in range(s0):
                    for n in range(s1):
                        for o in range(s2):

                            A[a+m, b+n, c+o] += R[l, k] * weights[k]
                            div[a+m, b+n, c+o] += weights[k]
                            l += 1
                k += 1


cdef void _col2im3D(double[:,:,::1] A, double[:,:,::1] div, double[::1,:] R, double[:] weights, int[:] block_shape) noexcept nogil:

    cdef:
        Py_ssize_t k = 0, l = 0
        Py_ssize_t a, b, c, m, n, o
        Py_ssize_t x = A.shape[0], y = A.shape[1], z = A.shape[2]
        Py_ssize_t s0 = block_shape[0], s1 = block_shape[1], s2 = block_shape[2]

    # with nogil:
    for a in range(x - s0 + 1):
        for b in range(y - s1 + 1):
            for c in range(z - s2 + 1):

                l = 0

                for m in range(s0):
                    for n in range(s1):
                        for o in range(s2):

                            A[a+m, b+n, c+o] += R[l, k] * weights[k]
                            div[a+m, b+n, c+o] += weights[k]
                            l += 1
                k += 1

cdef void _col2im4D_overlap(double[:,:,:,::1] A, double[:,:,::1] div, double[::1,:] R, double[:] weights, int[:] block_shape, int[:] overlap):
    cdef:
        Py_ssize_t k = 0, l = 0
        Py_ssize_t a, b, c, d, m, n, o, p
        Py_ssize_t x = A.shape[0], y = A.shape[1], z = A.shape[2], t = A.shape[3]
        Py_ssize_t s0 = block_shape[0], s1 = block_shape[1], s2 = block_shape[2]
        Py_ssize_t over0 = overlap[0], over1 = overlap[1], over2 = overlap[2]

    # with nogil:
    for a in range(0, x - s0, s0 - over0):
        for b in range(0, y - s1, s1 - over1):
            for c in range(0, z - s2, s2 - over2):

                l = 0

                for m in range(s0):
                    for n in range(s1):
                        for o in range(s2):
                            for p in range(t):
                                A[a+m, b+n, c+o, p] += R[l, k] * weights[k]
                                l += 1
                            div[a+m, b+n, c+o] += weights[k]
                k += 1

cdef void _col2im4D(double[:,:,:,::1] A, double[:,:,::1] div, double[::1,:] R, double[:] weights, int[:] block_shape) noexcept nogil:
    cdef:
        Py_ssize_t k = 0, l = 0
        Py_ssize_t a, b, c, d, m, n, o, p
        Py_ssize_t x = A.shape[0], y = A.shape[1], z = A.shape[2], t = A.shape[3]
        Py_ssize_t s0 = block_shape[0], s1 = block_shape[1], s2 = block_shape[2]

    # with nogil:
    for a in range(x - s0 + 1):
        for b in range(y - s1 + 1):
            for c in range(z - s2 + 1):

                l = 0

                for m in range(s0):
                    for n in range(s1):
                        for o in range(s2):
                            for p in range(t):
                                A[a+m, b+n, c+o, p] += R[l, k] * weights[k]
                                l += 1

                            div[a+m, b+n, c+o] += weights[k]
                k += 1


def col2im_nd(R, block_shape, end_shape, overlap, weights=None):
    """
    Returns a nd array of shape end_shape from a 2D array made of flatenned
    block that had originally a shape of block_shape.
    Inverse function of im2col_nd.
    """

    block_shape = np.array(block_shape, dtype=np.int32)
    end_shape = np.array(end_shape, dtype=np.int32)
    overlap = np.array(overlap, dtype=np.int32)
    dtype = np.float64

    if (overlap < 0).any() or ((block_shape < overlap).any()):
        raise ValueError(f'Invalid overlap value, it must lie between 0 and {min(block_shape) - 1}', overlap, block_shape)

    if weights is None:
        weights = np.ones(R.shape[1], dtype=dtype)
    else:
        weights = np.asarray(weights, dtype=dtype)

    R = np.asarray(R, order='F', dtype=dtype)
    A = np.zeros(end_shape, dtype=dtype)
    div = np.zeros(end_shape[:3], dtype=dtype)

    # if R is zeros, A will also be zeros
    if not np.any(R):
        return A

    overlap = overlap[:3]

    if len(A.shape) == 3:
        block_shape = block_shape[:3]

        if np.sum(block_shape - overlap) > len(A.shape):
            _col2im3D_overlap(A, div, R, weights, block_shape, overlap)
        else:
            _col2im3D(A, div, R, weights, block_shape)
    elif len(A.shape) == 4:
        # if np.any(overlap != 1):
        #     _col2im4D_overlap(A, div, R, weights, block_shape, overlap)
        # else:
        _col2im4D(A, div, R, weights, block_shape)
        div = div[..., None]
    else:
        raise ValueError("3D or 4D supported only!", A.shape)

    A /= div
    return A


def padding(A, block_shape, overlap):
    """
    Pad A at the end so that block_shape will cut an integer number of blocks
    across all dimensions. A is padded with 0s.
    """

    shape = A.shape[:3]
    block_shape = np.array(block_shape)[:3]
    overlap = np.array(overlap)[:3]

    fit = ((shape - block_shape) % (block_shape - overlap))
    fit = np.where(fit > 0, block_shape - overlap - fit, fit)

    if np.sum(fit) == 0:
        return A

    padding = np.array(shape) + fit
    if A.ndim == 4:
        padding = np.append(padding,  A.shape[3])

    padded = np.zeros(padding, dtype=A.dtype)
    padded[:shape[0], :shape[1], :shape[2]] = A

    return padded
