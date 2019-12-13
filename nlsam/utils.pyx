#cython: wraparound=False, cdivision=True, boundscheck=False

from __future__ import print_function

import numpy as np
cimport numpy as np
cimport cython


cdef void _im2col3D(double[::1,:,:] A, double[::1,:] R, int[:] size) nogil:

    cdef:
        int k = 0, l = 0
        int a, b, c, m, n, o
        int x = A.shape[0], y = A.shape[1], z =  A.shape[2]
        int s0 = size[0], s1 = size[1], s2 = size[2]

    with nogil:
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


cdef void _im2col3D_overlap(double[::1,:,:] A, double[::1,:] R, int[:] size, int[:] overlap):

    cdef:
        int k = 0, l = 0
        int a, b, c, m, n, o
        int x = A.shape[0], y = A.shape[1], z =  A.shape[2]
        int s0 = size[0], s1 = size[1], s2 = size[2]
        int over0 = overlap[0], over1 = overlap[1], over2 = overlap[2]

    for a in range(0, x - over0, s0 - over0):
        for b in range(0, y - over1, s1 - over1):
            for c in range(0, z - over2, s2 - over2):

                with nogil:
                    l = 0

                    for m in range(s0):
                        for n in range(s1):
                            for o in range(s2):
                                R[l, k] = A[a+m, b+n, c+o]
                                l += 1

                    k += 1


cdef void _im2col4D(double[::1,:,:,:] A, double[::1,:] R, int[:] size) nogil:

    cdef:
        int k = 0, l = 0
        int a, b, c, d, m, n, o, p
        int x = A.shape[0], y = A.shape[1], z = A.shape[2], t = A.shape[3]
        int s0 = size[0], s1 = size[1], s2 = size[2]

    with nogil:
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


def im2col_nd(A, block_shape, overlap, order='F'):
    """
    Returns a 2d array of shape flat(block_shape) by A.shape/block_shape made
    from blocks of a nd array.
    """

    block_shape = np.array(block_shape, dtype=np.int32)
    overlap = np.array(overlap, dtype=np.int32)

    if (overlap.any() < 0) or ((block_shape < overlap).any()):
        raise ValueError('Invalid overlap value, it must lie between 0' +
                         'and min(block_size)-1', overlap, block_shape)
    A = padding(A, block_shape, overlap)
    A = np.asarray(A, dtype=np.float64, order='F')

    if len(A.shape) != len(block_shape):
        raise ValueError("Number of dimensions mismatch!", A.shape, block_shape)

    dim0 = np.prod(block_shape)
    dim1 = np.prod(A.shape - block_shape + 1)
    R = np.zeros((dim0, dim1), dtype=np.float64, order='F')

    # if A is zeros, R is also gonna be zeros
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


cdef void _col2im3D_overlap(double[::1,:,:] A, double[::1,:,:] div, double[:,:] R, double[:] weights, int[:] block_shape, int[:] overlap):
    cdef:
        int k = 0, l = 0
        int a, b, c, m, n, o
        int x = A.shape[0], y = A.shape[1], z =  A.shape[2]
        int s0 = block_shape[0], s1 = block_shape[1], s2 = block_shape[2]
        int over0 = overlap[0], over1 = overlap[1], over2 = overlap[2]

    for a in range(0, x - over0, s0 - over0):
        for b in range(0, y - over1, s1 - over1):
            for c in range(0, z - over2, s2 - over2):

                with nogil:
                    l = 0

                    for m in range(s0):
                        for n in range(s1):
                            for o in range(s2):

                                A[a+m, b+n, c+o] += R[l, k] * weights[k]
                                div[a+m, b+n, c+o] += weights[k]
                                l += 1
                    k += 1


cdef void _col2im3D(double[::1,:,:] A, double[::1,:,:] div, double[::1,:] R, double[:] weights, int[:] block_shape) nogil:

    cdef:
        int k = 0, l = 0
        int a, b, c, m, n, o
        int x = A.shape[0], y = A.shape[1], z =  A.shape[2]
        int s0 = block_shape[0], s1 = block_shape[1], s2 = block_shape[2]

    with nogil:
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


cdef void _col2im4D(double[::1,:,:,:] A, double[::1,:,:] div, double[::1,:] R, double[:] weights, int[:] block_shape) nogil:
    cdef:
        int k = 0, l = 0
        int a, b, c, d, m, n, o, p
        int x = A.shape[0], y = A.shape[1], z = A.shape[2], t = A.shape[3]
        int s0 = block_shape[0], s1 = block_shape[1], s2 = block_shape[2]

    with nogil:
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


def col2im_nd(R, block_shape, end_shape, overlap, weights=None, order='F'):
    """
    Returns a nd array of shape end_shape from a 2D array made of flatenned
    block that had originally a shape of block_shape.
    Inverse function of im2col_nd.
    """

    block_shape = np.array(block_shape, dtype=np.int32)
    end_shape = np.array(end_shape, dtype=np.int32)
    overlap = np.array(overlap, dtype=np.int32)

    if (overlap.any() < 0) or ((block_shape < overlap).any()):
        raise ValueError('Invalid overlap value, it must lie between 0 \
                         \nand min(block_size)-1', overlap, block_shape)

    if weights is None:
        weights = np.ones(R.shape[1], dtype=np.float64, order=order)
    else:
        weights = np.asarray(weights, dtype=np.float64, order=order)

    R = np.asarray(R, dtype=np.float64, order=order)
    A = np.zeros(end_shape, dtype=np.float64, order=order)
    div = np.zeros(end_shape[:3], dtype=np.float64, order=order)

    # if R is zeros, A is also gonna be zeros
    if not np.any(R):
        return A

    if len(A.shape) == 3:
        block_shape = block_shape[:3]
        overlap = overlap[:3]

        if np.sum(block_shape - overlap) > len(A.shape):
            _col2im3D_overlap(A, div, R, weights, block_shape, overlap)
        else:
            _col2im3D(A, div, R, weights, block_shape)
    elif len(A.shape) == 4:
        _col2im4D(A, div, R, weights, block_shape)
        div = div[..., None]
    else:
        raise ValueError("3D or 4D supported only!", A.shape)

    return A / div


def padding(A, block_shape, overlap):
    """
    Pad A at the end so that block_shape will cut an integer number of blocks
    across all dimensions. A is padded with 0s.
    """

    block_shape = np.array(block_shape)
    overlap = np.array(overlap)
    fit = ((A.shape - block_shape) % (block_shape - overlap))
    fit = np.where(fit > 0, block_shape - overlap - fit, fit)

    if len(fit) > 3:
        fit[3:] = 0

    if np.sum(fit) == 0:
        return A

    padding = np.array(A.shape) + fit
    padded = np.zeros(padding, dtype=A.dtype, order='F')
    padded[:A.shape[0], :A.shape[1], :A.shape[2], ...] = A

    return padded
