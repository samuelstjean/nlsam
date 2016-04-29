#cython: wraparound=False, cdivision=True, boundscheck=False

from __future__ import print_function

import numpy as np
cimport numpy as np
cimport cython

from numpy.lib.stride_tricks import as_strided as ast
from scipy.sparse import issparse

def sparse_dot(a, b, order='F', dense_output=True):
    if issparse(a) or issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)


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


def im2col_nd(A,  block_shape, overlap, order='F'):
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
    dtype = A.dtype
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

    return R.astype(dtype, copy=False)


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

    dtype = R.dtype
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

    return (A / div).astype(dtype, copy=False)


def padding(A, block_shape, overlap):
    """
    Pad A at the end so that block_shape will cut an integer number of blocks
    across all dimensions. A is padded with value (default : 0).
    """

    block_shape = np.array(block_shape)
    overlap = np.array(overlap)
    fit = ((A.shape - block_shape) % (block_shape - overlap))
    fit = np.where(fit > 0, block_shape - overlap - fit, fit)

    if len(fit) > 3:
        fit[3:] = 0

    if np.sum(fit) == 0:
        return A

    # print("Block size doesn't fit in the volume. \
    #       \nIt will be padded at the end with value", value)

    padding = np.array(A.shape) + fit
    padded = np.zeros(padding, dtype=A.dtype, order='F')
    padded[:A.shape[0], :A.shape[1], :A.shape[2], ...] = A

    return padded



# Stolen from http://www.johnvinyard.com/blog/?p=268
def sliding_window(a, ws, ss=None, flatten=True):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)


def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')
