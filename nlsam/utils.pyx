# cython: wraparound=False, cdivision=True, boundscheck=False

from __future__ import print_function
import numpy as np
cimport numpy as np
cimport cython

from scipy.sparse import csc_matrix, issparse, lil_matrix

def sparse_dot(a, b, order='F', dense_output=True):
    # if issparse(A) is False:
    #     A = lil_matrix(A).tocsc()

    # if issparse(B) is False:
    #     B = lil_matrix(B).tocsc()

    # return (A * B).toarray(order=order)
    if issparse(a) or issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)


# def _im2col(A, size, overlap, order):

#     assert len(A.shape) == len(size), "number of dimensions mismatch!"

#     size = np.array(size)
#     overlap = np.array(overlap)
#     dim = ((A.shape - size) / (size - overlap)) + 1

#     R = np.zeros((np.prod(dim), np.prod(size)), dtype=A.dtype, order=order)
#     k = 0
#   #  print(A.shape,R.shape)
#     for a in range(0, A.shape[0] - overlap[0], size[0] - overlap[0]):
#         for b in range(0, A.shape[1] - overlap[1], size[1] - overlap[1]):
#             for c in range(0, A.shape[2] - overlap[2], size[2] - overlap[2]):

#                 R[k, :] = A[a:a + size[0],
#                             b:b + size[1],
#                             c:c + size[2]].ravel()

#                 k += 1

#     return R


cdef void _im2col3D(double[:,:,:] A, double[:,:] R, int[:] size) nogil:

    cdef:
        int k = 0, l = 0
        int a, b, c, m, n, o
        int x = A.shape[0], y = A.shape[1], z =  A.shape[2]
        int s0 = size[0], s1 = size[1], s2 = size[2]
        # int dim0 = np.prod((A.shape - size)) + 1
        # int dim1 = np.prod(size)
        # double [:,:] R = np.zeros((dim0, dim1), dtype=np.float64, order='F')
    # with gil:
    #     print(x - s0 + 1, y - s1 + 1, z - s2 + 1)
    #     print(x, y, z)
    #     print(R.shape[0], R.shape[1])
    # 1/0
    with nogil:
        for a in range(x - s0 + 1):
            for b in range(y - s1 + 1):
                for c in range(z - s2 + 1):
                    # with gil:
                    #     print(a,b,c)
                    l = 0

                    for m in range(s0):
                        for n in range(s1):
                            for o in range(s2):
                                R[k, l] = A[a+m, b+n, c+o]
                                l += 1
                                # with gil:
                                    # print(m,n,o,k,l)

                    k += 1


cdef void _im2col3D_overlap(double[:,:,:] A, double[:,:] R, int[:] size, int[:] overlap):

    cdef:
        int k = 0, l = 0
        int a, b, c, m, n, o
        int x = A.shape[0], y = A.shape[1], z =  A.shape[2]
        int s0 = size[0], s1 = size[1], s2 = size[2]
        int over0 = overlap[0], over1 = overlap[1], over2 = overlap[2]
        # int dim0 = np.prod((A.shape - size)) + 1
        # int dim1 = np.prod(size)
        # double [:,:] R = np.zeros((dim0, dim1), dtype=np.float64, order='F')
    # with gil:
    #     print(x - s0 + 1, y - s1 + 1, z - s2 + 1)
    #     print(x, y, z)
    #     print(R.shape[0], R.shape[1])
    # 1/0
    # with nogil:
    for a in range(0, x - over0, s0 - over0):
        for b in range(0, y - over1, s1 - over1):
            for c in range(0, z - over2, s2 - over2):
                # with gil:
                #     print(a,b,c)
                with nogil:
                    l = 0

                    for m in range(s0):
                        for n in range(s1):
                            for o in range(s2):
                                R[k, l] = A[a+m, b+n, c+o]
                                l += 1
                                # with gil:
                                    # print(m,n,o,k,l)

                    k += 1


cdef void _im2col4D(double[:,:,:,:] A, double[:,:] R, int[:] size) nogil:

    cdef:
        int k = 0, l = 0
        int a, b, c, d, m, n, o, p
        int x = A.shape[0], y = A.shape[1], z = A.shape[2], t = A.shape[3]
        int s0 = size[0], s1 = size[1], s2 = size[2] #, s3 = size[3]
        # int dim0 = np.prod((A.shape - size)) + 1
        # int dim1 = np.prod(size)
        # int d0 = R[0].shape
        # int d1 = R[1].shape
        # double [:,:] R = np.zeros((dim0, dim1), dtype=np.float64, order='F')

    with nogil:
        for a in range(x - s0 + 1):
            for b in range(y - s1 + 1):
                for c in range(z - s2 + 1):

                    l = 0
                    # with gil:
                    #     print(a,b,c)
                    for m in range(s0):
                        for n in range(s1):
                            for o in range(s2):
                                for p in range(t):

                                    R[k, l] = A[a+m, b+n, c+o, p]
                                    l += 1
                                    # with gil:
                                    #     print(m,n,o,p,k,l)

                    k += 1

#     return R

#    R = np.zeros((np.prod(dim[:-1])*size[-1], np.prod(size[:-1])), dtype=A.dtype, order=order)
#    k = 0
#    print(A.shape,R.shape)

#    for a in range(0, A.shape[0] - overlap[0], size[0] - overlap[0]):
#        for b in range(0, A.shape[1] - overlap[1], size[1] - overlap[1]):
#            for c in range(0, A.shape[2] - overlap[2], size[2] - overlap[2]):
#                for d in range(A.shape[-1]):

#                    R[k, :] = A[a:a + size[0],
#                                b:b + size[1],
#                                c:c + size[2],d].ravel()

#                    k += 1


# def _im2col_4d(A, size, overlap, order):

#     assert len(A.shape) == len(size), "number of dimensions mismatch!"

#     size = np.array(size)
#     overlap = np.array(overlap)
#     dim = ((A.shape - size) / (size - overlap)) + 1

#     R = np.zeros((np.prod(dim), np.prod(size)), dtype=A.dtype, order=order)
#     k = 0
#     #print(A.shape, size, overlap)
#     for a in range(0, A.shape[0]-overlap[0], size[0]-overlap[0]):
#         for b in range(0, A.shape[1]-overlap[1], size[1]-overlap[1]):
#             for c in range(0, A.shape[2]-overlap[2], size[2]-overlap[2]):
#                # for d in range(0, A.shape[3]-overlap[3], size[3]-overlap[3]):

#                     R[k, :] = A[a:a + size[0],
#                                 b:b + size[1],
#                                 c:c + size[2]].ravel() #,
#                                # d:d + size[3]].ravel()
#                     k += 1

#     return R


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
    #print(A.shape, block_shape, overlap, 'B.C.')
    A = padding(A, block_shape, overlap)
    dtype = A.dtype
    A = A.astype(np.float64)

    if len(A.shape) != len(block_shape):
        raise ValueError("Number of dimensions mismatch!", A.shape, block_shape)
    #print(A.shape, block_shape, overlap, 'A.D.')
    dim0 = np.prod(A.shape - block_shape + 1)
    dim1 = np.prod(block_shape)
    R = np.zeros((dim0, dim1), dtype=np.float64, order='F')
    # print("in", R.shape, A.shape, A.dtype, R.dtype)

    if len(A.shape) == 3:
        # with nogil:
        if np.sum(block_shape - overlap) > len(A.shape):
            # print("overlap", overlap)
            _im2col3D_overlap(A, R, block_shape, overlap)
        else:
            _im2col3D(A, R, block_shape)
            # print("no overlap")
    elif len(A.shape) == 4:
        # with nogil:
            _im2col4D(A, R, block_shape)
    else:
        raise ValueError("3D or 4D supported only!", A.shape)

    return R.astype(dtype)
    # return _im2col(A, block_shape, overlap, order=order)

   # if len(block_shape) == 3:
    #    return _im2col_3d(A, block_shape, overlap, order=order)

    #elif len(block_shape) == 4:
     #   return _im2col_4d(A, block_shape, overlap, order=order)

    #raise ValueError("invalid type of window")


# def _col2im(R, size, end_shape, overlap, weights, order):

#     size = np.array(size, dtype=np.int32)
#     end_shape = np.array(end_shape, dtype=np.int32)
#     overlap = np.array(overlap, dtype=np.int32)

#     A = np.zeros(end_shape, dtype=R.dtype, order=order)
#     div = np.zeros_like(A, dtype=R.dtype, order=order)
#     k = 0
#     #print(R.shape, A.shape)

#     for a in range(0, A.shape[0] - overlap[0], size[0] - overlap[0]):
#         for b in range(0, A.shape[1] - overlap[1], size[1] - overlap[1]):
#             for c in range(0, A.shape[2] - overlap[2], size[2] - overlap[2]):
#               #  print(R[:, k].shape, R.shape, size,  A[a:a + size[0], b:b + size[1], c:c + size[2]].shape)
#                # print(a,b,c)
#                 A[a:a + size[0],
#                   b:b + size[1],
#                   c:c + size[2]] += R[:, k].reshape(size) * weights[k]  # np.take(R, [k], axis=1).reshape(size) #

#                 div[a:a + size[0],
#                     b:b + size[1],
#                     c:c + size[2]] += weights[k]

#                 k += 1


#     #size = size[:-1]
#     #for a in range(0, A.shape[0] - overlap[0], size[0] - overlap[0]):
#     #    for b in range(0, A.shape[1] - overlap[1], size[1] - overlap[1]):
#     #        for c in range(0, A.shape[2] - overlap[2], size[2] - overlap[2]):
#     #            for d in range(A.shape[-1]):
#     #               #  print(R[:, k].shape, R.shape, size,  A[a:a + size[0], b:b + size[1], c:c + size[2]].shape)
#     #                # print(a,b,c)

#     #                A[a:a + size[0],
#     #                  b:b + size[1],
#     #                  c:c + size[2], d] += R[:, k].reshape(size) * weights[k]  # np.take(R, [k], axis=1).reshape(size) #

#     #                div[a:a + size[0],
#     #                    b:b + size[1],
#     #                    c:c + size[2], d] += weights[k]

#     #                k += 1
#     div[div==0] = 1
#     return A / div


cdef void _col2im3D_overlap(double[:,:,:] A, double[:,:,:] div, double[:,:] R, double[:] weights, int[:] block_shape, int[:] overlap):

    # size = np.array(size)
    # width = np.array(width)
    # overlap = np.array(overlap)
   # dim = ((width - size) / (size - overlap)) + 1

    # A = np.zeros(width, dtype=R.dtype, order=order)
    # div = np.zeros_like(A, dtype=R.dtype, order=order)
    #ones = np.ones(size, dtype='int32')
  #  ones = np.ones(A.shape[-1], dtype=R.dtype, order=order)
    cdef:
        int k = 0, l = 0
        int a, b, c, m, n, o
        int x = A.shape[0], y = A.shape[1], z =  A.shape[2]
        int s0 = block_shape[0], s1 = block_shape[1], s2 = block_shape[2]
        int over0 = overlap[0], over1 = overlap[1], over2 = overlap[2]


    # with gil:
    #    print("col2im4d")
       # print(R.shape, weights.shape)
    # with nogil:
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


cdef void _col2im3D(double[:,:,:] A, double[:,:,:] div, double[:,:] R, double[:] weights, int[:] block_shape) nogil:

    # size = np.array(size)
    # width = np.array(width)
    # overlap = np.array(overlap)
   # dim = ((width - size) / (size - overlap)) + 1

    # A = np.zeros(width, dtype=R.dtype, order=order)
    # div = np.zeros_like(A, dtype=R.dtype, order=order)
    #ones = np.ones(size, dtype='int32')
  #  ones = np.ones(A.shape[-1], dtype=R.dtype, order=order)
    cdef:
        int k = 0, l = 0
        int a, b, c, m, n, o
        int x = A.shape[0], y = A.shape[1], z =  A.shape[2]
        int s0 = block_shape[0], s1 = block_shape[1], s2 = block_shape[2]

    # with gil:
    #    print("col2im4d")
       # print(R.shape, weights.shape)
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
    # A /= div
    # return A / div


cdef void _col2im4D(double[:,:,:,:] A, double[:,:,:] div, double[:,:] R, double[:] weights, int[:] block_shape) nogil:

    # size = np.array(size)
    # width = np.array(width)
    # overlap = np.array(overlap)
   # dim = ((width - size) / (size - overlap)) + 1

    # A = np.zeros(width, dtype=R.dtype, order=order)
    # div = np.zeros_like(A, dtype=R.dtype, order=order)
    #ones = np.ones(size, dtype='int32')
  #  ones = np.ones(A.shape[-1], dtype=R.dtype, order=order)
    cdef:
        int k = 0, l = 0
        int a, b, c, d, m, n, o, p
        int x = A.shape[0], y = A.shape[1], z = A.shape[2], t = A.shape[3]
        int s0 = block_shape[0], s1 = block_shape[1], s2 = block_shape[2]
        # int gh = weights.shape[0]

 #   print("col2im4d", block_shape, width, overlap, A.shape)
  #  print(R.shape, weights.shape)
    with nogil:
        for a in range(x - s0 + 1):
            for b in range(y - s1 + 1):
                for c in range(z - s2 + 1):

                    l = 0
                    # with gil:
                    #     print(a,b,c,x,y,z,s0,s1,s2, gh)
                    for m in range(s0):
                        for n in range(s1):
                            for o in range(s2):
                                for p in range(t):
                                    # with gil:
                                    #     print(m,n,o,p,k,l, s0,s1,s2)
                                    A[a+m, b+n, c+o, p] += R[l, k] * weights[k]
                                    l += 1

                                div[a+m, b+n, c+o] += weights[k]
                    k += 1
    # A /= div


# def _col2im4D_old(R, size, width, overlap, weights, order):

#     size = np.array(size)
#     width = np.array(width)
#     overlap = np.array(overlap)
#    # dim = ((width - size) / (size - overlap)) + 1

#     A = np.zeros(width, dtype=R.dtype, order=order)
#     div = np.zeros_like(A, dtype=R.dtype, order=order)
#     #ones = np.ones(size, dtype='int32')
#   #  ones = np.ones(A.shape[-1], dtype=R.dtype, order=order)
#     k = 0
#  #   print("col2im4d", size, width, overlap, A.shape)
#   #  print(R.shape, weights.shape)

#     for a in range(0, A.shape[0]-overlap[0], size[0]-overlap[0]):
#         for b in range(0, A.shape[1]-overlap[1], size[1]-overlap[1]):
#             for c in range(0, A.shape[2]-overlap[2], size[2]-overlap[2]):
#              #   for d in range(0, A.shape[3]-overlap[3], size[3]-overlap[3]):

#                     A[a:a + size[0],
#                       b:b + size[1],
#                       c:c + size[2]] += R[:, k].reshape(size) #,
#                  #     d:d + size[3]] += (R[:, k]).reshape(size) ###* weights[:, k]).reshape(size)

#                     div[a:a + size[0],
#                         b:b + size[1],
#                         c:c + size[2]] += 1 #,
#                       #  d:d + size[3]] += 1

#                     k += 1

#     return A / div


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
        # print ("No weights specified, using ones")
        weights = np.ones(R.shape[1], dtype=np.float64)


    # return _col2im(R, block_shape, end_shape, overlap, weights, order)
    dtype = R.dtype
    R = R.astype(np.float64)
    A = np.zeros(end_shape, dtype=np.float64, order=order)
    div = np.zeros(end_shape[:3], dtype=np.float64, order=order)
    # print("in", R.shape, A.shape, A.dtype, R.dtype)
    # print(A.shape, div.shape, R.shape, weights.shape, block_shape.shape)
    # 1/0

    if len(A.shape) == 3:
        # print(A.dtype,R.dtype,block_shape.dtype)
        # with nogil:

        block_shape = block_shape[:3]
        overlap = overlap[:3]

        if np.sum(block_shape - overlap) > len(A.shape):
            # print("overlap", overlap)
            _col2im3D_overlap(A, div, R, weights, block_shape, overlap)
        else:
            _col2im3D(A, div, R, weights, block_shape)
            # print("no overlap")
    elif len(A.shape) == 4:
        # with nogil:
        # print(R.shape)
        _col2im4D(A, div, R, weights, block_shape)
        div = div[..., None]
    else:
        raise ValueError("3D or 4D supported only!", A.shape)

    return (A / div).astype(dtype)

   # if len(block_shape) == 3:
    #    return _col2im_3d(R, block_shape, end_shape, overlap, weights, order)

    #elif len(block_shape) == 4:
     #   return _col2im_4d(R, block_shape, end_shape, overlap, weights, order)


# def apply_mask(image, mask):
#     """
#     Applies a binary mask to an image file. The image and the mask need
#     to be of the same 3D dimension. The mask will be applied on all 4th
#     dimension images.
#     """

#     if (len(image.shape) - len(mask.shape)) == 1:
#             return image * np.squeeze(np.tile(mask[..., None], image.shape[-1]))

#     return image * mask


def padding(A, block_shape, overlap, value=0):
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

    print("Block size doesn't fit in the volume. \
          \nIt will be padded at the end with value", value)

    padding = np.array(A.shape) + fit
    padded = np.zeros(padding, dtype=A.dtype, order='F') #* value

    padded[:A.shape[0], :A.shape[1], :A.shape[2], ...] = A
    # if A.ndim == 3:
    #     return np.pad(A, ((0, fit[0]), (0, fit[1]), (0, fit[2])), mode='reflect')
    # else:
    #     return np.pad(A, ((0, fit[0]), (0, fit[1]), (0, fit[2]), (0, 0)), mode='reflect')

    # padded[-1:-padding[0], -1:-padding[1], -1:-padding[2], ...] = value

    return padded


def ZCA_whitening(data, threshold=10**-5, eps=10**-5):
    """
    Applies the ZCA whitening transformation to data.

    Parameters
    -----------
    data : 2d array
        The data to be whitened, with each training examples in colums and
        each row representing a variable

    threshold : float
        All singular values < threshold will be set to zero in the SVD.
        Use this to perform truncation of the eigen values to remove noise.
        (Default : 10**-5)

    eps : float
        Regularization parameter used in the normalization. This is to prevent
        possible division by zero and should be set to a really low value in
        regard to the computed singular values. (Default : 10**-5)

    Return
    --------
    (data_whitened, ZCA, ZCA_inverse, data_mean) : tuple of 2d arrays

    data_whitened is the result of the ZCA transformation.

    ZCA is the ZCA transformation matrix, in order to apply it to another
    dataset. The dataset must be centered (substract the mean of each columns
    to itself)

    ZCA_inverse is the left inverse of the ZCA transform. Use it to get back
    the original data, e.g. data = np.dot(ZCA_inverse, data_whitened)+data_mean

    data_mean is the mean of each column of data. Since the transform works on
    centered data, it must be substracted and added back.

    Notes
    -------
    See http://ufldl.stanford.edu/wiki/index.php/Implementing_PCA/Whitening
    and http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    p.5 and Appendix A for more information."""

    data_mean = np.mean(data, axis=0, keepdims=True)
    data -= data_mean
    print(data.shape)
    sigma = np.dot(data, data.T)/(data.shape[1])
    U, s, V = np.linalg.svd(sigma)

    ZCA = np.dot(np.dot(U, np.diag(1./np.sqrt(s + eps))), U.T)

    data_whitened = np.dot(ZCA, data)
    #print("eigen values", s, np.count_nonzero(s<0.1))
    return (data_whitened, ZCA, np.linalg.inv(ZCA), data_mean)


# def PCA_truncate(data, mask, block_size, overlap, threshold=10**-25):

#     from sklearn.decomposition import SparsePCA

#     orig_shape = data.shape
#     #data, mask = median_otsu(data)

#     data = im2col_nd(data, block_size, overlap).T
#     mask = np.sum(im2col_nd(mask, block_size[:-1], overlap[:-1]).T, axis=0)
#     truncated = np.zeros_like(data)

#     data_mean = np.mean(data, axis=0, keepdims=True)
#     data -= data_mean
#     print(data.shape)
#     for i in range(data.shape[1]):
#         print(i)
#         if mask[i]:
#             sample = data[:, i]
#             sigma = np.outer(sample, sample)/(sample.size)
#             U, s, V = np.linalg.svd(sigma)
#             #s[np.cumsum(s)/np.sum(s) > threshold] = 0
#             #print(s)
#             s[s < threshold] = 0
#             ##print(np.var(sample))
#             PCA = np.dot(np.dot(U, np.diag(s)), U.T)
#             truncated[:, i] = np.ravel(np.dot(PCA, sample)) + data_mean[:, i]
#             print("eigen values kept", np.count_nonzero(s), "out of", data.shape[0])
#             #print(s[s>0])
#             #print(s)
#             #print(np.cumsum(s)/np.sum(s))
#             SPCA = SparsePCA(n_jobs=8, verbose=True)

#             a = SPCA.fit_transform(sample[None, :].T)
#             print(SPCA.components_.shape, a.shape)
#             #SPCA.n_components = np.sum(SPCA.explained_variance_ratio) + data_mean[:, i]
#             truncated[:, i] = np.dot(SPCA.components_, a).T + data_mean[:, i]
#         else:
#             truncated[:, i] = np.zeros(data.shape[0])

#     truncated = col2im_nd(truncated, block_size, orig_shape, overlap)
#     truncated = truncated[:orig_shape[0], :orig_shape[1],
#                           :orig_shape[2], :orig_shape[3]]

#     return truncated


# def sigma_correction(SNR):
#     """
#     Estimates the standard deviation of the noise based on a rician correction
#     scheme. The correction to be applied is
#     sigma_corrected**2 = sigma_noise**2 / chi_SNR
#     Don't forget to take the square root when needed.

#     Input
#     ------
#     SNR : float
#     It should be computed as
#     mean(signal)/mode(std(noise)), where signal is the region of interest and
#     noise is an estimate of the background noise. The mode is a more robust way
#     to estimate the SNR, for example by taking all ofthe background noise and
#     computing the local standard deviation on blocks of 3x3x3 voxels,
#     then taking the mode of the standard deviations.

#     Output
#     -------
#     sigma_corrected : The correction factor for the standard deviation
#     of the noise.

#     Notes
#     --------
#     For more information, see
#     Koay, C. G., & Basser, P. J. (2006).
#     Analytically exact correction scheme for signal extraction from noisy
#     magnitude MR signals. Journal of magnetic resonance
#     (San Diego, Calif. : 1997), 179(2), 317-22. doi:10.1016/j.jmr.2006.01.016

#     It is also used in
#     Manjon, J. V, Coupe, P., Concha, L., Buades, A., Collins, D. L., & Robles,
#     M. (2013). Diffusion Weighted Image Denoising Using Overcomplete Local PCA.
#     PloS one, 8(9), e73021. doi:10.1371/journal.pone.0073021
#     """

#     return 2 + SNR**2 - np.pi/8 * np.exp(-SNR**2/2) * \
#         ((2 + SNR**2) * i0(SNR**2/4) + SNR**2 * i1(SNR**2/4))**2
