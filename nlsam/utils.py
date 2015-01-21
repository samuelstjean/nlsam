# cython: wraparound=False, cdivision=True, boundscheck=False

from __future__ import division, print_function

import numpy as np

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


def _im2col(A, size, overlap, order):

    assert len(A.shape) == len(size), "number of dimensions mismatch!"

    size = np.array(size)
    overlap = np.array(overlap)
    dim = ((A.shape - size) / (size - overlap)) + 1

    R = np.zeros((np.prod(dim), np.prod(size)), dtype=A.dtype, order=order)
    k = 0
  #  print(A.shape,R.shape)
    for a in range(0, A.shape[0] - overlap[0], size[0] - overlap[0]):
        for b in range(0, A.shape[1] - overlap[1], size[1] - overlap[1]):
            for c in range(0, A.shape[2] - overlap[2], size[2] - overlap[2]):

                R[k, :] = A[a:a + size[0],
                            b:b + size[1],
                            c:c + size[2]].ravel()

                k += 1

    return R

# cimport numpy as np

# cdef double [:,:] _im2col3D(double [:,:,:] A, int[:] size, int[:] overlap, str order):

#     assert len(A.shape) == len(size), "number of dimensions mismatch!"

#     # size = np.array(size)
#     # overlap = np.array(overlap)
#     cdef:
#         int k = 0
#         int dim0 = np.prod((A.shape - size) / (size - overlap)) + 1
#         int dim1 = np.prod(size)

#     R = np.zeros((dim0, dim1), dtype=A.dtype, order=order)
#     # k = 0
#   #  print(A.shape,R.shape)
#     for a in range(0, A.shape[0] - overlap[0], size[0] - overlap[0]):
#         for b in range(0, A.shape[1] - overlap[1], size[1] - overlap[1]):
#             for c in range(0, A.shape[2] - overlap[2], size[2] - overlap[2]):

#                 R[k, :] = A[a:a + size[0],
#                             b:b + size[1],
#                             c:c + size[2]].ravel()

#                 k += 1

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

    block_shape = np.array(block_shape)
    overlap = np.array(overlap)

    if (overlap.any() < 0) or ((block_shape < overlap).any()):
        raise ValueError('Invalid overlap value, it must lie between 0' +
                         'and min(block_size)-1', overlap, block_shape)
    #print(A.shape, block_shape, overlap, 'B.C.')
    A = padding(A, block_shape, overlap)
    #print(A.shape, block_shape, overlap, 'A.D.')
    return _im2col(A, block_shape, overlap, order=order)

   # if len(block_shape) == 3:
    #    return _im2col_3d(A, block_shape, overlap, order=order)

    #elif len(block_shape) == 4:
     #   return _im2col_4d(A, block_shape, overlap, order=order)

    #raise ValueError("invalid type of window")


def _col2im(R, size, end_shape, overlap, weights, order):

    size = np.array(size)
    end_shape = np.array(end_shape)
    overlap = np.array(overlap)

    A = np.zeros(end_shape, dtype=R.dtype, order=order)
    div = np.zeros_like(A, dtype=R.dtype, order=order)
    k = 0
    #print(R.shape, A.shape)

    for a in range(0, A.shape[0] - overlap[0], size[0] - overlap[0]):
        for b in range(0, A.shape[1] - overlap[1], size[1] - overlap[1]):
            for c in range(0, A.shape[2] - overlap[2], size[2] - overlap[2]):
              #  print(R[:, k].shape, R.shape, size,  A[a:a + size[0], b:b + size[1], c:c + size[2]].shape)
               # print(a,b,c)
                A[a:a + size[0],
                  b:b + size[1],
                  c:c + size[2]] += R[:, k].reshape(size) * weights[k]  # np.take(R, [k], axis=1).reshape(size) #

                div[a:a + size[0],
                    b:b + size[1],
                    c:c + size[2]] += weights[k]

                k += 1


    #size = size[:-1]
    #for a in range(0, A.shape[0] - overlap[0], size[0] - overlap[0]):
    #    for b in range(0, A.shape[1] - overlap[1], size[1] - overlap[1]):
    #        for c in range(0, A.shape[2] - overlap[2], size[2] - overlap[2]):
    #            for d in range(A.shape[-1]):
    #               #  print(R[:, k].shape, R.shape, size,  A[a:a + size[0], b:b + size[1], c:c + size[2]].shape)
    #                # print(a,b,c)

    #                A[a:a + size[0],
    #                  b:b + size[1],
    #                  c:c + size[2], d] += R[:, k].reshape(size) * weights[k]  # np.take(R, [k], axis=1).reshape(size) #

    #                div[a:a + size[0],
    #                    b:b + size[1],
    #                    c:c + size[2], d] += weights[k]

    #                k += 1
    div[div==0] = 1
    return A / div


# def _col2im_4d(R, size, width, overlap, weights, order):

#     size = np.array(size)
#     width = np.array(width)
#     overlap = np.array(overlap)
#    # dim = ((width - size) / (size - overlap)) + 1

#     A = np.zeros(width, dtype=R.dtype, order=order)
#     div = np.zeros_like(A, dtype=R.dtype, order=order)
#     #ones = np.ones(size, dtype='int16')
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
    Returns a nd array of shape end_shape from a 2d array made of flatenned
    block that had originally a shape of block_shape.
    Inverse function of im2col_nd.
    """

    block_shape = np.array(block_shape)
    end_shape = np.array(end_shape)
    overlap = np.array(overlap)

    if (overlap.any() < 0) or ((block_shape < overlap).any()):
        raise ValueError('Invalid overlap value, it must lie between 0 \
                         \nand min(block_size)-1', overlap, block_shape)

    if weights is None:
        # print ("No weights specified, using ones")
        weights = np.ones(R.shape[1], dtype=R.dtype)

    return _col2im(R, block_shape, end_shape, overlap, weights, order)

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

# from dipy.denoise.denspeed import add_padding_reflection
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
    padded = np.zeros(padding, dtype=A.dtype) #* value

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


def PCA_truncate(data, block_size, overlap, threshold=10**-25):

    from sklearn.decomposition import SparsePCA

    orig_shape = data.shape
    #data, mask = median_otsu(data)

    data = im2col_nd(data, block_size, overlap).T
    mask = np.sum(im2col_nd(mask, block_size[:-1], overlap[:-1]).T, axis=0)
    truncated = np.zeros_like(data)

    data_mean = np.mean(data, axis=0, keepdims=True)
    data -= data_mean
    print(data.shape)
    for i in range(data.shape[1]):
        print(i)
        if mask[i]:
            sample = data[:, i]
            sigma = np.outer(sample, sample)/(sample.size)
            U, s, V = np.linalg.svd(sigma)
            #s[np.cumsum(s)/np.sum(s) > threshold] = 0
            #print(s)
            s[s < threshold] = 0
            ##print(np.var(sample))
            PCA = np.dot(np.dot(U, np.diag(s)), U.T)
            truncated[:, i] = np.ravel(np.dot(PCA, sample)) + data_mean[:, i]
            print("eigen values kept", np.count_nonzero(s), "out of", data.shape[0])
            #print(s[s>0])
            #print(s)
            #print(np.cumsum(s)/np.sum(s))
            SPCA = SparsePCA(n_jobs=8, verbose=True)

            a = SPCA.fit_transform(sample[None, :].T)
            print(SPCA.components_.shape, a.shape)
            #SPCA.n_components = np.sum(SPCA.explained_variance_ratio) + data_mean[:, i]
            truncated[:, i] = np.dot(SPCA.components_, a).T + data_mean[:, i]
        else:
            truncated[:, i] = np.zeros(data.shape[0])

    truncated = col2im_nd(truncated, block_size, orig_shape, overlap)
    truncated = truncated[:orig_shape[0], :orig_shape[1],
                          :orig_shape[2], :orig_shape[3]]

    return truncated


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

def test_reshaping():
    from numpy.testing import assert_almost_equal

    a = np.random.randint(0, 10000, size=(30, 30, 30))
    assert_almost_equal(a, col2im_nd(im2col_nd(a, (2,2,2), (1,1,1)).T, (2,2,2), (30,30,30), (1,1,1)))
    assert_almost_equal(a, col2im_nd(im2col_nd(a, (3,3,3), (2,2,2)).T, (3,3,3), (30,30,30), (2,2,2)))

    b = np.random.randint(0, 10000, size=(30, 30, 30, 10))
    assert_almost_equal(b, col2im_nd(im2col_nd(b, (2,2,2,10), (1,1,1,1)).T, (2,2,2,10), (30,30,30,10), (1,1,1,1)))
    assert_almost_equal(b, col2im_nd(im2col_nd(b, (3,3,3,10), (2,2,2,1)).T, (3,3,3,10), (30,30,30,10), (2,2,2,1)))
