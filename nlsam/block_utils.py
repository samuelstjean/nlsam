from __future__ import division

import numpy as np
import numbers

from itertools import product
from numpy.lib.stride_tricks import as_strided


def im2col_nd(A, block_shape, overlap, reshape_2D=True):

    if isinstance(block_shape, numbers.Real):
        block_shape = [block_shape] * A.ndim

    if isinstance(overlap, numbers.Real):
        overlap = [overlap] * A.ndim

    overlap = np.array(overlap)
    block_shape = np.array(block_shape)
    # print(overlap)
    # overlap[3:] = 0
    # block_shape[3:] = A.shape[3:]

    if len(block_shape) > A.ndim:
        block_shape = block_shape[:A.ndim]
        overlap = overlap[:A.ndim]
    # print(overlap)
    with np.errstate(divide='ignore'):
        fit = (np.array(A.shape) - block_shape) % (block_shape - overlap)
        fit = np.where(fit > 0, block_shape - overlap - fit, fit)

    if len(fit) > 3:
        fit[3:] = 0

    if fit.sum() > 0:
        padding = [(0, f) for f in fit]
        A = np.pad(A, padding, 'constant', constant_values=0)

    extraction_step = block_shape - overlap
    mask = extraction_step == 0
    extraction_step[mask] = block_shape[mask]

    A_patch = extract_patches(A, block_shape, extraction_step)
    print(A.shape, A_patch.shape, block_shape, overlap, extraction_step)
    if reshape_2D:
        return A_patch.reshape(-1, np.prod(block_shape)).T
    return A_patch


def col2im_nd(R, block_shape, end_shape, overlap, mask=None, weights=None):

    if weights is None:
        weights = np.ones(R.shape[1], dtype=np.float32)

    if isinstance(weights, numbers.Real):
        weights = np.ones(R.shape[1], dtype=np.float32) * weights

    if mask is None:
        mask = np.ones(R.shape[1], dtype=np.bool)

    out = reconstruct_from_indexes(R, block_shape, end_shape, overlap, mask, weights)
    return out


def reconstruct_from_indexes(X, block_size, shape, overlap, mask, weights):

    i_h, i_w, i_l = shape[:3]
    p_h, p_w, p_l = block_size[:3]

    img = np.zeros(shape, dtype=np.float32)
    div = np.ones(shape, dtype=np.float32)

    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    n_l = i_l - p_l + 1

    skip = np.array(block_size) - np.array(overlap)
    print(skip, overlap, block_size)
    step = ([slice(i, i + p_h) for i in range(0, n_h, skip[0])],
            [slice(j, j + p_w) for j in range(0, n_w, skip[1])],
            [slice(k, k + p_l) for k in range(0, n_l, skip[2])])

    ijk = product(*step)
    # ijk = product(range(n_h), range(n_w), range(n_l))
    # print(X.shape, block_size, shape, overlap, mask.shape, mask.sum(), weights.shape)
    p = 0
    print(i_h, i_w, i_l, p_h, p_w, p_l, n_h, n_w, n_l)
    print(X.shape, mask.shape)
    # print(len([slice(i, i + p_h) for i in range(0, n_h, skip[0])]), len([slice(j, j + p_w) for j in range(0, n_w, skip[1])]), len([slice(k, k + p_l) for k in range(0, n_l, skip[2])]), len(list(ijk)))
    for idx, (i, j, k) in enumerate(ijk):
        # print(idx, i,j,k,p, mask.shape, img.shape, div.shape, X.shape, weights.shape, block_size)
        if mask[idx]:
            img[i, j, k] = np.reshape(X[:, p] * weights[p], block_size)
            div[i, j, k] = weights[p]
            p += 1

    out = img / div
    return out


# Stolen from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/image.py

# New BSD License
# Copyright (c) 2007–2018 The scikit-learn developers.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of the Scikit-learn Developers  nor the names of
#      its contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

def extract_patches(arr, patch_shape, extraction_step):
    """Extracts patches of any n-dimensional array in place using strides.
    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted
    patch_shape : integer or tuple of length arr.ndim
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.
    extraction_step : integer or tuple of length arr.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
    """

    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = [slice(None, None, st) for st in extraction_step]
    indexing_strides = arr[slices].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    return patches
