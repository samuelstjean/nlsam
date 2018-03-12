#! /usr/bin/env python

from __future__ import division, print_function

import numpy as np
from numpy.testing import assert_allclose, run_module_suite

from nlsam.block_utils import col2im_nd, im2col_nd, extract_patches
# from nlsam.utils import col2im_nd


def test_reshaping():

    a = np.random.randint(0, 10000, size=(30, 30, 30))
    assert_allclose(a, col2im_nd(im2col_nd(a, (2, 2, 2), (1, 1, 1)), (2, 2, 2), (30, 30, 30), (1, 1, 1)), rtol=1e-5)
    assert_allclose(a, col2im_nd(im2col_nd(a, (3, 3, 3), (2, 2, 2)), (3, 3, 3), (30, 30, 30), (2, 2, 2)), rtol=1e-5)

    b = np.random.randint(0, 10000, size=(30, 30, 30, 10))
    assert_allclose(b, col2im_nd(im2col_nd(b, (2, 2, 2, 10), (1, 1, 1, 1)), (2, 2, 2, 10), (30, 30, 30, 10), (1, 1, 1, 1)), rtol=1e-5)
    assert_allclose(b, col2im_nd(im2col_nd(b, (3, 3, 3, 10), (2, 2, 2, 1)), (3, 3, 3, 10), (30, 30, 30, 10), (2, 2, 2, 1)), rtol=1e-5)

    a = np.random.rand(1000).reshape(10, 10, 10)
    out = im2col_nd(a, (3, 3, 3), (2, 2, 2))
    redo = col2im_nd(out, (3, 3, 3), (10, 10, 10), (2, 2, 2))
    assert_allclose(a, redo, rtol=1e-5)

    out = im2col_nd(a, (3, 3, 3), (1, 1, 1))
    redo = col2im_nd(out, (3, 3, 3), (10, 10, 10), (1, 1, 1))
    assert_allclose(a, redo, rtol=1e-5)

    a = np.random.rand(100000).reshape(10, 100, 10, 10)
    out = im2col_nd(a, (3, 3, 3, 10), (1, 1, 1, 10))
    redo = col2im_nd(out, (3, 3, 3, 10), (10, 100, 10, 10), (1, 1, 1, 0))
    assert_allclose(a, redo, rtol=1e-5)

    # out = im2col_nd(a, (3, 3, 3, 10), (2, 2, 2, 0))
    # redo = col2im_nd(out, (3, 3, 3, 10), (10, 100, 10, 10), (2, 2, 2, 0))
    # assert_allclose(a, redo, rtol=1e-5)

    # a = np.random.rand(1000).reshape(10, 10, 10)
    # out = im2col_nd(a, (2, 2, 2), (0, 0, 0))
    # redo = col2im_nd(out, (2, 2, 2), (10, 10, 10), (0, 0, 0))
    # assert_allclose(a, redo, rtol=1e-5)

    # out = im2col_nd(a, (2, 2, 2), reshape_2D=False).reshape(-1, 2**3).T
    # redo = col2im_nd(out, (2, 2, 2), (10, 10, 10), (0, 0, 0))
    # assert_allclose(a, redo, rtol=1e-5)

    # a = np.random.rand(10000).reshape(10, 10, 10, 10)
    # out = im2col_nd(a, (2, 2, 2, 10), (0, 0, 0, 0))
    # redo = col2im_nd(out, (2, 2, 2, 10), (10, 10, 10, 10), (0, 0, 0, 0))
    # assert_allclose(a, redo, rtol=1e-5)

    out = im2col_nd(a, (2, 2, 2, 10), (1, 1, 1, 10))
    redo = col2im_nd(out, (2, 2, 2, 10), (10, 10, 10, 10), (1, 1, 1, 1))
    assert_allclose(a, redo, rtol=1e-5)


def test_extract_patches_strided():

    image_shapes_1D = [(10,), (10,), (11,), (10,)]
    patch_sizes_1D = [(1,), (2,), (3,), (8,)]
    patch_steps_1D = [(1,), (1,), (4,), (2,)]

    expected_views_1D = [(10,), (9,), (3,), (2,)]
    last_patch_1D = [(10,), (8,), (8,), (2,)]

    image_shapes_2D = [(10, 20), (10, 20), (10, 20), (11, 20)]
    patch_sizes_2D = [(2, 2), (10, 10), (10, 11), (6, 6)]
    patch_steps_2D = [(5, 5), (3, 10), (3, 4), (4, 2)]

    expected_views_2D = [(2, 4), (1, 2), (1, 3), (2, 8)]
    last_patch_2D = [(5, 15), (0, 10), (0, 8), (4, 14)]

    image_shapes_3D = [(5, 4, 3), (3, 3, 3), (7, 8, 9), (7, 8, 9)]
    patch_sizes_3D = [(2, 2, 3), (2, 2, 2), (1, 7, 3), (1, 3, 3)]
    patch_steps_3D = [(1, 2, 10), (1, 1, 1), (2, 1, 3), (3, 3, 4)]

    expected_views_3D = [(4, 2, 1), (2, 2, 2), (4, 2, 3), (3, 2, 2)]
    last_patch_3D = [(3, 2, 0), (1, 1, 1), (6, 1, 6), (6, 3, 4)]

    image_shapes = image_shapes_1D + image_shapes_2D + image_shapes_3D
    patch_sizes = patch_sizes_1D + patch_sizes_2D + patch_sizes_3D
    patch_steps = patch_steps_1D + patch_steps_2D + patch_steps_3D
    expected_views = expected_views_1D + expected_views_2D + expected_views_3D
    last_patches = last_patch_1D + last_patch_2D + last_patch_3D

    for (image_shape, patch_size, patch_step, expected_view,
         last_patch) in zip(image_shapes, patch_sizes, patch_steps,
                            expected_views, last_patches):
        image = np.arange(np.prod(image_shape)).reshape(image_shape)
        patches = extract_patches(image, patch_shape=patch_size,
                                  extraction_step=patch_step)

        ndim = len(image_shape)

        np.testing.assert_(patches.shape[:ndim] == expected_view)
        last_patch_slices = [slice(i, i + j, None) for i, j in
                             zip(last_patch, patch_size)]
        np.testing.assert_((patches[[slice(-1, None, None)] * ndim] ==
                            image[last_patch_slices].squeeze()).all())


if __name__ == "__main__":
    run_module_suite()
