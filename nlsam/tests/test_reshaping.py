#! /usr/bin/env python

from __future__ import division, print_function

import numpy as np
from numpy.testing import assert_almost_equal

from nlsam.utils import col2im_nd, im2col_nd


def test_reshaping():

    a = np.random.randint(0, 10000, size=(30, 30, 30))
    assert_almost_equal(a, col2im_nd(im2col_nd(a, (2,2,2), (1,1,1)).T, (2,2,2), (30,30,30), (1,1,1)))
    assert_almost_equal(a, col2im_nd(im2col_nd(a, (3,3,3), (2,2,2)).T, (3,3,3), (30,30,30), (2,2,2)))

    b = np.random.randint(0, 10000, size=(30, 30, 30, 10))
    assert_almost_equal(b, col2im_nd(im2col_nd(b, (2,2,2,10), (1,1,1,1)).T, (2,2,2,10), (30,30,30,10), (1,1,1,1)))
    assert_almost_equal(b, col2im_nd(im2col_nd(b, (3,3,3,10), (2,2,2,1)).T, (3,3,3,10), (30,30,30,10), (2,2,2,1)))

    a = np.random.rand(1000).reshape(10,10,10)
    out = im2col_nd(a, (3,3,3),(2,2,2)).T
    redo = col2im_nd(out, (3,3,3), (10,10,10), (2,2,2))
    np.testing.assert_allclose(a,redo)

    a = np.random.rand(100000).reshape(10,100,10,10)
    out = im2col_nd(a, (3,3,3,10),(2,2,2,0)).T
    redo = col2im_nd(out, (3,3,3,10), (10,100,10,10), (2,2,2,0))
    np.testing.assert_allclose(a,redo)

    a = np.random.rand(1000).reshape(10,10,10)
    out = im2col_nd(a, (2,2,2),(0,0,0)).T
    redo = col2im_nd(out, (2,2,2), (10,10,10), (0,0,0))
    np.testing.assert_allclose(a,redo)

    a = np.random.rand(10000).reshape(10,10,10,10)
    out = im2col_nd(a, (2,2,2,10),(0,0,0,0)).T
    redo = col2im_nd(out, (2,2,2,10), (10,10,10,10), (0,0,0,0))
    np.testing.assert_allclose(a,redo)


test_reshaping()
