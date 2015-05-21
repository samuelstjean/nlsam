# Copyright (c) 2008-2014, dipy developers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.

#     * Neither the name of the dipy developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import division, print_function, absolute_import
from os.path import splitext
from nlsam.six import string_types
import numpy as np


def read_bvals_bvecs(fbvals, fbvecs):
    """
    Read b-values and b-vectors from the disk

    Parameters
    ----------
    fbvals : str
             path of file with b-values, or None if you don't want to read bvals
    fbvecs : str
             path of file with b-vectorsl, or None if you don't want to read bvecs

    Returns
    -------
    bvals : array, (N,) or None
    bvecs : array, (N, 3) or None

    Notes
    -----
    Files can be either '.bvals'/'.bvecs' or '.txt' or '.npy' (containing arrays
    stored with the appropriate values).

    """

    # Loop over the provided inputs, reading each one in turn and adding them
    # to this list:
    vals = []
    for this_fname in [fbvals, fbvecs]:
        # If the input was None, we don't read anything and move on:
        if this_fname is None:
            vals.append(None)
        else:
            if isinstance(this_fname, string_types):
                base, ext = splitext(this_fname)
                if ext in ['.bvals', '.bval', '.bvecs', '.bvec', '.txt', '']:
                    vals.append(np.squeeze(np.loadtxt(this_fname)))
                elif ext == '.npy':
                    vals.append(np.squeeze(np.load(this_fname)))
                else:
                    e_s = "File type %s is not recognized"%ext
                    raise ValueError(e_s)
            else:
                raise ValueError('String with full path to file is required')

    # Once out of the loop, unpack them:
    bvals, bvecs = vals[0], vals[1]

    # If bvecs is None, you can just return now w/o making more checks:
    if bvecs is None:
        return bvals, bvecs

    if bvecs.shape[1] > bvecs.shape[0]:
        bvecs = bvecs.T
    if min(bvecs.shape) != 3:
        raise IOError('bvec file should have three rows')
    if bvecs.ndim != 2:
        raise IOError('bvec file should be saved as a two dimensional array')

    # If bvals is None, you don't need to check that they have the same shape:
    if bvals is None:
        return bvals, bvecs

    if max(bvals.shape) != max(bvecs.shape):
            raise IOError('b-values and b-vectors shapes do not correspond')

    return bvals, bvecs
