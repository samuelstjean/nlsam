#! /usr/bin/env python
# Caller for the 3D and 4D denoising

from __future__ import division, print_function

import nibabel as nib
import numpy as np

import os
import argparse


from nlsam.denoiser import denoise, greedy_set_finder
from nlsam.angular_tools import angular_neighbors
from nlsam.smoothing import local_standard_deviation

from dipy.io.gradients import read_bvals_bvecs

from time import time
from copy import copy
from ast import literal_eval
from multiprocessing import cpu_count


DESCRIPTION = """
    Convenient script to call the denoising dictionary learning/sparse coding
    functions. It enables the user to select a whole range of parameters to
    test instead of relying on scripts that call the relevant scripts.
    """

def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('input', action='store', metavar='DWI',
                   help='Path of the image file to denoise.')

    p.add_argument('block_size', action='store', metavar='block_size',
                   type=int, help='Number of angular neighbors used for denoising.')

    p.add_argument('bvals', action='store', metavar='bvals',
                   help='Path of the bvals file, in FSL format.')

    p.add_argument('bvecs', action='store', metavar='bvecs',
                   help='Path of the bvecs file, in FSL format.')

    p.add_argument('-sigma', action='store', required=False, type=str,
                   help='Path to standard deviation volume.')

    p.add_argument('-D', action='store', metavar='D',
                   required=False, default=None, type=str,
                   help='Path to a prelearned dictionnary D in npy format to \
                   use for the sparse coding step. Supplying D will skip the \
                   dictionnary learning part.')

    p.add_argument('-overlap', action='store', metavar='overlap',
                   required=False, default=None, type=str,
                   help='Specifies overlap between blocks, ranging from 0 \
                   (no overlap) to min(block_size)-1 (full overlap)')

    p.add_argument('-no_whitening', action='store_false', required=False,
                   help='If True, do not apply ZCA whitening. Each block \
                   will be mean centered and scaled to unit l2 norm. ')

    p.add_argument('-nb_atoms_D', action='store', metavar='nb_atoms_D',
                   required=False, default=128, type=int,
                   help='Number of atoms in the learned dictionnary D.')

    p.add_argument('-batchsize', action='store', metavar='batchsize',
                   required=False, default=512, type=float,
                   help='Size of a minibatch for the dictionnary \
                   learning algorithm.')

    p.add_argument('-lambda_D', action='store', metavar='lambda_D',
                   required=False, default=None, type=float,
                   help='Lambda parameter used for the penalisation in the \
                   dictionnary learning algorithm.')

    p.add_argument('-mode_D', action='store', metavar='mode_D',
                   required=False, default=2, type=int,
                   help='Type of the solved problem for the dictionnary \
                   learning algorithm. See spams documentation for more info')

    p.add_argument('-mode_alpha', action='store', metavar='mode_alpha',
                   required=False, default=2, type=int,
                   help='Type of the solved problem for the sparse coding \
                   step. See spams documentation for more info')

    p.add_argument('-pos_D', action='store_true', required=False,
                   default=False, help='Enforces positivity contraints for D \
                   in the dictionnary learning algorithm.')

    p.add_argument('-debug', action='store_true', required=False,
                   default=False, help='Print debug info and saves intermediate datasets. \
                   May be heavy on RAM usage')

    p.add_argument('-pos_alpha', action='store_true', required=False,
                   help='Enforces positivity contraints for alpha in the \
                   lasso algorithm.')

    p.add_argument('-lambda_lasso', action='store', metavar='lambda_lasso',
                   required=False, default=None, type=float,
                   help='Lambda parameter used for the penalisation in the \
                   lasso algorithm.')

    p.add_argument('-iter', action='store', metavar='iter',
                   required=False, default=1000, type=int,
                   help='Number of iterations in the dictionnary learning \
                   algorithm. A negative value specifies the number of second \
                   used for training instead of the number of iterations')

    p.add_argument('-cores', action='store', dest='cores',
                   metavar='cores', required=False, default=None, type=int,
                   help='Number of cores to use for multithreading')

    p.add_argument('-o', action='store', dest='savename',
                   metavar='savename', required=True, type=str,
                   help='Path and prefix for the saved denoised file.')

    p.add_argument('-mask', action='store', dest='mask_data',
                   metavar='', required=False, default=None, type=str,
                   help='Path to a binary mask. Only the data inside the mask \
                   will be reconstructed by the sparse coding algorithm.')

   p.add_argument('--no_subsampling', dest='no_subsampling', action='store_true',
                   default=False, required=False,
                   help='If supplied, process all volumes multiple time, as opposed to ' +
                   'only at least once.')

    p.add_argument('--no_symmetry', dest='no_symmetry', action='store_true',
                   default=False, required=False,
                   help='If supplied, assumes the set of bvals/bvecs to already be symmetrized, ' +
                   'i.e. All points (x,y,z) on the sphere and (-x,-y,-z) were acquired.')
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    print("Now denoising " + os.path.realpath(args.input))
    print("List of used parameters : ", vars(parser.parse_args()))

    debug = args.debug

    vol = nib.load(args.input)
    data = vol.get_data()
    affine = vol.get_affine()

    use_abs = False
    use_clip = False

    if args.no_subsampling:
        greedy_subsampler = False
    else:
        greedy_subsampler = True

    crop = False
    n_iter = 10
    implausible_signal_hack = True
    debug = False

    if use_abs:
        data = np.abs(data)
    elif use_clip:
        data[data < 0] = 0

    original_dtype = data.dtype

    block_size = np.array((3, 3, 3, int(args.block_size)))
    param_D = {}
    param_alpha = {}

    if len(block_size) != len(data.shape):
        raise ValueError('Block shape and data shape are not of the same \
                         dimensions', data.shape, block_size.shape)

    if args.overlap is not None:
        overlap = np.array(literal_eval(args.overlap))
    else:
        overlap = np.ones(len(block_size), dtype='int16')

    if args.cores is None:
        param_D['numThreads'] = cpu_count()
        param_alpha['numThreads'] = cpu_count()
    else:
        param_D['numThreads'] = args.cores
        param_alpha['numThreads'] = args.cores

    if args.lambda_lasso is None:
        param_alpha['lambda1'] = 1.2 / np.sqrt(np.prod(block_size))
    else:
        param_alpha['lambda1'] = args.lambda_lasso

    if args.lambda_D is None:
        param_D['lambda1'] = 1.2 / np.sqrt(np.prod(block_size))
    else:
        param_D['lambda1'] = args.lambda_D

    if args.D is not None:
        param_alpha['D'] = np.load(args.D)

    param_alpha['mode'] = args.mode_alpha

    param_D['mode'] = args.mode_D
    param_D['iter'] = args.iter
    param_D['K'] = args.nb_atoms_D
    param_D['posD'] = args.pos_D
    param_alpha['pos'] = args.pos_alpha
    param_D['posAlpha'] = args.pos_alpha

    if args.mask_data is not None:
        mask_train = nib.load(args.mask_data).get_data().squeeze().astype(np.bool)
    else:
        mask_train = None

    filename = args.savename

    # Testing neighbors stuff
    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)

    b0_thresh = 10
    b0_loc = tuple(np.where(bvals <= b0_thresh)[0])
    num_b0s = len(b0_loc)

    print("found " + str(num_b0s) + " b0s at position " + str(b0_loc))
    # Average multiple b0s, and just use the average for the rest of the script
    # patching them in at the end
    if num_b0s > 1:
        mean_b0 = np.mean(data[..., b0_loc], axis=-1)
        dwis = tuple(np.where(bvals > b0_thresh)[0])
        data = data[..., dwis]
        bvals = np.take(bvals, dwis, axis=0)
        bvecs = np.take(bvecs, dwis, axis=0)

        rest_of_b0s = b0_loc[1:]
        b0_loc = b0_loc[0]

        data = np.insert(data, b0_loc, mean_b0, axis=-1)
        bvals = np.insert(bvals, b0_loc, [0.], axis=0)
        bvecs = np.insert(bvecs, b0_loc, [0., 0., 0.], axis=0)
        b0_loc = tuple([b0_loc])
        num_b0s = 1
        print("Averaged b0s, new b0_loc is", b0_loc)

    else:
        rest_of_b0s = None

    # Double bvecs to find neighbors with assumed symmetry if needed
    if args.no_symmetry:
        sym_bvecs = np.delete(bvecs, b0_loc, axis=0)
    else:
        sym_bvecs = np.vstack((np.delete(bvecs, b0_loc, axis=0), np.delete(-bvecs, b0_loc, axis=0)))

    neighbors = (angular_neighbors(sym_bvecs, block_size[-1] - num_b0s) % (data.shape[-1] - num_b0s))[:data.shape[-1] - num_b0s]

    if args.mask_data is not None:
        mask_data = nib.load(args.mask_data).get_data().squeeze().astype(np.bool)
        mask = nib.load(args.mask_data).get_data().squeeze().astype(np.bool)
    else:

        mask = np.ones_like(data[..., 0], dtype=np.bool)
        mask_data = mask

        if debug:
            nib.save(nib.Nifti1Image(data, affine), filename + '_masked.nii.gz')
            nib.save(nib.Nifti1Image(mask.astype('int8'), affine), filename + '_mask.nii.gz')
            # nib.save(nib.Nifti1Image(mask_noise.astype('int8'), affine), filename + '_mask_noise.nii.gz')

    if args.sigma is not None:
        sigma = nib.load(args.sigma).get_data()**2
        print("Found sigma volume! Using", args.sigma, "as the noise standard deviation")
    else:
        print("No volume found for noise estimation, using local variance internally for bounding the reconstruction error")
        sigma = local_standard_deviation(data)**2
        nib.save(nib.Nifti1Image(sigma,np.eye(4)), filename + "_variance.nii.gz")

    # Always abs b0s, as it makes absolutely no sense physically not to
    print(np.sum(data[..., b0_loc] < 0), "b0s voxel < 0")
    nib.save(nib.Nifti1Image((data[..., b0_loc] < 0).astype(np.int16), np.eye(4)),'implausible_voxels.nii.gz')
    # Implausible signal hack
    print("Number of implausible signal", np.sum(data[..., b0_loc] < data))
    if implausible_signal_hack:
        data[..., b0_loc] = np.max(data, axis=-1, keepdims=True)
    print("Number of implausible signal after hack", np.sum(data[..., b0_loc] < data))

    nib.save(nib.Nifti1Image(data[..., b0_loc], np.eye(4)),'max_b0s_voxels.nii.gz')

    orig_shape = data.shape
    new_block_size = 3

    print("Choosing new full block size, now", new_block_size, "was", block_size)
    block_size = [new_block_size, new_block_size, new_block_size, block_size[-1]]

    print("overlap is", overlap)
    mask = mask_data

    full_block = np.append(block_size[:-1], data.shape[-1])
    padded_shape = data.shape

    new_block_size = 3

    print("Choosing new  block size, now", new_block_size, "was", block_size)
    block_size = [new_block_size, new_block_size, new_block_size, block_size[-1]]

    overlap = np.array(block_size, dtype=np.int16) - 1

    param_alpha['lambda1'] = 1.2 / np.sqrt(np.prod(block_size))
    param_D['lambda1'] = 1.2 / np.sqrt(np.prod(block_size))
    print("new alpha", param_alpha['lambda1'], param_D['lambda1'])

    b0 = data[..., b0_loc]
    data = np.delete(data, b0_loc, axis=-1)
    neighbors_shape = data.shape[:-1] + (data.shape[-1] * (block_size[-1] + num_b0s),)
    indexes = []
    for i in range(len(neighbors)):
        indexes += [(i,) + tuple(neighbors[i])]

    if greedy_subsampler:
        print("Greedy subsampler hack is on", len(indexes))
        indexes = greedy_set_finder(indexes)
        print("Number of indexes is now", len(indexes))

    if debug:
        nib.save(nib.Nifti1Image(data_neighbors, affine),
                 filename + '_neighbors.nii.gz')

    b0_block_size = tuple(block_size[:-1]) + ((block_size[-1] + num_b0s,))
    print(b0_block_size, block_size[:-1], tuple((block_size[-1] + num_b0s,)), block_size[-1], num_b0s)

    nbiter = 150
    K = np.asscalar(np.prod(b0_block_size))

    param_D['iter'] = nbiter
    param_D['K'] = K  # // 2
    alpha_ml = 0.3
    print("param run", K, nbiter, alpha_ml)
    param_alpha['lambda1'] = alpha_ml
    param_D['lambda1'] = alpha_ml
    denoised_shape = data.shape[:-1] + (neighbors_shape[-1],)
    denoised_shape = data.shape[:-1] + (data.shape[-1] + num_b0s,)
    data_denoised = np.zeros(denoised_shape, np.float64)
    print(data_denoised.shape)
    step = len(indexes[0]) + num_b0s
    for i, idx in enumerate(indexes):
        print(i, idx)

        print(i, i*step, (i + 1)*step)
        dwi_idx = tuple(np.where(idx <= b0_loc, idx, np.array(idx) + num_b0s))
        print(dwi_idx)

        data_denoised[..., b0_loc + dwi_idx] += \
            denoise(np.insert(data[..., idx], (0,), b0, axis=-1),
                    b0_block_size, overlap, param_alpha, param_D,
                    sigma, n_iter, 512, mask, mask,
                    mask, args.no_whitening, filename,
                    dtype=np.float64, debug=debug)

    divider = np.bincount(np.array(indexes, dtype=np.int16).ravel())
    divider = np.insert(divider, b0_loc, len(indexes))
    print(b0_loc, len(indexes), divider.shape)

    data_denoised = data_denoised[:orig_shape[0],
                                  :orig_shape[1],
                                  :orig_shape[2],
                                  :orig_shape[3]] / divider

    # Put back the original number of b0s
    if rest_of_b0s is not None:
        # Number of b0s at the end
        number_of_end_b0s = np.sum(np.array(rest_of_b0s) > data_denoised.shape[-1])

        data_denoised = np.concatenate((data_denoised, np.repeat(data_denoised[..., b0_loc], number_of_end_b0s, axis=-1)), axis=-1)
        # Stack the rest in-between
        rest_of_b0s = rest_of_b0s[:-number_of_end_b0s]
        b0_denoised = np.squeeze(data_denoised[..., b0_loc])

        for idx in rest_of_b0s:
            data_denoised = np.insert(data_denoised, idx, b0_denoised, axis=-1)

    if use_abs:
        data_denoised = np.abs(data_denoised)
    elif use_clip:
        data_denoised[data_denoised < 0] = 0

    nib.save(nib.Nifti1Image(data_denoised.astype(original_dtype), affine), filename)


if __name__ == "__main__":
    main()
