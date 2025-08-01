import os
import argparse
import logging

from multiprocessing import cpu_count
from ast import literal_eval

import nibabel as nib
import numpy as np

from nlsam.angular_tools import read_bvals_bvecs
from nlsam.denoiser import nlsam_denoise
from nlsam.smoothing import local_standard_deviation
from nlsam.bias_correction import stabilization, root_finder_sigma

from autodmri.estimator import estimate_from_nmaps, estimate_from_dwis

DESCRIPTION = """
Main script for the NLSAM denoising [1], including the bias correction framework from [2].
"""

EPILOG = """
References :

[1] St-Jean, S., Coupe, P., & Descoteaux, M. (2016).
Non Local Spatial and Angular Matching : Enabling higher spatial resolution diffusion MRI datasets through adaptive denoising.
Medical Image Analysis, 32(2016), 115-130. doi:10.1016/j.media.2016.02.010

[2] Koay CG, Ozarslan E and Basser PJ.
A signal transformational framework for breaking the noise floor and its applications in MRI.
Journal of Magnetic Resonance 2009; 197: 108-119.

[3] St-Jean S, De Luca A, Tax C.M.W., Viergever M.A, Leemans A.
Automated characterization of noise distributions in diffusion MRI data.
Medical Image Analysis, June 2020:101758. doi:10.1016/j.media.2020.101758
"""


def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION,
                                epilog=EPILOG,
                                add_help=False,
                                formatter_class=argparse.RawTextHelpFormatter)

    #####################
    # Required arguments
    #####################

    required = p.add_argument_group('Required arguments')

    required.add_argument('input', metavar='input',
                          help='Path of the image file to denoise.')

    required.add_argument('output', metavar='output',
                          help='Path for the saved denoised file.')

    required.add_argument('bvals', metavar='bvals',
                          help='Path of the bvals file, in FSL format.')

    required.add_argument('bvecs', metavar='bvecs',
                          help='Path of the bvecs file, in FSL format.')

    #####################
    # Optional arguments
    #####################

    optionals = p.add_argument_group('Optional arguments')

    optionals.add_argument('-N', metavar='float', default='auto',
                          help='Number of receiver coils of the scanner. \n'
                          'Use N = 1 in the case of a SENSE (Philips) reconstruction and '
                          'N >= 1 for GRAPPA based reconstruction (Siemens).\n'
                          'You can also pass "auto" (the default) to estimate it automatically or a filename to load it as a volume.')

    optionals.add_argument('--angular_block_size', default=5, metavar='int', type=int,
                           help='Number of angular neighbors used for denoising.')

    optionals.add_argument('-m', '--mask', metavar='file', required=True,
                           help='Path to a binary mask. Only the data inside the mask will be reconstructed and used for estimation.\n'
                           'This is now a required input to prevent sampling (and reconstructing) background noise instead of the data.')

    optionals.add_argument('--b0_threshold', metavar='int', default=10, type=int,
                           help='Lowest bvalue to be considered as a b0. Default 10')

    optionals.add_argument('--split_b0s', action='store_true',
                           help='If set and multiple b0s are present, they are split amongst the '
                           'training data.')

    optionals.add_argument('--split_shell', action='store_true',
                           help='If set, each shell/bvalue is processed separately by itself.')

    optionals.add_argument('--bval_threshold', metavar='int', default=25, type=int,
                           help='Any bvalue within += bval_threshold of each others will be considered on the same shell (e.g. b=990 and b=1000 are on the same shell). Default 25')

    optionals.add_argument('--block_size', dest='spatial_block_size',
                           metavar='tuple', type=literal_eval, default=(3, 3, 3),
                           help='Size of the 3D spatial patch to be denoised. Default : 3, 3, 3')

    optionals.add_argument('--is_symmetric', action='store_true',
                           help='If supplied, assumes the set of bvals/bvecs to be already symmetrized,\n'
                           'i.e. All points (x,y,z) on the sphere and (-x,-y,-z) were acquired, such as in full grid DSI.')

    optionals.add_argument('--iterations', metavar='int', default=10, type=int,
                           help='Maximum number of iterations for the l1 reweighting. Default 10.')

    optionals.add_argument('--no_subsample', action='store_false',
                           help='If set, process all the dwis multiple times, '
                           'but note that this option lengthen the total time.\n'
                           'The default is to find the smallest subset so that each dwi is '
                           'processed at least once.')

    g1 = optionals.add_mutually_exclusive_group()

    g1.add_argument('--load_mhat', metavar='file',
                    help='Load this volume as a m_hat value estimation for the stabilization.\n'
                    'This is used to replace the argument --sh_order with another initialisation for the algorithm.')

    g = optionals.add_mutually_exclusive_group()

    g.add_argument('--noise_est',
                   dest='noise_method',
                   metavar='string',
                   choices=['local_std', 'auto'],
                   default='auto',
                   help='Noise estimation method used for estimating sigma.\n'
                   'local_std : Compute local noise standard deviation '
                   'with correction factor. No a priori needed.\n'
                   'auto (default): Automatically estimate sigma and N from background in the data.')

    g.add_argument('--load_sigma', metavar='file',
                   help='Load this file as the noise standard deviation volume.\n'
                   'Will be squared internally to turn into variance.')

    #####################
    # Advanced arguments
    #####################

    advanced = p.add_argument_group('Advanced noise estimation')

    advanced.add_argument('--noise_maps', metavar='file',
                          help='Path of the noise map(s) volume for automatic estimation.\n'
                          'Either supply a 3D noise map or a stack of 3D maps as a 4D volume.\n'
                          'This is intended for noise maps collected by the scanner (so that there is no signal in those measurements)\n'
                          'which are properly scaled with the rest of the data you collected.\n'
                          'If in doubt, it is safer to use another estimation method with --noise_est\n'
                          'Note that you also need to pass "auto" as a value for N since it also gets estimated')

    advanced.add_argument('--noise_mask', dest='save_piesno_mask', metavar='file',
                          help='If supplied, output filename for saving the mask of noisy voxels found by the automatic estimation.')

    advanced.add_argument('--save_stab', metavar='file',
                          help='Path to save the intermediate noisy bias corrected volume.')

    advanced.add_argument('--save_sigma', metavar='file',
                          help='Path to save the intermediate standard deviation volume.')

    advanced.add_argument('--save_N', metavar='file',
                          help='Path to save the intermediate N (degree of freedoms) volume.')

    advanced.add_argument('--save_difference', metavar='file',
                          help='Path to save the absolute difference volume, abs(noisy - denoised).')

    advanced.add_argument('--save_eta', metavar='file',
                          help='Path to save the intermediate underlying signal intensity eta volume.')

    advanced.add_argument('--no_clip_eta', action='store_false',
                          help='If set, allows eta to take negative values during stabilization, which is physically impossible.')

    ############
    # The rest
    ############

    misc = p.add_argument_group('Logging and multicores options')

    misc.add_argument('--no_stabilization', action='store_true',
                      help='If set, does not correct the data for the noise non gaussian bias.\n'
                      'Useful if your data is already bias corrected or you would like to do it afterwards.')

    misc.add_argument('--no_denoising', action='store_true',
                      help='If set, does not run the nlsam denoising.\n'
                      'Useful if you only want to bias correct your data or get the noise estimation maps only.')

    misc.add_argument('--cores', metavar='int', type=int,
                      help='Number of cores to use for multithreading.')

    misc.add_argument('--use_f32', action='store_true',
                      help='If supplied, use float32 for inner computations. This option lowers ram usage, but\n'
                      'could lead to numerical precision issues, so use carefully and inspect the final output.')

    misc.add_argument('-f', '--force', action='store_true', dest='overwrite',
                      help='If set, the output denoised volume will be overwritten '
                      'if it already exists.')

    misc.add_argument('-v', '--verbose', action='store_true',
                      help='If set, print useful information message during processing.')

    misc.add_argument('-l', '--log', dest='logfile', metavar='file',
                      help='Save the logging output to this file. Implies verbose output.')

    misc.add_argument("-h", "--help", action="help", help="Show this help message and exit.")

    ############
    # Old stuff
    ############

    deprecated = p.add_argument_group('Deprecated options')

    deprecated.add_argument('--fix_implausible', action='store_true', dest='implausible_signal_fix',
                            help='This option has been removed and has no effect.')

    deprecated.add_argument('--sh_order', metavar='int', default=0, type=int, choices=[0, 2, 4, 6, 8],
                            help='This option has been removed and has no effect.')

    deprecated.add_argument('--mp_method', metavar='string',
                            help='This option has been removed and has no effect.')

    deprecated.add_argument('--use_threading', action='store_true',
                            help='This option has been removed and has no effect.')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    noise_method = args.noise_method
    subsample = args.no_subsample
    is_symmetric = args.is_symmetric
    n_iter = args.iterations
    b0_threshold = args.b0_threshold
    bval_threshold = args.bval_threshold
    split_b0s = args.split_b0s
    split_shell = args.split_shell
    block_size = np.array(args.spatial_block_size + (args.angular_block_size,))
    clip_eta = args.no_clip_eta
    logger = logging.getLogger('nlsam')
    verbose = args.verbose

    if args.logfile is not None:
        handler = logging.FileHandler(args.logfile)
        verbose = True
    else:
        handler = logging.StreamHandler(args.logfile)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if verbose:
        logger.setLevel(logging.INFO)
        logger.info('Verbosity is on')

    if args.no_stabilization:
        logger.info('Stabilization disabled!')

    if args.no_denoising:
        logger.info('Denoising disabled!')

    if args.use_f32:
        logger.info('Computations will be made using float32!')
        dtype = np.float32
    else:
        dtype = np.float64

    if args.implausible_signal_fix:
        logger.warning('Option --implausible_signal_fix has been deprecated')

    if args.sh_order:
        logger.warning('Option --sh_order has been deprecated')

    if args.mp_method:
        logger.warning('Option --mp_method has been deprecated')

    if args.use_threading:
        logger.warning('Option --use_threading has been deprecated')

    ##########################################
    #  Load up data and do some sanity checks
    ##########################################

    overwritable_files = [args.output,
                          args.save_sigma,
                          args.save_difference,
                          args.save_piesno_mask,
                          args.save_stab,
                          args.save_eta]

    for f in overwritable_files:
        if f is not None and os.path.isfile(f):
            if args.overwrite:
                logger.warning(f'Overwriting {os.path.realpath(f)}')
            else:
                parser.error(f'{f} already exists! Use -f or --force to overwrite it.')

    # Load up N or turn it into a number if it fails
    if args.N == 'auto':
        N = 'auto'
    else:
        try:
            N = nib.load(args.N).get_fdata(caching='unchanged', dtype=np.float32)
        except IOError:
            N = float(args.N)

    # this is to prevent triggering an error check since noise_method is not used in this case and skipped in an else-if clause
    if args.load_sigma is not None:
        noise_method = None

    if args.N != 'auto':
        if args.noise_maps is not None:
            parser.error(f'You need to pass -N auto when using noise maps, but you passed -N {args.N}')

        if noise_method == 'auto':
            parser.error(f'You need to pass -N auto when using --noise_est auto, but you passed -N {args.N}')

    if args.noise_maps is None:
        if args.N == 'auto' and noise_method != 'auto':
            parser.error(f'You need to pass --noise_est auto when using -N auto, but you passed --noise_est {noise_method}. Pass -N explicitly or use --noise_est auto')

    if args.load_sigma is None:
        if isinstance(N, np.ndarray):
            parser.error(f'You need to pass --load_sigma sigma.nii.gz when loading N as a volume, but you passed -N {args.N}')
    else:
        if args.N == 'auto':
            parser.error(f'You need to pass -N explicitly when using --load_sigma sigma.nii.gz, but you passed -N {args.N}')

    vol = nib.load(args.input)
    data = vol.get_fdata(caching='unchanged', dtype=np.float32)
    affine = vol.affine
    header = vol.header
    header.set_data_dtype(np.float32)
    logger.info(f"Loading data {os.path.realpath(args.input)}")

    if args.mask is not None:
        mask = nib.load(args.mask).get_fdata(caching='unchanged').astype(bool)
        logger.info(f"Loading mask {os.path.realpath(args.mask)}")
    else:
        mask = np.ones(data.shape[:-1], dtype=bool)

    # load m_hat if we supplied the argument
    if args.load_mhat is not None:
        m_hat = nib.load(args.load_mhat).get_fdata(caching='unchanged', dtype=np.float32)
        logger.info(f"Loading m_hat {os.path.realpath(args.load_mhat)}")
        if m_hat.shape != data.shape:
            raise ValueError(f'm_hat shape {m_hat.shape} is different from data shape {data.shape}!')
    else:
        m_hat = data

    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)

    if args.cores is None or args.cores > cpu_count():
        n_cores = cpu_count()
    else:
        n_cores = args.cores

    if len(block_size) != len(data.shape):
        raise ValueError(f'Block shape {block_size} and data shape {data.shape} are not of the same length')

    if data.shape[:-1] != mask.shape:
        raise ValueError(f'data shape is {data.shape}, but mask shape {mask.shape} is different!')

    #########################
    #  Noise estimation part
    #########################

    if args.load_sigma is not None:
        sigma = nib.load(args.load_sigma).get_fdata(caching='unchanged', dtype=np.float32)
        logger.info(f"Loading sigma {os.path.realpath(args.load_sigma)}")

        # If we have a bunch of 4D views, we only want a 3D array
        if sigma.ndim == data.ndim:
            sigma = np.median(sigma, axis=-1)
            logger.warning(f"{os.path.realpath(args.load_sigma)} was {sigma.ndim + 1}D, but was downcasted to {sigma.ndim}D.")

        if data.shape[:-1] != sigma.shape:
            raise ValueError(f'data shape is {data.shape}, but sigma shape {sigma.shape} is different!')

    elif args.noise_maps is not None:
        noise_maps = nib.load(args.noise_maps).get_fdata(caching='unchanged', dtype=np.float32)
        logger.info(f"Loading {os.path.realpath(args.noise_maps)} as a noise map")

        # Needs to be 4D
        if noise_maps.ndim != 4:
            noise_maps = noise_maps[..., None]

        sigma, N, mask_noise = estimate_from_nmaps(noise_maps, ncores=n_cores, verbose=verbose)

        if args.save_piesno_mask is not None:
            nib.save(nib.Nifti1Image(mask_noise.astype(np.int16), affine), args.save_piesno_mask)
            logger.info(f"Mask of noisy voxels saved as {os.path.realpath(args.save_piesno_mask)}")

        if args.save_N is not None:
            nib.save(nib.Nifti1Image(N, affine), args.save_N)
            logger.info(f"N map saved as {os.path.realpath(args.save_N)}")

    elif noise_method == 'local_std':
        logger.info(f"Estimating noise with method {noise_method}")
        sigma = local_standard_deviation(data, n_cores=n_cores, verbose=verbose)

        # Compute the corrected value for each 3D volume
        if N > 0:
            sigma = root_finder_sigma(data, sigma, N, mask=mask, verbose=verbose, n_cores=n_cores)

    elif noise_method == 'auto':
        logger.info(f"Estimating noise with method {noise_method}")

        sigma_1D, N_1D, mask_noise = estimate_from_dwis(data, return_mask=True, ncores=n_cores, verbose=verbose)
        sigma = np.broadcast_to(sigma_1D[None, None, :, None], data.shape)
        N = np.broadcast_to(N_1D[None, None, :, None], data.shape)

        if args.save_piesno_mask is not None:
            nib.save(nib.Nifti1Image(mask_noise.astype(np.int16), affine), args.save_piesno_mask)
            logger.info(f"Mask of noisy voxels saved as {os.path.realpath(args.save_piesno_mask)}")

        if args.save_N is not None:
            nib.save(nib.Nifti1Image(N, affine), args.save_N)
            logger.info(f"N map saved as {os.path.realpath(args.save_N)}")

    if args.save_sigma is not None:
        nib.save(nib.Nifti1Image(sigma, affine), args.save_sigma)
        logger.info(f"Sigma map saved as {os.path.realpath(args.save_sigma)}")

    ##################
    #  Stabilizer part
    ##################

    if args.no_stabilization:
        data_stabilized = data
    else:
        logger.info("Now performing stabilization")

        # We may have a 3D sigma map, so broadcast to 4D for indexing
        if sigma.ndim == 3:
            sigma = np.broadcast_to(sigma[..., None], data.shape)

        data_stabilized, eta = stabilization(data,
                                             m_hat,
                                             sigma,
                                             N,
                                             mask=mask,
                                             clip_eta=clip_eta,
                                             return_eta=True,
                                             n_cores=n_cores,
                                             verbose=verbose)

        if args.save_eta is not None:
            nib.save(nib.Nifti1Image(eta, affine), args.save_eta)
            logger.info(f"eta volume saved as {os.path.realpath(args.save_eta)}")

        if args.save_stab is not None:
            nib.save(nib.Nifti1Image(data_stabilized, affine), args.save_stab)
            logger.info(f"Stabilized data saved as {os.path.realpath(args.save_stab)}")

        del m_hat, eta

    ##################
    #  Denoising part
    ##################

    if not args.no_denoising:
        logger.info(f"Now denoising {os.path.realpath(args.input)}")

        # If we have a bunch of 4D views, we only want a 3D array
        if sigma.ndim == data.ndim:
            sigma = np.median(sigma, axis=-1)

        if args.save_difference is None:
            del data

        data_denoised = nlsam_denoise(data_stabilized,
                                      sigma,
                                      bvals,
                                      bvecs,
                                      block_size,
                                      mask=mask,
                                      is_symmetric=is_symmetric,
                                      n_cores=n_cores,
                                      split_b0s=split_b0s,
                                      split_shell=split_shell,
                                      subsample=subsample,
                                      n_iter=n_iter,
                                      b0_threshold=b0_threshold,
                                      bval_threshold=bval_threshold,
                                      dtype=dtype,
                                      verbose=verbose)

        nib.save(nib.Nifti1Image(data_denoised.astype(np.float32), affine, header), args.output)
        logger.info(f"Denoised data saved as {os.path.realpath(args.output)}")

        if args.save_difference is not None:
            nib.save(nib.Nifti1Image(np.abs(data_denoised - data).astype(np.float32), affine, header), args.save_difference)
            logger.info(f"Difference map saved as {os.path.realpath(args.save_difference)}")

    return None


def main_workaround_joblib():
    # Until joblib.loky support pyinstaller, we use dask instead for the frozen binaries
    import sys
    frozen = getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

    if frozen:
        from multiprocessing import freeze_support
        freeze_support()

        from dask.distributed import Client
        from joblib import parallel_backend

        # Use the argparser to only fetch the number of cores, everything is still processed in main()
        parser = buildArgsParser()
        args = parser.parse_args()

        if args.cores is None or args.cores > cpu_count():
            n_cores = cpu_count()
        else:
            n_cores = args.cores

        client = Client(threads_per_worker=1, n_workers=n_cores)
        with parallel_backend("dask"):
            main()
    else:
        main()


if __name__ == "__main__":
    main_workaround_joblib()
