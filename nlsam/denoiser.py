from __future__ import division, print_function

import numpy as np

try:
    from spams import spams
except ImportError:
    try:
        import spams
    except ImportError:
        try:
            from spams_python import spams
        except ImportError:
            raise ValueError("Couldn't find spams library")

from time import time
from itertools import repeat
from multiprocessing import Pool, cpu_count
# from copy import copy

# from scipy.sparse.linalg import lsqr as sparse_lstsq

from scilpy.denoising.utils import sparse_dot, im2col_nd, col2im_nd, padding
# from nlsam.smoothing import local_standard_deviation
import scipy.sparse as ssp
# from sklearn.linear_model import LassoCV, LassoLars, Lasso
# from scipy.optimize import nnls

# import warnings
# warnings.filterwarnings("ignore")
#from sklearn.decomposition import MiniBatchDictionaryLearning as DL
# from scipy.sparse import csc_matrix
# from nlsam.smoothing import local_standard_deviation


def universal_worker(input_pair):
    """http://stackoverflow.com/a/24446525"""
    function, args = input_pair
    return function(*args)


def pool_args(function, *args):
    return zip(repeat(function), zip(*args))


def greedy_set_finder(sets):
    """Returns a list of subsets that spans the input sets with a greedy algorithm
    http://en.wikipedia.org/wiki/Set_cover_problem#Greedy_algorithm"""

    sets = [set(s) for s in sets]
    universe = set()

    for s in sets:
        universe = universe.union(s)
        sets

    # max_number = len(sets)
    output = []

    while len(universe) != 0:

        max_intersect = 0

        for i, s in enumerate(sets):

            n_intersect = len(s.intersection(universe))

            # if n_intersect == max_number:
            #     element = i
            #     break

            if n_intersect > max_intersect:
                max_intersect = n_intersect
                element = i

        output.append(tuple(sets[element]))
        universe = universe.difference(sets[element])

    return output


# def processer(data, mask, variance, block_size, overlap, param_alpha, param_D, downscaling, beta=0.5, gamma=1., dtype=np.float64, eps=1e-10, n_iter=10):
def processer(arglist):

    data, mask, variance, block_size, overlap, param_alpha, param_D, downscaling, beta, gamma, dtype, eps, n_iter = arglist

    orig_shape = data.shape
    var_array = variance[mask]
    mask = np.repeat(mask[..., None], orig_shape[-1], axis=-1)
    mask_array = im2col_nd(mask, block_size, overlap).T
    train_idx = np.sum(mask_array, axis=0) > mask_array.shape[0]/2

    X = im2col_nd(data, block_size, overlap).T
    # var_array = im2col_nd(np.repeat(variance[..., None], orig_shape[-1], axis=-1), block_size, overlap).T

    # If mask is empty, return a bunch of zeros as blocks
    if not np.any(train_idx):
        return np.zeros_like(data)

    X_full_shape = X.shape
    X = np.asfortranarray(X[:, train_idx], dtype=dtype)
    # var_array = var_array[:, train_idx].squeeze()

    # param_alpha['lambda1'] = (2 * np.log(X.shape[0]))**((1+gamma)/2)
    # w = np.ones((param_alpha['D'].shape[1], X.shape[1]), dtype=dtype, order='F')
    # eps = np.ones((X.shape[1]), dtype=dtype, order='F')
    # alpha = np.ones((param_alpha['D'].shape[1], X.shape[1]), dtype=dtype, order='F')
    # alpha = ssp.lil_matrix((param_alpha['D'].shape[1], X.shape[1]), dtype=dtype)
    # alpha_old = np.ones((param_alpha['D'].shape[1], X.shape[1]), dtype=dtype, order='F')
    W = np.ones((param_alpha['D'].shape[1], X.shape[1]), dtype=dtype, order='F')
    eps = np.finfo(dtype).eps
    param_alpha['mode'] = 1
    param_alpha['lambda1'] = np.asscalar(np.percentile(var_array, 95))
    param_alpha['L'] = int(0.5 * X.shape[0])
    # param_alpha['lambda1'] = np.asscalar(var_array.max())
    X_old = np.ones_like(X)
    # X_orig = np.copy(X)
    # X_converged = np.empty_like(X)
    not_converged = np.ones(X.shape[1], dtype=np.bool)
    alpha_converged = np.zeros((param_alpha['D'].shape[1], X.shape[1]), dtype=dtype)
    L2_norm = np.sqrt(np.sum(X**2, axis=0, dtype=np.float32))
    # X_init = np.copy(X)
    # print(alpha_converged.shape)
    # alpha = np.ones_like(alpha_converged)
    # old_conv = np.ones(X.shape[1], dtype=np.bool)

    for i in range(n_iter):

        if np.all(not_converged == 0):
            print("broke the loop", i, np.abs(X - X_old).max(), np.abs(X_old - X).min(), np.max(X), np.min(X), np.max(X) * 0.02)
            break

        param_alpha['W'] = W
        alpha = spams.lassoWeighted(np.asfortranarray(X[:, not_converged]), **param_alpha)
        X[:, not_converged] = sparse_dot(param_alpha['D'], alpha)

        # Convergence if max(X-X_old) < 2 % max(X)
        X_conv = np.max(np.abs(X[:, not_converged] - X_old[:, not_converged]), axis=0) > (np.max(X[:, not_converged], axis=0) * 0.02)
        # X_conv = (np.sqrt(np.sum((X[:, not_converged] - X_old[:, not_converged])**2, axis=0)) / np.sqrt(np.sum(X[:, not_converged]**2, axis=0))) > 10**-5
        x, y, idx = ssp.find(alpha)
        #alpha_converged[:, not_converged] = alpha.toarray()
        alpha_converged[:, not_converged][x, y] = idx
        not_converged[not_converged] = np.logical_or(X_conv, np.any(X[:, not_converged] < 0, axis=0))

        W = np.array(np.maximum(np.repeat(np.sum(np.abs(X[:, not_converged] - X_old[:, not_converged]), axis=0, keepdims=True), param_alpha['D'].shape[1], axis=0), eps), dtype=dtype, order='F', copy=False)
        X_old[:, not_converged] = X[:, not_converged]

    alpha = alpha_converged

    # X = sparse_dot(param_alpha['D'], alpha)
    weigths = np.ones(X_full_shape[1], dtype=dtype, order='F')
    weigths[train_idx] = 1. / np.array((alpha != 0).sum(axis=0) + 1., dtype=dtype).squeeze()
    X2 = np.zeros(X_full_shape, dtype=dtype, order='F')

    X2[:, train_idx] = X

    return col2im_nd(X2, block_size, orig_shape, overlap, weigths)


def denoise(data, block_size, overlap, param_alpha, param_D, variance, n_iter=10, noise_std=None,
            batchsize=512, mask_data=None, mask_train=None, mask_noise=None,
            whitening=True, savename=None, dtype=np.float64, debug=False):
    print(block_size)
    # return data
    # b0 = data.astype(dtype)[..., 0]
    # data = data.astype(dtype)[..., 1:]
    # block_size = block_size[:-1] + (block_size[-1] - 1,)# (3,3,3,5)
    #overlap=(2,2,2,0)
    #print(block_size)
    #1/0
    # block_size=np.array([3,3,3,6])
    # overlap = np.ones(4,dtype=np.int16)*2

    orig_shape = data.shape
    print("real min and max", data.min(), data.max())
    print("test using", dtype)
    print(data.shape)
    block_mean = np.zeros(block_size[-1], dtype=dtype)
    # for i in range(block_size[-1]):
    #    block_mean[i] = np.mean(data[..., i])
    #    data[..., i] -= block_mean[i]
    # sklearn usage - no need to transpose, but breaks everything else (spams+custom normalisation) assuming shape of (features, sample)
    # overlap*=0
    #print(data.shape,block_size,overlap)

    #X_shape = (np.prod(np.array(data.shape) + 1), np.prod(block_size))
    #X = np.zeros(X_shape, dtype=dtype)
    # no overlapping blocks for training
    no_over = (0, 0, 0, 0)
    X = im2col_nd(data, block_size, no_over).T

    # I, J, K = 2*block_size[0]-1, 2*block_size[1]-1, 2*block_size[2]-1
    # print(I,J,K, data.shape)
    # for i in range(0, data.shape[0], I):
    #     for j in range(0, data.shape[1], J):
    #         for k in range(0, data.shape[2], K):
    #             print(X.shape, X[2*block_size[2]-1].shape)
    #             print(i,j,k)
    #             print("im2col")
    #             print(im2col_nd(data[i:i+I, j:j+J, k:k+K], block_size, overlap).shape)
    #             # print("to")
    #             # print(im2col_nd(data[i:i+J, j:j+J, k:k+K], block_size, overlap).T.shape)
    #             # print(i*J*K+j*K+k)
    #             # print((i*J*K+j*K+k+1)*np.prod(block_size[:-1]))

    #             X[(i*J*K+j*K+k)*np.prod(block_size[:-1]):(i*J*K+j*K+k+1)*np.prod(block_size[:-1])] = im2col_nd(data[i:i+I, j:j+J, k:k+K], block_size, overlap)

    # print(X.shape)
    # 1/0
 #   print(X.shape)

  #  xorig = copy(X)

    # Using only training mask data
    # if mask_train is None:
    #     print ("No mask specified for the training data. In order to limit \
    #             \nnoisy samples, a mask consisting of half the volume size \
    #             \nwill be used.")

    #     # Defining the edges of approximately half the mask
    #     a = np.array(data.shape) // 2 - np.array(data.shape) // 4
    #     b = np.array(data.shape) // 2 + np.array(data.shape) // 4

    #     mask_train = np.zeros(data.shape[:3], dtype=bool, order='F')
    #     mask_train[a[0]:b[0]+1, a[1]:b[1]+1, a[2]:b[2]+1] = 1

    print(np.min(X), np.max(X), X.shape)

    # Whitening with ZCA
    # http://ufldl.stanford.edu/wiki/index.php/Implementing_PCA/Whitening
    whitening=False

    if whitening is True:

        print ("Now performing ZCA whitening", X.shape, 'feature, sample')

        X_mean = np.mean(X, axis=0, keepdims=True, dtype=dtype)
        X -= X_mean

        sigma = np.dot(X, X.T)/(X.shape[1]-1)
        U, s, V = np.linalg.svd(sigma)

        epsilon = 10**-5
        ZCA = np.dot(np.dot(U, np.diag(1./np.sqrt(s + epsilon))), U.T)

        X = np.dot(ZCA, X)
        X_norm2 = np.sum(X**2, axis=0, keepdims=True, dtype=dtype)
        X_norm2 = np.ones_like(X_norm2)
        X /= X_norm2
        X[np.isnan(X)] = 0

        #print("eigen values", s, np.count_nonzero(s<epsilon))
        #1/0

    else:

        print("Using normalisation")

       # X_mean = np.mean(X, axis=0, keepdims=True)
        #X_std = np.std(X, axis=0, keepdims=True)

        # Test no normalisation
        #X_mean = np.zeros_like(X_mean)
        #X_std = np.ones_like(X_std)
        #X_std[X_std == 0] = X[X_std == 0]

        # X_mean = np.zeros_like(X, dtype=np.float32)
        # X_std = np.ones_like(X, dtype=np.float32)

        # for i in range(block_size[-1]):

        #     idx = range(np.prod(block_size[:-1])*i, np.prod(block_size[:-1])*(i+1))
        #     X_mean[idx, :] = np.mean(X[idx, :], axis=0, keepdims=True, dtype=dtype)
        #     X_std[idx, :] = np.std(X[idx, :], axis=0, keepdims=True, dtype=dtype)

        #     # test alpha + sigma

          #  std = data[mask_data != 0]    ## changer pour std de chaque vol au lieu de global?
         #   print(np.std(std), "std")
           # print(noise_std[i], "std")
            #X_std[idx, :] = np.ones_like(X_std[idx, :]) * noise_std[i] # np.std(std) ##515/3

        #    X_std[idx, :] = np.ones_like(X_std[idx, :])
         #   X_mean[idx, :] = np.zeros_like(X_mean[idx, :])
          #  print("Test no second normalisation")
            #X_mean *= 0
            #X_std *= 1
            ##param_alpha = 0.1 * param_alpha
           # param_alpha['lambda1'] = 0.1 * param_alpha['lambda1']

     #   X_mean = np.zeros_like(X)
       # X_std = np.ones_like(X)

        # X_mean = np.mean(X, axis=0, keepdims=True, dtype=dtype)
        # X_std = np.std(X, axis=0, keepdims=True, dtype=dtype)
        #X_std[X_std == 0.] = 1.
        X_mean = 0.#np.zeros_like(X)
        X_std = 1.#np.ones_like(X)

        # X -= X_mean
        # X /= X_std
        # X[np.isnan(X)] = 0

        # X_norm2 = np.sqrt(np.sum(X**2, axis=0, keepdims=True, dtype=dtype))
        X_norm2 = 1.#np.ones_like(X_norm2)
        # X /= X_norm2
        # X[np.isnan(X)] = 0

    #X_recon = ((X * X_norm2 * X_std) + X_mean)
    #print(np.sum(np.abs(X_recon-xorig)))
    #1/0
    #import nibabel as nib
    #X_deblock = col2im_nd(X, (block_size), orig_shape, overlap)

    #nib.save(nib.Nifti1Image(X_deblock, np.eye(4)), 'whitening.nii.gz')

    # deb = time()
    # X = np.array(X, dtype=dtype, order='F')
    #X = spams.normalize(X)

    # print ('temps pour shifter X en fortran order :', time()-deb)

    #mask_train = padding(mask_train, block_size, overlap)
    #mask_train = mask_train[..., 0]
    # print(mask_train.shape)
    # train_data_mask = im2col_nd(mask_train, block_size[:3], overlap[:3]).sum(axis=1, keepdims=True)
    # train_data_mask[:, np.sum(train_data_mask, axis=0) > 1] = 1
    # #print (train_data_mask.shape, X.shape)
    # stacks = X.shape[1] / train_data_mask.shape[1]
    # print (stacks, train_data_mask.shape, X.shape, "stack, train, X")
    # #train_data_mask = np.repeat(train_data_mask, stacks, axis=1)
    # train_data_mask = np.repeat(train_data_mask, X.shape[0], axis=0)
    # print (train_data_mask.shape, mask_train.shape, X.shape)

    #deb = time()
    #train_data = np.array(X * train_data_mask, dtype=dtype, order='F')

    #print (X.shape, train_data.shape, train_data_mask.shape, np.sum(train_data_mask)/8, np.min(train_data_mask), np.max(train_data_mask))

    #train_data = train_data
    #X = np.array(X, dtype=dtype, order='F')
    #print ('temps pour shifter train_data en fortran order :', time()-deb)

    #train_data = X
    #train_data = X * train_data_mask
    #X *= train_data_mask
    #X = np.array(X, dtype=dtype, order='F')

    #train_data = np.array(X, dtype=dtype, order='F')

    # Solving for D
    param_D['verbose'] = False
    # param_alpha['pos'] = True
    param_D['posAlpha'] = True
    param_D['posD'] = True
    param_alpha['pos'] = True
    # param_D['whiten'] = True
    param_D['mode'] = 2
    param_D['lambda1'] = 1.2 / np.sqrt(np.prod(block_size))
    #print(param_D['lambda1'])
    # param_D['mode'] = 1
    # param_D['lambda1'] = np.asscalar(np.percentile(variance, 90))
    # print(np.sqrt(param_D['lambda1']))
    # print(1. / np.sqrt(np.prod(block_size)))
    # 1/0
    param_D['K'] = int(2*np.prod(block_size))

    # param_alpha['D'] = np.load('/home/local/USHERBROOKE/stjs2902/Bureau/phantomas_mic/b1000/D.npy')

    if 'D' in param_alpha:
        print ("D is already supplied, \
               \nhot-starting dictionnary learning from D")
        # D = param_alpha['D']
        param_D['D'] = param_alpha['D'] #D
        param_D['iter'] = 150

        start = time()
        step = block_size[0]

        mask = mask_data
        mask_data = im2col_nd(np.repeat(mask_data[..., None], data.shape[-1], axis=-1), block_size, no_over).T
        train_idx = np.sum(mask_data, axis=0) > mask_data.shape[0]/2
        train_data = X[:, train_idx]
        train_data = np.asfortranarray(train_data[:, np.any(train_data != 0, axis=0)], dtype=dtype)
        # train_data = np.asfortranarray(train_data[train_data>0], dtype=dtype)
        # train_data = np.asfortranarray(train_data[:, np.sum(train_data > 0, axis=1) == train_data.shape[0]], dtype=dtype)
        train_data /= np.sqrt(np.sum(train_data**2, axis=0, keepdims=True))
        # train_data[np.isnan(train_data)] = 0
        print("train data shape", train_data.shape, 'X shape', X.shape, train_data.min(), train_data.max())

        # param_D['verbose'] = True

        print("N iter per batch is", param_D['iter'])
        # param_D['iter'] = int(param_D['iter'] * np.ceil(X.shape[1]/param_D['batchsize']))
        print("N iter total is", param_D['iter'])
        # print (param_D)
        param_alpha['D'] = spams.trainDL(train_data, **param_D)
        param_alpha['D'] /= np.sqrt(np.sum(param_alpha['D']**2, axis=0, keepdims=True, dtype=dtype))
        param_D['D'] = param_alpha['D']

        print ("Training done : total time = ",
               time()-start, np.min(param_alpha['D']), np.max(param_alpha['D']))

    else:
        # if 'batchsize' not in param_D:

        if 'K' not in param_D:
            print ('No dictionnary size specified. 256 atoms will be chosen.')
            param_D['K'] = 256

        param_D['iter'] = 150

        start = time()
        # param_D['verbose'] = True
        # print("N iter per batch is", param_D['iter'])
        # param_D['iter'] = int(param_D['iter'] * np.ceil(X.shape[1]/param_D['batchsize']))
        print("N iter total is", param_D['iter'])
        # param_D['K'] = X.shape[0] * 2
        print("number of atom is", param_D['K'], "block size is", X.shape[0])
        # print (param_D)
        # 1/0
        # idx_non_zero = (train_data.sum(0) > 0).reshape(train_data.shape[0], -1)
        # train_data = train_data[:, idx_non_zero]
        # train_idx = np.nonzero(np.sum(X, axis=0) > X.shape[0]/2)[0]
        step = block_size[0]
        # print ("Step train size assumes full overlap!", step, X[:, train_idx].shape)
        # train_data = np.asfortranarray(X[:, train_idx].reshape(X.shape[0], -1))

        param_D['batchsize'] = 500 #train_data.shape[1]//10

        # print("train data shape", train_data.shape, 'X shape', X.shape, 'dtype', train_data.dtype, X.dtype)
        # print(param_D)

        #train_idx = np.nonzero(np.sum(train_data_mask, axis=0, keepdims=True))
        #train_data = np.asfortranarray(X, dtype=dtype)
        #train_data = np.asfortranarray(X[train_data_mask].reshape(X.shape[0], -1))
        mask = mask_data
        # mask_data = mask_data.astype(np.int16)
        # print(mask_data.sum(), np.min(mask_data[...,None]*data), np.max(mask_data[...,None]*data))
        mask_data = im2col_nd(np.repeat(mask_data[..., None], data.shape[-1], axis=-1), block_size, no_over).T
        # print(mask_data.sum(), mask_data.shape)
        train_idx = np.sum(mask_data, axis=0) > mask_data.shape[0]/2
        # print(mask_data.shape[0]/2, train_idx.max())
        # 1/0
        # print(mask_data.sum(), mask_data.dtype, mask_data.shape, train_idx.dtype, train_idx.sum(), train_idx.shape)
        # print(X.shape, train_idx.shape, mask_data.shape, train_idx.dtype, np.sum(train_idx))
        train_data = X[:, train_idx]
        train_data = np.asfortranarray(train_data[:, np.any(train_data != 0, axis=0)], dtype=dtype)
        # print((np.sum(np.abs(train_data), axis=0) > 0).shape, np.any(train_data != 0, axis=0).shape)
        # print(np.all((np.sum(np.abs(train_data), axis=0) > 0) == np.any(train_data != 0, axis=0)))
        # 1/0
        # print(np.sum(train_data>0), np.sum(train_data>0,axis=1), train_data.shape)
        # print(np.sum(train_data > 0, axis=0).shape)
        # print(train_data.shape, "train data shape full")
        # train_data = np.asfortranarray(train_data[:, np.sum(train_data > 0, axis=0) == train_data.shape[0]], dtype=dtype)
        # train_data = np.asfortranarray(X[:, mask_data], dtype=dtype)
        # print(np.min(train_data), np.max(train_data), np.min(X), np.max(X))
        # train_data = np.asfortranarray(X)
        print("train data shape", train_data.shape, 'X shape', X.shape, train_idx.shape, train_data.min(), train_data.max())
       ## print(np.sum(train_data**2,0,keepdims=True))
        # print(train_data.min(), train_data.max(), X[:, train_idx].min(), X[:, train_idx].max(), X.min(), X.max(), dtype)
        # print(train_idx.min(), train_idx.max(), train_idx.dtype, np.sum(mask_data, axis=0).dtype)
        # print(X[:, train_idx].min(), X[:, train_idx].max())
        # print(np.sum(np.sqrt(np.sum(train_data**2, axis=0, keepdims=True), dtype=dtype)==0), np.sum(train_data**2, axis=0, keepdims=True).shape, mask_data.shape[0]/2)
        # 1/0
        train_data /= np.sqrt(np.sum(train_data**2, axis=0, keepdims=True), dtype=dtype)
        # 1/0
        # train_data[np.isnan(train_data)] = 0
        # print(train_data.min(), train_data.max())
        # print(X[:, train_idx].min(), X[:, train_idx].max())
        # print(np.min(train_idx), np.max(train_idx))
        #print(train_data.shape, np.sum(train_data**2,0,keepdims=True).shape)
        #print(np.sum(train_data**2,0,keepdims=True))
        #print(train_data.dtype)
        #print(np.min(train_data), np.max(train_data))
        #1/0
        # param_D['lambda1'] = 0.5
        #train_data = np.asfortranarray(im2col_nd(data.mean(-1, keepdims=True), (3,3,3,1), (0,0,0,0)).T)
        param_alpha['D'] = spams.trainDL(train_data, **param_D)
        # print ("Training done : total time = ",
        #        time()-start, np.min(param_alpha['D']), np.max(param_alpha['D']))
        # D = X[:, X.sum(0) > 0].reshape(X.shape[0], -1)
        # idx = np.random.randint(0, D.shape[1], param_D['K'])
        #
        # print(D.shape)
        param_alpha['D'] /= np.sqrt(np.sum(param_alpha['D']**2, axis=0, keepdims=True, dtype=dtype))
        # param_alpha['D'] = D

        print ("Training done : total time = ",
               time()-start, np.min(param_alpha['D']), np.max(param_alpha['D']))

        if savename is not None:
            np.save(savename + '_D.npy', param_alpha['D'])
    # np.save('D.npy', D)
    # 1/0
    # Solving for alpha
    #####import matplotlib.pyplot as plt
    # print(D.reshape(np.append(block_size, -1)).reshape(np.append(block_size[0]*block_size[1], -1)).shape)
    # print (spams.displayPatches(D.reshape(np.append(block_size, -1)).reshape(np.append(block_size[0]*block_size[1], -1))).squeeze().shape)
    #####plt.imsave(savename + '_D.png', spams.displayPatches(D.reshape(np.append(block_size, -1)).reshape(np.append(block_size[0]*block_size[1], -1))).squeeze(), cmap=plt.gray())
    del train_data
    # if mask_data is None:
    #     print ("Using train_data_mask")
    #     param_alpha['B'] =  np.array(train_data_mask, dtype=bool, order='F')
    #     # del train_data_mask, mask_train
    # else:
    #     print ("Using supplied mask")

    #     # mask_data = padding(mask_data, block_size[:3], overlap[:3])
    #     # #mask_orig = mask_data

    #     # if len(mask_data.shape) == 3:
    #     #     mask_data = mask_data[..., None]

    #     # mask_data = np.repeat(mask_data, orig_shape[-1], axis=-1)
    #     # mask_data = im2col_nd(mask_data, block_size, overlap).T
    #     # print (mask_data.shape, X.shape)

    #     param_alpha['B'] = np.array(mask_data, order='F', dtype=bool)
    #     #param_alpha['B'] = np.ones_like(X, order='F', dtype=bool)

    #     #print (mask_data.shape)
    #     #print (np.repeat(mask_data[..., None], orig_shape[-1], axis=-1).shape)
    #     # im2col_nd(mask_data, block_size[:3], overlap[:3]).shape)
    #     #print (im2col_nd(np.repeat(mask_data[..., None], orig_shape[-1], axis=-1), block_size, overlap).shape)
    #     #1/0

    #     #param_alpha['B'] = np.repeat(im2col_nd(mask_data, block_size[:3], overlap[:3]), stacks, axis=0).T
    #     #param_alpha['B'] = np.array(param_alpha['B'], order='F', dtype=bool)
    #     #param_alpha['B'] = np.ones_like(param_alpha['B'], order='F', dtype=bool)

    #print (mask_data.dtype, mask_data.flags, np.isfortran(mask_data))

    # print (mask_data_block.dtype)
    # print (type(mask_data_block))
    # print (mask_data_block.shape)
    # print (mask_data_block.flags)
    # print (mask_data.shape)
    # print (mask_data.flags)
    # print (X.shape)

    #param_alpha['verbose'] = True
   # start = time()
    #alpha = spams.lassoMask(X, **param_alpha)

    #print ("Recon a partir de l'original")
    #X = im2col_nd(data, block_size, overlap).T
    #X = np.array(X, dtype=dtype, order='F')
    #del data

    # Test sparsity + normalize
    #param_alpha['L'] = X.shape[0]//3
    #X = spams.normalize(X)

    print ("X", X.min(), X.max())

    #X = np.array(xorig, dtype=dtype, order='F')

    # del param_alpha['B']
    start = time()
#    alpha = spams.lassoMask(X, **param_alpha)
    #truc = param_alpha['B']
   # del param_alpha['B']
   ## param_alpha['L'] = 2*64 // D.shape[1]
    param_alpha['mode'] = 2
    #del param_alpha['mode']
    # del param_alpha['lambda1']
    param_alpha['lambda1'] = 1.2 / np.sqrt(np.prod(block_size)) #noise_std#**2 # np.ones(X.shape[-1], dtype=np.float64) * 515.**2
    #param_alpha['lambda1'] = int(param_D['K'])*X.shape[-1]


    # print(param_alpha)
    #param_alpha['L'] = int(param_D['K'] * 0.2)
    #param_alpha['eps'] = 512.**2
    #print(X.shape)
    # 1/0
    # list_groups = np.arange(0, X.shape[-1], overlap[0]+1, dtype=np.int32)




    # list_groups = np.arange(0, X.shape[-1], block_size[1], dtype=np.int32)
    # param_alpha['list_groups'] = list_groups
    # param_alpha['eps'] = param_alpha['lambda1']
    # del param_alpha['lambda1'], param_alpha['pos'], param_alpha['mode']
    # print(list_groups)



    #del param_alpha['pos']

    #param_alpha['pos'] = False
    # X[~mask_data] = 0
    #del param_alpha['pos']

    #print(X.shape)

    # param_alpha['L'] = int(0.5 * X.shape[0])
    ##param_alpha['lambda1'] = 3. / np.sqrt(np.prod(block_size))

    #param_alpha['A0'] = csc_matrix((param_alpha['D'].shape[1], X.shape[1]), dtype=dtype)
    #alpha = spams.lasso(X, **param_alpha)
    # block_size=np.array([3,3,3,6])
    # overlap = np.ones(4,dtype=np.int16) * 2
    # param_alpha['L'] = int(X.shape[0])
    # param_alpha['lambda1'] = 0.2

    # I, J, K = 3*block_size[0], 3*block_size[1], 3*block_size[2]
    # orig_shape = data.shape
    print(data.shape)
    data = padding(data, block_size, overlap)
    # #data_pad = padding(data, (I,J,K, 3), (I-1,J-1,K-1,0))
    # data_out = np.zeros(data_pad.shape, dtype=dtype)
    # print(data_pad.shape)
    print(block_size, overlap)
    print(data.shape, "after padding")
    # print(param_alpha)
    # 1/0
    #gprime = lambda w: np.array(1. / (2. * np.sqrt(np.abs(w.todense())) + np.finfo(float).eps), dtype=dtype, order='F')

   # 1/0
    deb = time()
    # data_out = np.zeros_like(data)
    # data_pad = data
    # print(noise_std.shape)
    # sigma2 = np.ones_like(data)
    # # noise_std = padding(noise_std, (2,2,2), (1,1,1))
    # for i in range(data.shape[-1]):
    #     temp = np.zeros_like(data[..., 0])
    #     temp[:noise_std.shape[0], :noise_std.shape[1], :noise_std.shape[2]] = noise_std**2
    #     sigma2[..., i] = temp
    # del temp
    # print(noise_std.shape)
    # print(data_out.shape)
    # print(im2col_nd(noise_std, block_size[:3], overlap[:3]).shape)
    # print(im2col_nd(sigma2, block_size, overlap).shape)
    # print(np.asfortranarray(im2col_nd(data_pad, block_size, overlap).T, dtype=dtype).shape)
    # 1/0
    # from sklearn.linear_model import LassoLarsCV, MultiTaskLassoCV, MultiTaskLasso, LassoLars, LassoCV, MultiTaskElasticNetCV

    # data_denoised = np.zeros_like(data)
    # first_iter = True

    # for downscaling in [1]:#, 1,1]:#0.75, 0.5]:

    # divider = np.zeros_like(data, dtype=np.int16)

    # if param_alpha['numThreads'] == -1:
    #     n_cores = cpu_count()
    # else:

    n_cores = param_alpha['numThreads']
    # print(n_cores)
    param_alpha['numThreads'] = 1
    param_D['numThreads'] = 1
    # print(param_alpha['numThreads'], param_D['numThreads'])
    # if first_iter:
    #     variance = local_standard_deviation(data)**2
    # else:
    #     variance = local_standard_deviation(data_denoised)**2

# Critere d'arret -> variance a zero?

# for i in range(0, data.shape[0]):#, I-block_size[0]):
#     for j in range(0, data.shape[1]):#, J-block_size[1]):
    # for k in range(data.shape[2] - block_size[2]):#0, , K-block_size[2]):

    #             # print("Current block processed", k)
    #             data_subset = data[..., k:k+block_size[2], :] #i+I, j:j+J] #, k:k+K]
    #             mask_subset = mask[..., k:k+block_size[2]] #i+I, j:j+J] #, k:k+K]
    #             # # variance_subset = 1. #variance[..., k:k+block_size[2]]

    #             data_denoised[..., k:k+block_size[2], :] += processer(data_subset, mask_subset,
    #                                                         block_size, overlap, param_alpha, downscaling, beta=0.5, gamma=0.25)
    #             divider[..., k:k+block_size[2], :] += np.ones_like(data_subset, dtype=np.int16)
    # data_subset = data_denoised

    print('Multiprocessing Stuff')
    time_multi = time()
    pool = Pool(processes=n_cores)

    downscaling = 1.
    beta = 0.5
    gamma = 1

    eps = 1e-10

    # variance *=2
    # variance = local_standard_deviation(data)**2
    # import nibabel as nib
    # nib.save(nib.Nifti1Image(np.sqrt(variance), np.eye(4)), "var.nii")
    # 1/0
    print(np.max(variance),np.min(variance))
    # print(n_cores. data.shape[1], data.shape[1]/n_cores**2)

    arglist = [(data[:, k:k+block_size[1], ...], mask[:, k:k+block_size[1]], variance[:, k:k+block_size[1]], block_size_subset, overlap_subset, param_alpha_subset, param_D_subset, downscaling_subset, beta_subset, gamma_subset, dtype_subset, eps_subset, n_iter_subset)
               for k, block_size_subset, overlap_subset, param_alpha_subset, param_D_subset, downscaling_subset, beta_subset, gamma_subset, dtype_subset, eps_subset, n_iter_subset
               in zip(range(data.shape[1] - block_size[1] + 1),
                      repeat(block_size),
                      repeat(overlap),
                      repeat(param_alpha),
                      repeat(param_D),
                      repeat(downscaling),
                      repeat(beta),
                      repeat(gamma),
                      repeat(dtype),
                      repeat(eps),
                      repeat(n_iter))]

    data_denoised = pool.map(processer, arglist)
    pool.close()
    pool.join()

    param_alpha['numThreads'] = n_cores
    param_D['numThreads'] = n_cores

    print('Multiprocessing done', time()-time_multi)
    # print(len(data_denoised))
    # print(np.asarray(data_denoised).shape)
    # print(type(data_denoised), type(data_denoised[0]), data_denoised[0][0].shape, type(data_denoised[0][1]), data_denoised[0][1].shape )
    #data_subset = np.zeros_like(data)




      #   n_slices = len(data_denoised)
      #   X2 = np.zeros((data_denoised[0][0].shape[0], data_denoised[0][0].shape[1] * n_slices), dtype=dtype, order='F')
      #   weigths = np.zeros(data_denoised[0][1].shape[0] * n_slices, dtype=dtype, order='F')

      #   # Put together the multiprocessed results
      #   for k in range(n_slices): #0, , K-block_size[2]):
      #       # print(k, type(data_denoised), type(data_denoised[k]))
      #       # print(k * n_slices, (k+1) * n_slices + 1)
      #       # print(X2[k * n_slices:(k+1) * n_slices + 1].shape, weigths[k * n_slices:(k+1) * n_slices + 1].shape, data_denoised[k][0].shape, data_denoised[k][1].shape)
      #       # print(weigths.shape)

      #       idx = k * data_denoised[k][0].shape[1], (k+1) * data_denoised[k][0].shape[1]
      #       # print(len(range(idx[0],idx[1])), len(data_denoised), weigths.shape, idx, data_denoised[k][0].shape, data_denoised[k][1].shape)
      #       # print(X2[:, range(k, k+n_slices)].shape, weigths[range(k, k+n_slices)].shape, X2.shape, weigths.shape)
      #     #  print(range(k * n_slices, (k+1) * n_slices))
      #       X2[:, k * data_denoised[k][0].shape[1]:(k+1) * data_denoised[k][0].shape[1]], weigths[k * data_denoised[k][0].shape[1]:(k+1) * data_denoised[k][0].shape[1]] = data_denoised[k]

      #       # print(idx)
      # #  1/0
      #       # print(np.min(X2[:, idx[0]:idx[1]]), np.max(X2[:, idx[0]:idx[1]]), np.min(weigths[idx[0]:idx[1]]), np.max(weigths[idx[0]:idx[1]]))
      #       # print(np.alltrue(X2[:, idx[0]:idx[1]] == data_denoised[k][0]),  np.alltrue(weigths[idx[0]:idx[1]] == data_denoised[k][1]))
      #       # print(X2[:, idx[0]:idx[1]].shape == data_denoised[k][0].shape,  weigths[idx[0]:idx[1]].shape == data_denoised[k][1].shape)
      #   # weigths = np.ones_like(weigths)
      #   print(X2.shape, weigths.shape, X2.min(), X2.max(), weigths.min(), weigths.max())
      #   data_subset = col2im_nd(X2, block_size, data.shape, overlap, weigths)
        # print(np.sum(np.isnan(data_subset)), data_subset.shape)

    # Put together the multiprocessed results
    data_subset = np.zeros_like(data)
    divider = np.zeros_like(data, dtype=np.int16)
    ones = np.ones_like(data_denoised[0], dtype=np.int16)

    for k in range(len(data_denoised)):
        data_subset[:, k:k+block_size[1], ...] += data_denoised[k]
        divider[:, k:k+block_size[1], ...] += ones

    data_subset /= divider
            # print(data_denoised.shape, data_denoised[..., k:k+block_size[2], :].shape, data_subset.shape)
            # print(k, k+block_size[2])
            #  data_subset #, k:k+K] = data_subset
    print(np.sum(np.isnan(data_subset)), 'nans')
        # divider[divider == 0] = 1
        # data_subset /= divider
        # first_iter = False
        # data = data_subset
    print("temps fit :", time()-deb)
    print("data_out", data_subset.min(), data_subset.max(), np.sum(data_subset<0))
    return data_subset #.astype(dtype)[:orig_shape[0], :orig_shape[1], :orig_shape[2]]
