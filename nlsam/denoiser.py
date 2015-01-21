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
# from copy import copy

from nlsam.utils import sparse_dot, im2col_nd, col2im_nd, padding
#from sklearn.decomposition import MiniBatchDictionaryLearning as DL
# from scipy.sparse import csc_matrix
from nlsam.smoothing import local_standard_deviation


def processer(data, mask, block_size, overlap, D, param_alpha, downscaling, beta=0.5, dtype=np.float64):

    orig_shape = data.shape
    # print(data.shape, mask.shape)
    # mask = np.repeat(mask[..., None], orig_shape[-1], axis=-1)
    mask_array = im2col_nd(mask, block_size[:3], overlap[:3]).T
    train_idx = np.sum(mask_array, axis=0) > mask_array.shape[0]/2
    # print(mask_array.shape, train_idx.shape)

    # If mask is empty, return a bunch of zeros
    if np.sum(train_idx) == 0:
        return np.zeros_like(data)

    sigma2 = local_standard_deviation(data)**2
    # print(sigma2.shape)
    var_array = im2col_nd(sigma2, block_size[:3], overlap[:3]).T
    # print(var_array.shape)
    X = im2col_nd(data, block_size, overlap).T
    X_full_shape = X.shape
    X = np.asfortranarray(X[:, train_idx], dtype=dtype)

    param_alpha['W'] = np.array(np.ones((D.shape[-1], X.shape[-1]), dtype=dtype) * beta * var_array[var_array.shape[0]//2, train_idx], order='F')
    param_alpha['L'] = int(0.5 * X.shape[0])
    param_alpha['lambda1'] = downscaling * 1.2 / np.sqrt(np.prod(block_size))

    # print(param_alpha['W'].shape, param_alpha['D'].shape, X.shape)
    # print(var_array.shape, var_array[var_array.shape[0]//2, train_idx].shape, np.sum(train_idx))
    # 1/0
    # print(param_alpha['W'].dtype, param_alpha['D'].dtype, X.dtype)
    # print(param_alpha['W'].flags, param_alpha['D'].flags, X.flags)
    # print(data.shape, sigma2.shape, block_size, overlap)
    # print(train_idx.shape, orig_shape)
    alpha = spams.lassoWeighted(X, **param_alpha)
    X = sparse_dot(D, alpha)
    # print(alpha.shape, X_full_shape, X.shape, train_idx.shape)
    weigths = np.zeros(X_full_shape[1], dtype=dtype)
    weigths[train_idx] = 1. / np.array((np.abs(alpha) > 1e-15).sum(axis=0) + 1., dtype=dtype).squeeze()
    X2 = np.zeros(X_full_shape, dtype=dtype)
    # print(X2.shape, X.shape, X2[:, train_idx].shape, np.sum(train_idx), train_idx.shape)
    X2[:, train_idx] = X
    # print(weigths.shape, X2.shape, train_idx.shape)
    # del param_alpha['W']
    return col2im_nd(X2, block_size, orig_shape, overlap, weigths)


def denoise(data, block_size, overlap, param_alpha, param_D, noise_std=None,
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
    X = im2col_nd(data, block_size, [0, 0, 0, 0]).T

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
    if mask_train is None:
        print ("No mask specified for the training data. In order to limit \
                \nnoisy samples, a mask consisting of half the volume size \
                \nwill be used.")

        # Defining the edges of approximately half the mask
        a = np.array(data.shape) // 2 - np.array(data.shape) // 4
        b = np.array(data.shape) // 2 + np.array(data.shape) // 4

        mask_train = np.zeros(data.shape[:3], dtype=bool, order='F')
        mask_train[a[0]:b[0]+1, a[1]:b[1]+1, a[2]:b[2]+1] = 1

    print (np.min(X), np.max(X), X.shape)

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

        X -= X_mean
        X /= X_std
        X[np.isnan(X)] = 0

        # X_norm2 = np.sqrt(np.sum(X**2, axis=0, keepdims=True, dtype=dtype))
        X_norm2 = 1.#np.ones_like(X_norm2)
        X /= X_norm2
        X[np.isnan(X)] = 0

    #X_recon = ((X * X_norm2 * X_std) + X_mean)
    #print(np.sum(np.abs(X_recon-xorig)))
    #1/0
    #import nibabel as nib
    #X_deblock = col2im_nd(X, (block_size), orig_shape, overlap)

    #nib.save(nib.Nifti1Image(X_deblock, np.eye(4)), 'whitening.nii.gz')

    deb = time()
    # X = np.array(X, dtype=dtype, order='F')
    #X = spams.normalize(X)

    print ('temps pour shifter X en fortran order :', time()-deb)

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
    # param_D['whiten'] = True
    param_D['mode'] = 2
    param_D['lambda1'] = 1.2 / np.sqrt(np.prod(block_size))
    print(param_D['lambda1'])
    # print(1. / np.sqrt(np.prod(block_size)))
    # 1/0
    param_D['K'] = int(2*np.prod(block_size))
    param_D['iter'] = 1000
    # param_alpha['D'] = np.load('/home/local/USHERBROOKE/stjs2902/Bureau/phantomas_mic/b1000/D.npy')

    if 'D' in param_alpha:
        print ("D is already supplied, \
               \nhot-starting dictionnary learning from D")
        D = param_alpha['D']
        param_D['D'] = D
        param_D['iter'] = 100

        start = time()
        step = overlap[0] + 1

        mask = mask_data
        mask_data = im2col_nd(mask_data, block_size[:3], [0, 0, 0]).T
        train_idx = np.sum(mask_data, axis=0) > mask_data.shape[0]/2
        train_data = np.asfortranarray(X[:, train_idx], dtype=dtype)
        train_data /= np.sqrt(np.sum(train_data**2, axis=0, keepdims=True))
        print("train data shape", train_data.shape, 'X shape', X.shape)

        # param_D['verbose'] = True

        print("N iter per batch is", param_D['iter'])
        # param_D['iter'] = int(param_D['iter'] * np.ceil(X.shape[1]/param_D['batchsize']))
        print("N iter total is", param_D['iter'])
        print (param_D)
        D = spams.trainDL(train_data, **param_D)
        print ("Training done : total time = ",
               time()-start, np.min(D), np.max(D))

    else:
        # if 'batchsize' not in param_D:

        if 'K' not in param_D:
            print ('No dictionnary size specified. 256 atoms will be chosen.')
            param_D['K'] = 256

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
        step = overlap[0] + 1
        # print ("Step train size assumes full overlap!", step, X[:, train_idx].shape)
        # train_data = np.asfortranarray(X[:, train_idx].reshape(X.shape[0], -1))

        param_D['batchsize'] = 500 #train_data.shape[1]//10

        # print("train data shape", train_data.shape, 'X shape', X.shape, 'dtype', train_data.dtype, X.dtype)
        print(param_D)

        #train_idx = np.nonzero(np.sum(train_data_mask, axis=0, keepdims=True))
        #train_data = np.asfortranarray(X, dtype=dtype)
        #train_data = np.asfortranarray(X[train_data_mask].reshape(X.shape[0], -1))
        mask = mask_data
        mask_data = im2col_nd(mask_data, block_size[:3], [0, 0, 0]).T
        train_idx = np.sum(mask_data, axis=0) > mask_data.shape[0]/2
        print(X.shape, train_idx.shape, mask_data.shape)
        train_data = np.asfortranarray(X[:, train_idx], dtype=dtype)
        print(np.min(train_data), np.max(train_data))
        # train_data = np.asfortranarray(X)
        print(train_data.shape, X.shape, train_idx.shape)
       ## print(np.sum(train_data**2,0,keepdims=True))
        train_data /= np.sqrt(np.sum(train_data**2, axis=0, keepdims=True), dtype=dtype)
        #print(train_data.shape, np.sum(train_data**2,0,keepdims=True).shape)
        #print(np.sum(train_data**2,0,keepdims=True))
        #print(train_data.dtype)
        #print(np.min(train_data), np.max(train_data))
        #1/0
        # param_D['lambda1'] = 0.5
        #train_data = np.asfortranarray(im2col_nd(data.mean(-1, keepdims=True), (3,3,3,1), (0,0,0,0)).T)
        D = spams.trainDL(train_data, **param_D)

        # D = X[:, X.sum(0) > 0].reshape(X.shape[0], -1)
        # idx = np.random.randint(0, D.shape[1], param_D['K'])
        #
        # print(D.shape)
        param_alpha['D'] = D

        print ("Training done : total time = ",
               time()-start, np.min(D), np.max(D))

        if savename is not None:
            np.save(savename + '_D.npy', D)
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
    param_alpha['pos'] = True

    print(param_alpha)
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

    param_alpha['D'] /= np.sqrt(np.sum(param_alpha['D']**2, axis=0, keepdims=True, dtype=dtype))
    #print(X.shape)

    param_alpha['L'] = int(0.5 * X.shape[0])
    ##param_alpha['lambda1'] = 3. / np.sqrt(np.prod(block_size))

    #param_alpha['A0'] = csc_matrix((param_alpha['D'].shape[1], X.shape[1]), dtype=dtype)
    #alpha = spams.lasso(X, **param_alpha)
    # block_size=np.array([3,3,3,6])
    # overlap = np.ones(4,dtype=np.int16) * 2
    # param_alpha['L'] = int(X.shape[0])
    # param_alpha['lambda1'] = 0.2

    I, J, K = 3*block_size[0], 3*block_size[1], 3*block_size[2]
    # orig_shape = data.shape
    data = padding(data, block_size, overlap)
    # #data_pad = padding(data, (I,J,K, 3), (I-1,J-1,K-1,0))
    # data_out = np.zeros(data_pad.shape, dtype=dtype)
    # print(data_pad.shape)
    print(block_size, overlap)
    print(data.shape)
    print(param_alpha)
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

    data_denoised = np.zeros_like(data)
    divider = np.zeros_like(data, dtype=np.int16)

    # for i in range(0, data.shape[0]):#, I-block_size[0]):
    #     for j in range(0, data.shape[1]):#, J-block_size[1]):
    for k in range(data.shape[-2] - block_size[-2]):#0, , K-block_size[2]):

                # print("Current block processed", k)
                data_subset = data[..., k:k+block_size[2], :] #i+I, j:j+J] #, k:k+K]
                mask_subset = mask[..., k:k+block_size[2]] #i+I, j:j+J] #, k:k+K]

                for downscaling in [1]:#, 0.75, 0.5]:
                    data_subset = processer(data_subset, mask_subset, block_size, overlap, D, param_alpha, downscaling)

                # print(data_denoised.shape, data_denoised[..., k:k+block_size[2], :].shape, data_subset.shape)
                # print(k, k+block_size[2])
                data_denoised[..., k:k+block_size[2], :] += data_subset #, k:k+K] = data_subset
                divider[..., k:k+block_size[2], :] += np.ones_like(data_subset, dtype=np.int16)

    print("temps fit :", time()-deb)
    print("data_out", data_denoised.min(), data_denoised.max())
    return (data_denoised.astype(dtype)/divider)[:orig_shape[0], :orig_shape[1], :orig_shape[2]]

  #   while True:
  #               # i,j,k = (0,0,0)
  #               # I,J,K = data.shape[:-1]
  #               pad_shape = data.shape#[i:i+I, j:j+J, k:k+K].shape
  #               # print(pad_shape,block_size,overlap)
  #               X_full = im2col_nd(data, block_size, overlap).T
  #               X = np.asfortranarray(X_full[:, train_idx], dtype=dtype)
  #               X_full_shape = X_full.shape
  #               del X_full
  #               # bmean = np.mean(data_pad[i:i+I, j:j+J, k:k+K], axis=-1, keepdims=True)

  #               # list_groups = np.arange(0, X.shape[-1], block_size[1], dtype=np.int32)
  #               # param_alpha['list_groups'] = list_groups
  #               # param_D['lambda1'] = 2*1.2 / np.sqrt(np.prod(block_size))
  #               # alpha = spams.lasso(X, **param_alpha)
  #               # X = sparse_dot(D, alpha)
  #               # alpha = LassoCV().fit(D, X).coef_.T
  #               print(sigma2.shape, X_full_shape, X.shape, "sigma2.shape, X_full_shape, X.shape")

  #               # We want lambda_n / sqrt(n) -> 0 and
  #               # lambda_n n**(gamma-1)/2 -> infinity
  #               # alpha, res, _, _ = np.linalg.lstsq(D, X)
  #               gamma = 2
  #               beta = 0.5
  #               # 1/0
  #               var_array = im2col_nd(sigma2, block_size, overlap)[train_idx]#[:, 0]
  #               # param_alpha['lambda1'] = 0
  #               # param_alpha['W'] = np.array(1./(np.abs(alpha)**gamma + np.finfo(float).eps), dtype=dtype, order='F')
  #               # print(np.ones((D.shape[-1], X.shape[-1]), order='F', dtype=dtype).shape, var_array.shape)
  #               param_alpha['W'] = np.ones((D.shape[-1], X.shape[-1]), order='F', dtype=dtype) * beta * var_array[:, var_array.shape[1]//2]#* np.median(var_array, axis=-1)[:, None].T
  #               param_alpha['W'] = param_alpha['W'].astype(dtype, copy=False, order='F')
  #               param_alpha['L'] = int(0.5 * X.shape[0])
  #               param_alpha['lambda1'] = 1.2 / np.sqrt(np.prod(block_size))
  #               print(param_alpha.keys())
  #               print(param_alpha['D'].shape, param_alpha['W'].shape, X.shape)
  #               print(param_alpha['D'].dtype, param_alpha['W'].dtype, X.dtype)

  #               time1 = time()
  #               alpha = spams.lassoWeighted(X, **param_alpha)
  #               X = sparse_dot(D, alpha)
  #               print(alpha.shape)
  #               # del param_alpha['W']
  #               print("solving time 1 :", time()-time1)

  #               weights = 1. / np.array((np.abs(alpha) > 1e-15).sum(axis=0) + 1., dtype=dtype).squeeze()

  #               w = np.zeros(X_full_shape[1], dtype=dtype)
  #               w[train_idx] = weights
  #               X2 = np.zeros(X_full_shape, dtype=dtype)
  #               X2[:, train_idx] = X
  #               # X = X2
  #               data_out = col2im_nd(X2, block_size, pad_shape, overlap, w)
  #               del X2


  #               # local_variance = local_standard_deviation(data_out)**2
  #               # print(local_variance.shape)
  #               # var_array = im2col_nd(local_variance, block_size[:3], overlap[:3])[train_idx]#[:, 0]
  #               # param_alpha['W'] = np.ones((D.shape[-1], X.shape[-1]), order='F', dtype=dtype) * beta * var_array[:, var_array.shape[1]//2]# np.median(var_array, axis=-1)[:, None].T
  #               del param_alpha['W']
  #               param_alpha['L'] = int(0.5 * X.shape[0])
  #               param_alpha['lambda1'] = 0.75 * 1.2 / np.sqrt(np.prod(block_size))
  #               # print(param_alpha.keys())
  #               # print(param_alpha['D'].shape, param_alpha['W'].shape, X.shape)
  #               X = im2col_nd(data_out, block_size, overlap).T
  #               X = np.asfortranarray(X[:, train_idx], dtype=dtype)
  #               time1 = time()
  #               alpha = spams.lasso(X, **param_alpha)
  #               X = sparse_dot(D, alpha)
  #               print("solving time 2 :", time()-time1)

  #               weights = 1. / np.array((np.abs(alpha) > 1e-15).sum(axis=0) + 1., dtype=dtype).squeeze()
  #               w = np.zeros(X_full_shape[1], dtype=dtype)
  #               w[train_idx] = weights
  #               X2 = np.zeros(X_full_shape, dtype=dtype)
  #               X2[:, train_idx] = X
  #               data_out = col2im_nd(X2, block_size, pad_shape, overlap, w)
  #               del X2

  #               param_alpha['L'] = int(0.5 * X.shape[0])
  #               param_alpha['lambda1'] = 0.5 * 1.2 / np.sqrt(np.prod(block_size))
  #               # print(param_alpha.keys())
  #               # print(param_alpha['D'].shape, param_alpha['W'].shape, X.shape)
  #               X = im2col_nd(data_out, block_size, overlap).T
  #               X = np.asfortranarray(X[:, train_idx], dtype=dtype)
  #               time1 = time()
  #               alpha = spams.lasso(X, **param_alpha)
  #               X = sparse_dot(D, alpha)
  #               print("solving time 3 :", time()-time1)

  #               weights = 1. / np.array((np.abs(alpha) > 1e-15).sum(axis=0) + 1., dtype=dtype).squeeze()
  #               w = np.zeros(X_full_shape[1], dtype=dtype)
  #               w[train_idx] = weights
  #               X2 = np.zeros(X_full_shape, dtype=dtype)
  #               X2[:, train_idx] = X
  #               # X = X2
  #               data_out = col2im_nd(X2, block_size, pad_shape, overlap, w)
  #               del X2

  #               param_alpha['L'] = int(0.5 * X.shape[0])
  #               param_alpha['lambda1'] = 0.25 * 1.2 / np.sqrt(np.prod(block_size))
  #               # print(param_alpha.keys())
  #               # print(param_alpha['D'].shape, param_alpha['W'].shape, X.shape)
  #               X = im2col_nd(data_out, block_size, overlap).T
  #               X = np.asfortranarray(X[:, train_idx], dtype=dtype)
  #               time1 = time()
  #               alpha = spams.lasso(X, **param_alpha)
  #               X = sparse_dot(D, alpha)
  #               print("solving time 4 :", time()-time1)


  #               weights = 1. / np.array((np.abs(alpha) > 1e-15).sum(axis=0) + 1., dtype=dtype).squeeze()
  #               w = np.zeros(X_full_shape[1], dtype=dtype)
  #               w[train_idx] = weights
  #               X2 = np.zeros(X_full_shape, dtype=dtype)
  #               X2[:, train_idx] = X
  #               # X = X2
  #               data_out = col2im_nd(X2, block_size, pad_shape, overlap, w)
  #               del X2

  #               param_alpha['L'] = int(0.5 * X.shape[0])
  #               param_alpha['lambda1'] = 0.05 * 1.2 / np.sqrt(np.prod(block_size))
  #               # print(param_alpha.keys())
  #               # print(param_alpha['D'].shape, param_alpha['W'].shape, X.shape)
  #               X = im2col_nd(data_out, block_size, overlap).T
  #               X = np.asfortranarray(X[:, train_idx], dtype=dtype)
  #               time1 = time()
  #               alpha = spams.lasso(X, **param_alpha)
  #               X = sparse_dot(D, alpha)
  #               print("solving time 5 :", time()-time1)

  #               # print(alpha.shape)
  #               # del param_alpha['W']

  #               # We want lambda_n / sqrt(n) -> 0 and
  #               # lambda_n n**(gamma-1)/2 -> infinity
  #               # alpha, res, _, _ = np.linalg.lstsq(D, X)
  #               # param_alpha['W'] = np.array(1./(np.abs(alpha)**gamma + np.finfo(float).eps), dtype=dtype, order='F')
  #               # alpha = spams.lassoWeighted(X, **param_alpha)
  #               # X = sparse_dot(D, alpha)

  #               # 1/0
  #               # param_alpha['L'] = int(0.2 * X.shape[0])
  #               # print(alpha.shape)
  #               # ((alpha).mean(axis=0) - alpha.mean(axis=0)**2)
  #               # param_alpha['W'] = gprime(alpha) #np.ones(alpha.shape, dtype=np.float64, order='F') * ((alpha**2).mean(axis=0) - alpha.mean(axis=0)**2)

  #               # alpha, res, _, _ = np.linalg.lstsq(D, X)
  #               # param_alpha['W'] = np.array(1./(np.abs(alpha)**gamma + np.finfo(float).eps), dtype=dtype, order='F')
  #               # alpha = spams.lassoWeighted(X, **param_alpha)
  #               # X = sparse_dot(D, alpha)
  #               # param_alpha['lambda1'] = 0.75*1.2 / np.sqrt(np.prod(block_size))
  #               # alpha = spams.lasso(X, **param_alpha)
  #               # X = sparse_dot(D, alpha)

  #               # # param_alpha['L'] = int(0.25 * X.shape[0])
  #               # # param_alpha['W'] = gprime(alpha)#np.ones_like(alpha, dtype=np.float64, order='F') * ((alpha**2).mean(axis=0) - alpha.mean(axis=0)**2)
  #               # param_alpha['lambda1'] = 0.5*1.2 / np.sqrt(np.prod(block_size))
  #               # alpha = spams.lasso(X, **param_alpha)
  #               # X = sparse_dot(D, alpha)

  #               # # param_alpha['L'] = int(0.3 * X.shape[0])
  #               # # param_alpha['W'] = gprime(alpha)#np.ones_like(alpha, dtype=np.float64, order='F') * ((alpha**2).mean(axis=0) - alpha.mean(axis=0)**2)
  #               # param_alpha['lambda1'] = 0.25*1.2 / np.sqrt(np.prod(block_size))
  #               # alpha = spams.lasso(X, **param_alpha)
  #               # X = sparse_dot(D, alpha)

  #               # # param_alpha['W'] = gprime(alpha)#np.ones_like(alpha, dtype=np.float64, order='F') * ((alpha**2).mean(axis=0) - alpha.mean(axis=0)**2)
  #               # param_alpha['lambda1'] = 0.05*1.2 / np.sqrt(np.prod(block_size))
  #               # alpha = spams.lasso(X, **param_alpha)
  #               # X = sparse_dot(D, alpha)



  #               # param_D['lambda1'] = 0.01*1.2 / np.sqrt(np.prod(block_size))
  #               # alpha = spams.lasso(X, **param_alpha)
  #               # X = sparse_dot(D, alpha)
  #               # param_alpha['L'] = int(0.35 * X.shape[0])
  #               # param_D['lambda1'] = 0.05*1.2 / np.sqrt(np.prod(block_size))
  #               # alpha = spams.lasso(X, **param_alpha)
  #               # X = sparse_dot(D, alpha)
  #               # param_D['lambda1'] = 3*.012 / np.sqrt(np.prod(block_size))
  #               # alpha = spams.lasso(X, **param_alpha)
  #               # X = sparse_dot(D, alpha)


  #               # param_D['lambda1'] = 3*.012 / np.sqrt(np.prod(block_size))
  #               # alpha = spams.lasso(X, **param_alpha)
  #               # X = sparse_dot(D, alpha)


  #               # param_D['lambda1'] = 3*.012 / np.sqrt(np.prod(block_size))
  #               # alpha = spams.lasso(X, **param_alpha)
  #               # X = sparse_dot(D, alpha)
  #               # param_alpha['lambda1'] /= 10
  #               # alpha = spams.lasso(X, **param_alpha)
  #               # X = sparse_dot(D, alpha)

  #               # print(alpha.shape, D.T.shape, X.T.shape)
  #               # alpha = MultiTaskLassoCV(n_alphas=5, alphas=0.2*np.ones(D.shape[1]), fit_intercept=True, n_jobs=8).fit(D, X).coef_.T
  #               # print(alpha.shape, D.T.shape, X.T.shape)
  #               # print(np.sum(np.abs(alpha),0))
  #               # X = sparse_dot(D, alpha)
  #               # param_alpha['lambda1'] /= 5
  #               # alpha = spams.lasso(X, **param_alpha)
  #               # X = sparse_dot(D, alpha)

  #               # param_alpha['lambda1'] /= 2
  #               # alpha = spams.lasso(X, **param_alpha)
  #               # X = sparse_dot(D, alpha)

  #               # list_groups = np.array([0], dtype=np.int32)
  #               # list_groups = np.arange(0, X.shape[-1], overlap[0]+1, dtype=np.int32)
  #               # param_alpha['list_groups'] = list_groups
  #               # param_alpha['eps'] = 0.5*1.2 / np.sqrt(np.prod(block_size))
  #               # args = param_alpha['lambda1'], param_alpha['pos'], param_alpha['mode']
  #               # del param_alpha['lambda1'], param_alpha['pos'], param_alpha['mode']
  #               # alpha = spams.somp(X, **param_alpha)
  #               # X = sparse_dot(D, alpha)
  #               # del param_alpha['list_groups'], param_alpha['eps']
  #               # param_alpha['lambda1'], param_alpha['pos'], param_alpha['mode'] = args



  #               # print(alpha.shape, D.T.shape, X.T.shape)
  #               # alpha = MultiTaskElasticNetCV(n_alphas=5, fit_intercept=True, n_jobs=8).fit(D, X).coef_.T
  #               # print(alpha.shape, D.T.shape, X.T.shape)
  #               # print(np.sum(np.abs(alpha),0))
  #               # X = sparse_dot(D, alpha)

  #               #param_alpha['L'] = int(0.2 * X.shape[0])
  #               # param_alpha['lambda1'] = 0.6 / np.sqrt(np.prod(block_size))
  #               # # param_D['iter'] = 100
  #               # # param_alpha['D'] = np.sqrt(np.sum(X**2, axis=0, keepdims=True))
  #               # # D = spams.trainDL(X, **param_D)
  #               # alpha = spams.lasso(X, **param_alpha)
  #               # X = sparse_dot(D, alpha)


  #               weights = 1. / np.array((np.abs(alpha) > 1e-15).sum(axis=0) + 1., dtype=dtype).squeeze()
  #               #print(block_size, pad_shape, overlap, dtype)
  #               ##X=im2col_nd(data_pad[i:i+I, j:j+J, k:k+K], block_size, overlap).T
  #               # print(np.sum(np.isnan(data_out[i:i+I, j:j+J, k:k+K])))
  #               w = np.zeros(X_full_shape[1], dtype=dtype)
  #               w[train_idx] = weights
  #               X2 = np.zeros(X_full_shape, dtype=dtype)
  #               X2[:, train_idx] = X
  #               X = X2

  #               data_out = col2im_nd(X, block_size, pad_shape, overlap, w) #[i:i+I, j:j+J, k:k+K]

  #               # aa = data_out[i:i+I, j:j+J, k:k+K]
  #               # data_out[i:i+I, j:j+J, k:k+K] *= (bmean / np.mean(data_out[i:i+I, j:j+J, k:k+K], axis=-1, keepdims=True))
  #               # print(np.sum(np.isnan(data_out[i:i+I, j:j+J, k:k+K])), data_out[i:i+I, j:j+J, k:k+K].shape)

  #   # temp = np.zeros(data.shape[:-1] + (6,),dtype=np.float32)
  #   # temp[...,0]=b0
  #   # temp[..., 1:] = data_out
  #   # data_out = temp
  #   # print("removed b0s")

  #   print("temps fit :", time()-deb)
  #   print ("data_out", data_out.min(), data_out.max())
  #   return data_out.astype(dtype)[:orig_shape[0], :orig_shape[1], :orig_shape[2]]

  #   1/0

  #   alpha = spams.lasso(X, **param_alpha)
  #   print(alpha.shape)
  #   del param_alpha['mode']
  #   del param_alpha['lambda1']
  #   del param_alpha['pos']

  #   #param_alpha['eps'] = (noise_std**2) / (overlap[0]+1)
  #   #param_alpha['L'] = int(0.15*X.shape[0])

  #   #param_alpha['L'] = int(0.15 * X.shape[0])
  #   #alpha = spams.omp(X, **param_alpha)

  #   # param_alpha['mode'] = 0
  #   # param_alpha['pos'] = 0
  #   # param_alpha['lambda1'] = 0
  #   ###from sklearn.decomposition import sparse_encode
  #   ###print("shapes :", X.T.shape, D.T.shape)
  #   ###alpha = sparse_encode(X.T, D.T, n_jobs=8, algorithm='omp').T
  #   ###DL = DictionaryLearning(n_components=D.shape[0], alpha=1, n_iter=500)
  #   ####dico.set_params(transform_algorithm='omp', **kwargs)
  #   ####alpha = DL.transform(X)
  #   ###patches = np.dot(code, V)

  #   #from sklearn.linear_model import RandomizedLasso
  #   #RLasso = RandomizedLasso(n_jobs=8, fit_intercept=False, normalize=False)
  #   #RLasso.fit(D, X)
  #   #alpha = RLasso.coef_

  #   #param_alpha['B'] = truc
  #   print ("Fit alpha lasso: total time = ", time()-start)
  #   #from scipy.sparse import csc_matrix
  #   #alpha = csc_matrix((alpha.shape[0], alpha.shape[1]))
  #   #start = time()
  #   ####alpha = spams.cd(X, D, alpha, mode=2, itermax=1000,
  #   ####                 lambda1=param_alpha['lambda1'])
  #   #print ("Fit alpha cd: total time = ", time()-start)

  #   #print ("Using pre-masking")
  #   #mask_data = param_alpha['B']
  #   #print (mask_data.shape, X.shape)
  #   # mask_data est en 2d et deja arrange pour juste contenir le train data

  #   #del param_alpha['B']
  #   #print (X_cut.flags, X_cut.shape, X.shape)
  #   #alpha = spams.lasso(X_cut, **param_alpha)
  #   #print ("Fit alpha : total time = ", time()-start)

  #   #X_cut_recon = sparse_dot(D, alpha)
  #   #print (X_cut_recon.shape, X[mask_data].shape)
  #   #X = np.zeros_like(X)  #X_cut_recon.shape
  #   #X[mask_data] = X_cut_recon#.ravel()
  #   #del X_cut_recon, X_cut
  #   #print (np.max(alpha), np.min(alpha))
  #   print ("sparsity alpha", alpha.nnz, np.prod(alpha.shape), alpha.nnz/np.prod(alpha.shape), alpha.shape)
  #   start = time()
  #   X = sparse_dot(D, alpha)
  #   print(np.min(X), np.max(X), np.sum(X<0), np.sum(X>0))

  #   # print(D.shape)
  #   # print(np.sum(D**2,0))
  #   # print(np.sum(D**2,0).shape)
  #   # 1/0

  #   # Test new normalisation factor
  #   #weights = np.zeros(alpha.shape[-1], dtype=np.float64)
  #   #weights = 1 / (1 + np.asarray(alpha != np.zeros((1, alpha.shape[1]))).sum(axis=0, keepdims=True))

  #   #weights = np.ones_like(weights)
  #   weights = 1. / np.array((np.abs(alpha) > 1e-15).sum(axis=0) + 1., dtype=dtype).squeeze()
  #   # weights = np.ones_like(weights)
  #   print(alpha.shape, weights.shape)
  #   print(weights[len(weights)//2-10:len(weights)//2+10])
  #   #print(alpha)
  #   print((np.abs(alpha) > 1e-7).sum(axis=0))
  # #  1/0
  #   #print("weights", weights.shape)
  #   #for idx in range(alpha.shape[-1]):
  #   #    weights[idx] = len(alpha[:, idx].data)

  #   #plt.figure()
  #   #plt.hist(alpha[alpha>0].todense().T, 256)
  #   #plt.plot(np.sort(alpha[alpha>0].todense(), axis=None))
  #   #plt.savefig(savename+'_histo_alpha.png')
  #   #plt.clf()
  #   #plt.hist(X[X>0].T, 256)
  #   #plt.plot(np.sort(X[X>0], axis=None))
  #   #plt.savefig(savename+'_histo_X.png')

  #   #1/0
  #   #X = X_normalised    #################################################################################
  #   #1/0
  #   print ("X", X.min(), X.max())
  # #  print ("Diff", np.sum(np.abs(X-xorig)))

  #   #1/0
  #   #del D, alpha
  #   print ("mult sparse : total time = ", time()-start)

  #   #print ("Objective value = ", R, np.min(X), np.max(X))
  #   del param_D, D, param_alpha, alpha

  #   if whitening:
  #       print ("Inverting ZCA")
  #       X_recon = np.dot(np.linalg.inv(ZCA), X*X_norm2) + X_mean
  #       #np.dot(U * np.sqrt(s + epsilon), U.T) + X_mean
  #   else:
  #       print("Inverting normalisation")
  #       X_recon = (X * X_norm2 * X_std) + X_mean

  #   #X_recon[param_alpha['B']==0] = xorig[param_alpha['B']==0]
  #  # print ("Diff", np.sum(np.abs((X_recon-xorig)[mask_data.astype('bool')])))
  #   #print(X_recon.min(), X_recon.max(), xorig.min(), xorig.max())
  #   #1/0
  #   deb = time()
  #   print (X_recon.shape, (block_size), orig_shape, overlap)
  #   X_deblock = col2im_nd(X_recon, (block_size), orig_shape, overlap, weights)
  #   print(X_deblock.shape)
  #   # X_deblock = X_deblock[:orig_shape[0], :orig_shape[1],
  #   #                       :orig_shape[2], :orig_shape[3]]
  #   print ("temps deblock :", time()-deb)
  #   print ("Max", X_deblock.max(), "Min", X_deblock.min())
  #   print(X_deblock.shape)
  #   #print ("Somme des negatifs", np.sum(X_deblock < 0))

  #   # Removing negligible values
  #   #print ("Stuff below zero", np.sum(X_deblock < 0),
  #   #       np.sum(X_deblock[X_deblock < 0]))
  #   #print (np.min(X_deblock), np.max(X_deblock[X_deblock<0]))
  #   #X_deblock[X_deblock < 10**-8] = 0
  #   #print ("Stuff below zero, after thresholding", np.sum(X_deblock < 0),
  #   #       np.sum(X_deblock[X_deblock < 0]))




  #   # print(X_recon[:,::3].shape, X_recon[:,1::3].shape, X_recon[:,2::3].shape, X_recon.shape)
  #   # print("block_size", block_size, orig_shape,X_recon[:,:-1:3].shape)
  #   # o1 = col2im_nd(X_recon[:, 0::3],  (block_size),   (55,3,55,5), (0,0,0,0), weights)
  #   # o2 = col2im_nd(X_recon[:, 1::3], (block_size),   (55,3,55,5), (0,0,0,0), weights)
  #   # o3 = col2im_nd(X_recon[:, 2::3], (block_size),   (55,3,55,5), (0,0,0,0), weights)



  #   # import nibabel as nib
  #   # print(o1.shape,o2.shape, o3.shape)
  #   # nib.save(nib.Nifti1Image(o1,np.eye(4)),'o1.nii.gz')
  #   # nib.save(nib.Nifti1Image(o2,np.eye(4)),'o2.nii.gz')
  #   # nib.save(nib.Nifti1Image(o3,np.eye(4)),'o3.nii.gz')
  #   # nib.save(nib.Nifti1Image(X_deblock,np.eye(4)),'X_deblock.nii.gz')

  #   # 1/0


  #   # import nibabel as nib
  #   # nib.save(nib.Nifti1Image(X_deblock,np.eye(4)),'X_deblock.nii.gz')
  #   # 1/0
  #   for i in range(block_size[-1]):
  #       X_deblock[..., i] += block_mean[i]

  #   # temp = np.zeros(data.shape[:-1] + (6,),dtype=np.float32)
  #   # temp[...,0]=b0
  #   # temp[..., 1:] = X_deblock
  #   # X_deblock = temp

  #   return X_deblock.astype(dtype)
