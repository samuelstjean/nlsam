import numpy as np

from nlsam.blocks import im2col_nd, col2im_nd, unpad, padding

def moments(data):
    # data is nsamples, local_window, nvolumes
    assert data.ndim == 3
    assert data.shape[0] > data.shape[1] > data.shape[2]

    size = data.shape[1]
    sumncx = np.sum(data**2, axis=-1)
    sample_mean = np.mean(sumncx, axis=-1)
    sample_var = np.var(sumncx, axis=-1)

    # lbda = np.sum(mu**2 / sigma**2)
    # mean = 2 * size * N + np.sum(lbda)
    # var = 4 * (size * N + np.sum(lbda))

    # sample_var = 4 * size * N + sample_mean - 2 * size * N
    N = (sample_var - sample_mean) / (2 * size)

    return sigma, N


voxels = 3
window = 10**3
ngrads = 75
means = 1000 #np.random.randint(250, 750, size=[voxels, 1, ngrads])
mus = means * np.ones([voxels, 1, ngrads])
sigma = 25
N = 25

data = np.zeros((voxels, window, ngrads))
for _ in range(N):
    noise = np.random.normal(loc=0, scale=sigma, size=(voxels, window, ngrads, 2))
    data += ((mus + noise[..., 0])**2 + (noise[..., 1])**2)

# data[:] = np.sqrt(data)
# data[:] = data**2

nchi2 = np.sum(data, axis=-1) / sigma**2
K = N * (2*N * ngrads)
lbda = N * np.sum(mus**2, axis=-1).squeeze() / sigma**2 # only N since half of the components are zero mean

print(f'K = {K}, lbda = {lbda}')
print(np.mean(nchi2, axis=-1), K + lbda, (K + lbda) / np.mean(nchi2, axis=-1))
print(np.var(nchi2, axis=-1), 2 * (K + 2*lbda), 2 * (K + 2*lbda) / np.var(nchi2, axis=-1))

# sigma = np.sqrt(1 + 2*lbda)

# np.sum(mus**2, axis=-1).squeeze() * N , np.mean(nchi2, axis=-1)

# from scipy.stats import ncx2

# params1 = ncx2.fit(nchi2.ravel(), method='MLE')
# params2 = ncx2.fit(nchi2.ravel(), method='MM')
