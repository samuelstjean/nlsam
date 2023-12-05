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
window = 7**3
ngrads = 50
means = 1000 # np.random.randint(250, 750, size=[voxels, 1, ngrads])
mus = means * np.ones([voxels, 1, ngrads])
sigma = 18
N = 4

data = np.zeros((voxels, window, ngrads))
for _ in range(N):
    noise = np.random.normal(loc=0, scale=sigma, size=(voxels, window, ngrads, 2))
    data += ((mus + noise[..., 0])**2 + (noise[..., 1])**2)

data[:] = np.sqrt(data)
# data[:] = data**2

# mus_approx = np.sqrt(data / N).mean(axis=1, keepdims=True)
mus_approx = np.mean(data, axis=1, keepdims=True) / np.sqrt(N)
nchi2 = np.sum(data**2, axis=-1) / sigma**2
K = 2*N * ngrads
lbda = N * np.sum(mus_approx**2, axis=-1).squeeze() / sigma**2 # only N since half of the components are zero mean

print(f'K = {K}, lbda = {lbda}')
print(np.mean(nchi2, axis=-1), K + lbda, (K + lbda) / np.mean(nchi2, axis=-1))
print(np.var(nchi2, axis=-1), 2 * (K + 2*lbda), 2 * (K + 2*lbda) / np.var(nchi2, axis=-1))


# from scipy.stats import ncx2

# params1 = ncx2.fit(nchi2.ravel(), method='MLE')
# params2 = ncx2.fit(nchi2.ravel(), method='MM')

# If you have \(k\) independent normal random variables \(X_1, X_2, \ldots, X_k\) with arbitrary means \(\mu_1, \mu_2, \ldots, \mu_k\) and a common variance \(\sigma^2\), the resulting sum of squares \(Y = X_1^2 + X_2^2 + \ldots + X_k^2\) follows a noncentral chi-squared distribution with \(k\) degrees of freedom and a noncentrality parameter \(\lambda\), where:\[ \lambda = \frac{\mu_1^2}{\sigma^2} + \frac{\mu_2^2}{\sigma^2} + \ldots + \frac{\mu_k^2}{\sigma^2} \]The moments of this noncentral chi-squared distribution can be expressed in terms of the moments of the underlying normal distributions. The first few moments are:1. **Mean (First Moment):**   \[ E(Y) = k\sigma^2 + \lambda \]2. **Variance (Second Central Moment):**   \[ \text{Var}(Y) = 2(k + 2\lambda)\sigma^4 \]3. **Skewness (Third Standardized Moment):**   \[ \text{Skew}(Y) = \frac{2\sqrt{2}\lambda^{3/2}}{(k\sigma^2 + 3\lambda)^{3/2}} \]4. **Kurtosis (Fourth Standardized Moment):**   \[ \text{Kurt}(Y) = \frac{24\lambda^2 + 48\lambda(k\sigma^2 + \lambda) + (k^2\sigma^4 + 12k\sigma^2 + 6\lambda^2)}{(k\sigma^2 + 4\lambda)(k\sigma^2 + 5\lambda)} - 3 \]These formulas express the moments of the noncentral chi-squared distribution in terms of the moments of the underlying normal variables, the common variance \(\sigma^2\), and the noncentrality parameter \(\lambda\). They provide a way to characterize the distribution based on the characteristics of the underlying normal variables and the shift introduced by the noncentrality parameter.

# lbda = N**2 * np.sum(np.mean(data, axis=1, keepdims=True)**2, axis=-1).squeeze() / sigma**2 # only N since half of the components are zero mean

nchi_mean = np.mean(nchi2, axis=-1)
nchi_var = np.var(nchi2, axis=-1)
U = np.sum(mus_approx**2, axis=-1).squeeze()
# U = np.sum(mus**2, axis=-1).squeeze()

N_approx1 = nchi_mean / (2 * ngrads + U / sigma**2)
N_approx2 = nchi_var / (4 * (ngrads + U / sigma**2))

print(N, N_approx1, N_approx2)

# sigma_approx = -U * (4*nchi_mean - nchi_var) / (2 * (ngrads * (2*nchi_mean - nchi_var)))
S = nchi_mean
V = nchi_var
T = ngrads
# sigma_approx1 = -np.sqrt(2)/2 * np.sqrt(-U * (4*S - V) / (T * (2*S - V)))
# sigma_approx2 = np.sqrt(2)/2 * np.sqrt(U * (-4*S + V) / (T * (2*S - V)))


sigma_approx1 = np.sqrt(N*U / (S - 2*N*T))
sigma_approx2 = np.sqrt(4*N*U / (V - 4*N*T))

print(sigma, sigma_approx1, sigma_approx2)
