import numpy as np
from scipy.special import sph_harm, factorial
from scipy.signal import correlate


def cart2sph(vecs):
    x, y, z = np.moveaxis(vecs, 1, 0)

    x_2 = x**2
    y_2 = y**2
    z_2 = z**2

    xy = np.sqrt(x_2 + y_2)

    r = np.sqrt(x_2 + y_2 + z_2)
    # in sph_harm, theta must be in the [0, 2 pi] interval
    theta = np.arctan2(y, x) + np.pi
    assert np.all((0 <= theta) & (theta <= 2 * np.pi))
    phi = np.arctan2(xy, z)
    assert np.all((0 <= phi) & (phi <= np.pi))

    return r, theta, phi


def autocorr_with_err(x):

    mu = np.mean(x)
    sigma = np.std(x)
    x = (x - mu) / sigma

    N = x.size
    counts = np.arange(N, 1, -1)

    c = correlate(x, x, mode="full")
    c = c[N - 1 : -1].real
    c_mean = c / counts

    x2 = x**2
    c2 = correlate(x2, x2, mode="full")
    c2 = c2[N - 1 : -1].real
    c2_mean = c2 / counts

    std_err = np.sqrt((c2_mean - c_mean**2) / counts)

    return c_mean, std_err


def rotacf(vecs, n=2):
    _, thetas, phis = cart2sph(vecs)

    sph_harm_list = []

    for m in range(-n, n + 1):
        # sh = sph_harm(m, n, thetas, phis) * np.sqrt(factorial(n + m) / factorial(n - m)) # * (4 * np.pi) / (2 * n + 1))
        sh = sph_harm(m, n, thetas, phis)
        sph_harm_list.append(sh)

    sph_harm_sum = sum(sph_harm_list)

    return autocorr_with_err(sph_harm_sum)
