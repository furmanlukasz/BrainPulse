from numba import jit
import numpy as np
from pynndescent.distances import wasserstein_1d
import numpy as np


@jit(nopython=True)
def wasserstein_squereform(stft_):
    result = np.zeros((stft_.shape[0], stft_.shape[0]))

    for i, fft in enumerate(stft_):
        for j, v in enumerate(stft_):
            d = wasserstein_1d(fft, v)
            result[i, j] = d
    return result


@jit(nopython=True)
def wasserstein_squereform_binary(stft_, eps_):
    result = np.zeros((stft_.shape[0], stft_.shape[0]))

    for i, fft in enumerate(stft_):
        for j, v in enumerate(stft_):
            d = wasserstein_1d(fft, v)
            if d <= eps_:
                dist = 1
            else:
                dist = 0

            result[i, j] = dist
    return result


@jit(nopython=True)
def wasserstein_1d_array(stft_):
    result = np.zeros((stft_.shape[0] * stft_.shape[0]))
    k = 0
    for i, fft in enumerate(stft_):
        for j, v in enumerate(stft_):
            d = wasserstein_1d(fft, v)
            result[k] = d
            k += 1

    return result

def set_epsilon(matrix, eps):
    return np.heaviside(eps - matrix, 0)