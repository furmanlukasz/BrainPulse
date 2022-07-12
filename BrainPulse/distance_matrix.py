from numba import jit
from pynndescent.distances import wasserstein_1d, spearmanr
import numpy as np
from pyrqa.computation import RPComputation
from pyrqa.time_series import TimeSeries, EmbeddedSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.metric import EuclideanMetric
# from pyrqa.metric import Sigmoid
# from pyrqa.metric import Cosine
from pyrqa.neighbourhood import Unthresholded
import torch


#
# @jit(parallel=True)
@jit(nopython=True)
def wasserstein_squereform(stft_):
    result = np.zeros((stft_.shape[0], stft_.shape[0]))

    for i, fft in enumerate(stft_):
        for j, v in enumerate(stft_):
            d = wasserstein_1d(fft, v)
            result[i, j] = d
    return result


@jit(nopython=True)
def spearmanr_squereform(stft_):
    result = np.zeros((stft_.shape[0], stft_.shape[0]))

    for i, fft in enumerate(stft_):
        for j, v in enumerate(stft_):
            d = spearmanr(fft, v)
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


def EuclideanPyRQA_RP(signal, embedding = 2,timedelay = 9):
    time_series = TimeSeries(signal,
                             embedding_dimension=embedding,
                             time_delay=timedelay)
    settings = Settings(time_series,
                        analysis_type=Classic,
                        neighbourhood=Unthresholded(),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=1)

    computation = RPComputation.create(settings)
    result = computation.run()
    return result.recurrence_matrix


def EuclideanPyRQA_RP_stft(signal, embedding = 2,timedelay = 9):
    time_series = EmbeddedSeries(signal)
    settings = Settings(time_series,
                        analysis_type=Classic,
                        neighbourhood=Unthresholded(),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=1)

    computation = RPComputation.create(settings)
    result = computation.run()
    return result.recurrence_matrix