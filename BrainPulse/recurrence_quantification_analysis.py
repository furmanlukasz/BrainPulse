from numba import jit
import numpy as np
from pyrqa.computation import RPComputation, RQAComputation
from pyrqa.time_series import TimeSeries, EmbeddedSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.metric import EuclideanMetric
# from pyrqa.metric import Sigmoid
# from pyrqa.metric import Cosine
from pyrqa.neighbourhood import Unthresholded,FixedRadius

def get_results(recurrence_matrix,
                minimum_diagonal_line_length,
                minimum_vertical_line_length,
                minimum_white_vertical_line_length):

    number_of_vectors = recurrence_matrix.shape[0]
    diagonal = diagonal_frequency_distribution(recurrence_matrix)
    vertical = vertical_frequency_distribution(recurrence_matrix)
    white = white_vertical_frequency_distribution(recurrence_matrix)

    number_of_vert_lines = number_of_vertical_lines(vertical, minimum_vertical_line_length)
    number_of_vert_lines_points = number_of_vertical_lines_points(vertical, minimum_vertical_line_length)

    RR = recurrence_rate(recurrence_matrix)
    DET = determinism(number_of_vectors, diagonal, minimum_diagonal_line_length)
    L = average_diagonal_line_length(number_of_vectors, diagonal, minimum_diagonal_line_length)
    Lmax = longest_diagonal_line_length(number_of_vectors, diagonal)
    DIV = divergence(Lmax)
    Lentr = entropy_diagonal_lines(number_of_vectors, diagonal, minimum_diagonal_line_length)
    DET_RR = ratio_determinism_recurrence_rate(DET, RR)
    LAM = laminarity(number_of_vectors, vertical, minimum_vertical_line_length)
    V = average_vertical_line_length(number_of_vectors, vertical, minimum_vertical_line_length)
    Vmax = longest_vertical_line_length(number_of_vectors, vertical)
    Ventr = entropy_vertical_lines(number_of_vectors, vertical, minimum_vertical_line_length)
    LAM_DET = laminarity_determinism(LAM, DET)
    W = average_white_vertical_line_length(number_of_vectors, white, minimum_white_vertical_line_length)
    Wmax = longest_white_vertical_line_length(number_of_vectors, white)
    Wentr = entropy_white_vertical_lines(number_of_vectors, white, minimum_white_vertical_line_length)
    TT = trapping_time(number_of_vert_lines_points, number_of_vert_lines)

    return [RR, DET, L, Lmax, DIV, Lentr, DET_RR, LAM, V, Vmax, Ventr, LAM_DET, W, Wmax, Wentr, TT]


@jit(nopython=True)
def diagonal_frequency_distribution(recurrence_matrix):
    # Calculating the number of states - N
    number_of_vectors = recurrence_matrix.shape[0]
    diagonal_frequency_distribution = np.zeros(number_of_vectors + 1)

    # Calculating the diagonal frequency distribution - P(l)
    for i in range(number_of_vectors - 1, -1, -1):
        diagonal_line_length = 0
        for j in range(0, number_of_vectors - i):
            if recurrence_matrix[i + j, j] == 1:
                diagonal_line_length += 1
                if j == (number_of_vectors - i - 1):
                    diagonal_frequency_distribution[diagonal_line_length] += 1.0
            else:
                if diagonal_line_length != 0:
                    diagonal_frequency_distribution[diagonal_line_length] += 1.0
                    diagonal_line_length = 0
    for k in range(1, number_of_vectors):
        diagonal_line_length = 0
        for i in range(number_of_vectors - k):
            j = i + k
            if recurrence_matrix[i, j] == 1:
                diagonal_line_length += 1
                if j == (number_of_vectors - 1):
                    diagonal_frequency_distribution[diagonal_line_length] += 1.0
            else:
                if diagonal_line_length != 0:
                    diagonal_frequency_distribution[diagonal_line_length] += 1.0
                    diagonal_line_length = 0

    return diagonal_frequency_distribution


@jit(nopython=True)
def vertical_frequency_distribution(recurrence_matrix):
    number_of_vectors = recurrence_matrix.shape[0]

    # Calculating the vertical frequency distribution - P(v)
    vertical_frequency_distribution = np.zeros(number_of_vectors + 1)
    for i in range(number_of_vectors):
        vertical_line_length = 0
        for j in range(number_of_vectors):
            if recurrence_matrix[i, j] == 1:
                vertical_line_length += 1
                if j == (number_of_vectors - 1):
                    vertical_frequency_distribution[vertical_line_length] += 1.0
            else:
                if vertical_line_length != 0:
                    vertical_frequency_distribution[vertical_line_length] += 1.0
                    vertical_line_length = 0

    return vertical_frequency_distribution


@jit(nopython=True)
def white_vertical_frequency_distribution(recurrence_matrix):
    number_of_vectors = recurrence_matrix.shape[0]

    # Calculating the white vertical frequency distribution - P(w)
    white_vertical_frequency_distribution = np.zeros(number_of_vectors + 1)
    for i in range(number_of_vectors):
        white_vertical_line_length = 0
        for j in range(number_of_vectors):
            if recurrence_matrix[i, j] == 0:
                white_vertical_line_length += 1
                if j == (number_of_vectors - 1):
                    white_vertical_frequency_distribution[white_vertical_line_length] += 1.0
            else:
                if white_vertical_line_length != 0:
                    white_vertical_frequency_distribution[white_vertical_line_length] += 1.0
                    white_vertical_line_length = 0

    return white_vertical_frequency_distribution


@jit(nopython=True)
def recurrence_rate(recurrence_matrix):
    # Calculating the recurrence rate - RR
    number_of_vectors = recurrence_matrix.shape[0]
    return np.float(np.sum(recurrence_matrix)) / np.power(number_of_vectors, 2)


def determinism(number_of_vectors, diagonal_frequency_distribution_, minimum_diagonal_line_length):
    # Calculating the determinism - DET
    numerator = np.sum(
        [l * diagonal_frequency_distribution_[l] for l in range(minimum_diagonal_line_length, number_of_vectors)])
    denominator = np.sum([l * diagonal_frequency_distribution_[l] for l in range(1, number_of_vectors)])
    return numerator / denominator


def average_diagonal_line_length(number_of_vectors, diagonal_frequency_distribution_, minimum_diagonal_line_length):
    # Calculating the average diagonal line length - L
    numerator = np.sum(
        [l * diagonal_frequency_distribution_[l] for l in range(minimum_diagonal_line_length, number_of_vectors)])
    denominator = np.sum(
        [diagonal_frequency_distribution_[l] for l in range(minimum_diagonal_line_length, number_of_vectors)])
    return numerator / denominator


@jit(nopython=True)
def longest_diagonal_line_length(number_of_vectors, diagonal_frequency_distribution_):
    # Calculating the longest diagonal line length - Lmax
    for l in range(number_of_vectors - 1, 0, -1):
        if diagonal_frequency_distribution_[l] != 0:
            longest_diagonal_line_length = l
            break
    return longest_diagonal_line_length


@jit(nopython=True)
def divergence(longest_diagonal_line_length_):
    # Calculating the  divergence - DIV
    return 1. / longest_diagonal_line_length_


@jit(nopython=True)
def entropy_diagonal_lines(number_of_vectors, diagonal_frequency_distribution_, minimum_diagonal_line_length):
    # Calculating the entropy diagonal lines - Lentr
    sum_diagonal_frequency_distribution = np.float(
        np.sum(diagonal_frequency_distribution_[minimum_diagonal_line_length:-1]))
    entropy_diagonal_lines = 0
    for l in range(minimum_diagonal_line_length, number_of_vectors):
        if diagonal_frequency_distribution_[l] != 0:
            entropy_diagonal_lines += (diagonal_frequency_distribution_[
                                           l] / sum_diagonal_frequency_distribution) * np.log(
                diagonal_frequency_distribution_[l] / sum_diagonal_frequency_distribution)
    entropy_diagonal_lines *= -1
    return entropy_diagonal_lines


@jit(nopython=True)
def ratio_determinism_recurrence_rate(determinism_, recurrence_rate_):
    # Calculating the  divergence - DIV
    return determinism_ / recurrence_rate_


def laminarity(number_of_vectors, vertical_frequency_distribution_, minimum_vertical_line_length):
    # Calculating the laminarity - LAM
    numerator = np.sum(
        [v * vertical_frequency_distribution_[v] for v in range(minimum_vertical_line_length, number_of_vectors + 1)])
    denominator = np.sum([v * vertical_frequency_distribution_[v] for v in range(1, number_of_vectors + 1)])
    return numerator / denominator


def average_vertical_line_length(number_of_vectors, vertical_frequency_distribution_, minimum_vertical_line_length):
    # Calculating the average vertical line length - V
    numerator = np.sum(
        [v * vertical_frequency_distribution_[v] for v in range(minimum_vertical_line_length, number_of_vectors + 1)])
    denominator = np.sum(
        [vertical_frequency_distribution_[v] for v in range(minimum_vertical_line_length, number_of_vectors + 1)])
    return numerator / denominator


@jit(nopython=True)
def longest_vertical_line_length(number_of_vectors, vertical_frequency_distribution_):
    # Calculating the longest vertical line length - Vmax
    longest_vertical_line_length_ = 0
    for v in range(number_of_vectors, 0, -1):
        if vertical_frequency_distribution_[v] != 0:
            longest_vertical_line_length_ = v
            break
    return longest_vertical_line_length_


@jit(nopython=True)
def entropy_vertical_lines(number_of_vectors, vertical_frequency_distribution_, minimum_vertical_line_length):
    # Calculating the entropy vertical lines - Ventr
    sum_vertical_frequency_distribution = np.float(
        np.sum(vertical_frequency_distribution_[minimum_vertical_line_length:]))
    entropy_vertical_lines_ = 0
    for v in range(minimum_vertical_line_length, number_of_vectors + 1):
        if vertical_frequency_distribution_[v] != 0:
            entropy_vertical_lines_ += (vertical_frequency_distribution_[
                                           v] / sum_vertical_frequency_distribution) * np.log(
                vertical_frequency_distribution_[v] / sum_vertical_frequency_distribution)
    entropy_vertical_lines_ *= -1
    return entropy_vertical_lines_


@jit(nopython=True)
def laminarity_determinism(laminarity_, determinism_):
    # Calculating the ratio laminarity_determinism - LAM/DET
    return laminarity_ / determinism_


def average_white_vertical_line_length(number_of_vectors, white_vertical_frequency_distribution_,
                                       minimum_white_vertical_line_length):
    # Calculating the average white vertical line length - W
    numerator = np.sum([w * white_vertical_frequency_distribution_[w] for w in
                        range(minimum_white_vertical_line_length, number_of_vectors + 1)])
    denominator = np.sum([white_vertical_frequency_distribution_[w] for w in
                          range(minimum_white_vertical_line_length, number_of_vectors + 1)])
    return numerator / denominator


@jit(nopython=True)
def longest_white_vertical_line_length(number_of_vectors, white_vertical_frequency_distribution_):
    # Calculating the longest white vertical line length - Wmax
    longest_white_vertical_line_length_ = 0
    for w in range(number_of_vectors, 0, -1):
        if white_vertical_frequency_distribution_[w] != 0:
            longest_white_vertical_line_length_ = w
            break
    return longest_white_vertical_line_length_


@jit(nopython=True)
def entropy_white_vertical_lines(number_of_vectors, white_vertical_frequency_distribution_,
                                 minimum_white_vertical_line_length):
    # Calculating the entropy white vertical lines - Wentr
    sum_white_vertical_frequency_distribution = np.float(
        np.sum(white_vertical_frequency_distribution_[minimum_white_vertical_line_length:]))
    entropy_white_vertical_lines_ = 0
    for w in range(minimum_white_vertical_line_length, number_of_vectors + 1):
        if white_vertical_frequency_distribution_[w] != 0:
            entropy_white_vertical_lines_ += (white_vertical_frequency_distribution_[
                                                  w] / sum_white_vertical_frequency_distribution) * np.log(
                white_vertical_frequency_distribution_[w] / sum_white_vertical_frequency_distribution)
    entropy_white_vertical_lines_ *= -1
    return entropy_white_vertical_lines_

def number_of_vertical_lines(vertical_frequency_distribution_, minimum_vertical_line_length):
    if minimum_vertical_line_length > 0:
        return np.sum(vertical_frequency_distribution_[minimum_vertical_line_length - 1:])

    return np.uint(0)


def number_of_vertical_lines_points(vertical_frequency_distribution_, minimum_vertical_line_length):
    if minimum_vertical_line_length > 0:
        return np.sum(
            ((np.arange(vertical_frequency_distribution_.size) + 1) * vertical_frequency_distribution_)[minimum_vertical_line_length - 1:])

    return np.uint(0)

@jit(nopython=True)
def trapping_time(number_of_vertical_lines_points_, number_of_vertical_lines_):
    """
    Trapping time (TT).
    """
    try:
        return np.float32(number_of_vertical_lines_points_ / number_of_vertical_lines_)
    except:
        return 0





def return_pyRQA_results(signal, nbr):
    time_series = EmbeddedSeries(signal)
    settings = Settings(time_series,
                        analysis_type=Classic,
                        neighbourhood=FixedRadius(nbr),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=1)
    computation = RQAComputation.create(settings,
                                        verbose=True)
    result = computation.run()
    return result
