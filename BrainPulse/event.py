"""Event segmentation using a Hidden Markov Model

Adapted from the brainiak package for this workshop.
See https://brainiak.org/ for full documentation."""

import numpy as np
from scipy import stats
import logging
import copy
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.exceptions import NotFittedError
import itertools


def masked_log(x):
    y = np.empty(x.shape, dtype=x.dtype)
    lim = x.shape[0]
    for i in range(lim):
        if x[i] <= 0:
            y[i] = float('-inf')
        else:
            y[i] = np.log(x[i])
    return y


class EventSegment(BaseEstimator):

    def _default_var_schedule(step):
        return 4 * (0.98 ** (step - 1))

    def __init__(self, n_events=2,
                 step_var=_default_var_schedule,
                 n_iter=500, event_chains=None,
                 split_merge=False, split_merge_proposals=1):
        self.n_events = n_events
        self.step_var = step_var
        self.n_iter = n_iter
        self.split_merge = split_merge
        self.split_merge_proposals = split_merge_proposals
        if event_chains is None:
            self.event_chains = np.zeros(n_events)
        else:
            self.event_chains = event_chains

    def _fit_validate(self, X):
        if len(np.unique(self.event_chains)) > 1:
            raise RuntimeError("Cannot fit chains, use set_event_patterns")

        # Copy X into a list and transpose
        X = copy.deepcopy(X)
        if type(X) is not list:
            X = [X]
        for i in range(len(X)):
            X[i] = check_array(X[i])
            X[i] = X[i].T

        # Check that number of voxels is consistent across datasets
        n_dim = X[0].shape[0]
        for i in range(len(X)):
            assert (X[i].shape[0] == n_dim)

        # Double-check that data is z-scored in time
        for i in range(len(X)):
            X[i] = stats.zscore(X[i], axis=1, ddof=1)

        return X

    def fit(self, X, y=None):
        seg = []
        X = self._fit_validate(X)
        n_train = len(X)
        n_dim = X[0].shape[0]
        self.classes_ = np.arange(self.n_events)

        # Initialize variables for fitting
        log_gamma = []
        for i in range(n_train):
            log_gamma.append(np.zeros((X[i].shape[1], self.n_events)))
        step = 1
        best_ll = float("-inf")
        self.ll_ = np.empty((0, n_train))
        while step <= self.n_iter:
            iteration_var = self.step_var(step)

            # Based on the current segmentation, compute the mean pattern
            # for each event
            seg_prob = [np.exp(lg) / np.sum(np.exp(lg), axis=0)
                        for lg in log_gamma]
            mean_pat = np.empty((n_train, n_dim, self.n_events))
            for i in range(n_train):
                mean_pat[i, :, :] = X[i].dot(seg_prob[i])
            mean_pat = np.mean(mean_pat, axis=0)

            # Based on the current mean patterns, compute the event
            # segmentation
            self.ll_ = np.append(self.ll_, np.empty((1, n_train)), axis=0)
            for i in range(n_train):
                logprob = self._logprob_obs(X[i], mean_pat, iteration_var)
                log_gamma[i], self.ll_[-1, i] = self._forward_backward(logprob)

            if step > 1 and self.split_merge:
                curr_ll = np.mean(self.ll_[-1, :])
                self.ll_[-1, :], log_gamma, mean_pat = \
                    self._split_merge(X, log_gamma, iteration_var, curr_ll)

            # If log-likelihood has started decreasing, undo last step and stop
            if np.mean(self.ll_[-1, :]) < best_ll:
                self.ll_ = self.ll_[:-1, :]
                break

            self.segments_ = [np.exp(lg) for lg in log_gamma]
            self.event_var_ = iteration_var
            self.event_pat_ = mean_pat
            best_ll = np.mean(self.ll_[-1, :])
            
            seg.append(self.segments_[0].copy())

            step += 1

        return seg

    def _logprob_obs(self, data, mean_pat, var):
        n_vox = data.shape[0]
        t = data.shape[1]

        # z-score both data and mean patterns in space, so that Gaussians
        # are measuring Pearson correlations and are insensitive to overall
        # activity changes
        data_z = stats.zscore(data, axis=0, ddof=1)
        mean_pat_z = stats.zscore(mean_pat, axis=0, ddof=1)

        logprob = np.empty((t, self.n_events))

        if type(var) is not np.ndarray:
            var = var * np.ones(self.n_events)

        for k in range(self.n_events):
            logprob[:, k] = -0.5 * n_vox * np.log(
                2 * np.pi * var[k]) - 0.5 * np.sum(
                (data_z.T - mean_pat_z[:, k]).T ** 2, axis=0) / var[k]

        logprob /= n_vox
        return logprob

    def _forward_backward(self, logprob):
        logprob = copy.copy(logprob)
        t = logprob.shape[0]
        logprob = np.hstack((logprob, float("-inf") * np.ones((t, 1))))

        # Initialize variables
        log_scale = np.zeros(t)
        log_alpha = np.zeros((t, self.n_events + 1))
        log_beta = np.zeros((t, self.n_events + 1))

        # Set up transition matrix, with final sink state
        self.p_start = np.zeros(self.n_events + 1)
        self.p_end = np.zeros(self.n_events + 1)
        self.P = np.zeros((self.n_events + 1, self.n_events + 1))
        label_ind = np.unique(self.event_chains, return_inverse=True)[1]
        n_chains = np.max(label_ind) + 1

        # For each chain of events, link them together and then to sink state
        for c in range(n_chains):
            chain_ind = np.nonzero(label_ind == c)[0]
            self.p_start[chain_ind[0]] = 1 / n_chains
            self.p_end[chain_ind[-1]] = 1 / n_chains

            p_trans = (len(chain_ind) - 1) / t
            if p_trans >= 1:
                raise ValueError('Too few timepoints')
            for i in range(len(chain_ind)):
                self.P[chain_ind[i], chain_ind[i]] = 1 - p_trans
                if i < len(chain_ind) - 1:
                    self.P[chain_ind[i], chain_ind[i+1]] = p_trans
                else:
                    self.P[chain_ind[i], -1] = p_trans
        self.P[-1, -1] = 1

        # Forward pass
        for i in range(t):
            if i == 0:
                log_alpha[0, :] = self._log(self.p_start) + logprob[0, :]
            else:
                log_alpha[i, :] = self._log(np.exp(log_alpha[i - 1, :])
                                            .dot(self.P)) + logprob[i, :]

            log_scale[i] = np.logaddexp.reduce(log_alpha[i, :])
            log_alpha[i] -= log_scale[i]

        # Backward pass
        log_beta[-1, :] = self._log(self.p_end) - log_scale[-1]
        for i in reversed(range(t - 1)):
            obs_weighted = log_beta[i + 1, :] + logprob[i + 1, :]
            offset = np.max(obs_weighted)
            log_beta[i, :] = offset + self._log(
                np.exp(obs_weighted - offset).dot(self.P.T)) - log_scale[i]

        # Combine and normalize
        log_gamma = log_alpha + log_beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)

        ll = np.sum(log_scale[:(t - 1)]) + np.logaddexp.reduce(
            log_alpha[-1, :] + log_scale[-1] + self._log(self.p_end))

        log_gamma = log_gamma[:, :-1]

        return log_gamma, ll

    def _log(self, x):
        xshape = x.shape
        _x = x.flatten()
        y = masked_log(_x)
        return y.reshape(xshape)

    def set_event_patterns(self, event_pat):
        if event_pat.shape[1] != self.n_events:
            raise ValueError(("Number of columns of event_pat must match "
                              "number of events"))
        self.event_pat_ = event_pat.copy()

    def find_events(self, testing_data, var=None, scramble=False):
        if var is None:
            if not hasattr(self, 'event_var_'):
                raise NotFittedError(("Event variance must be provided, if "
                                      "not previously set by fit()"))
            else:
                var = self.event_var_

        if not hasattr(self, 'event_pat_'):
            raise NotFittedError(("The event patterns must first be set "
                                  "by fit() or set_event_patterns()"))
        if scramble:
            mean_pat = self.event_pat_[:, np.random.permutation(self.n_events)]
        else:
            mean_pat = self.event_pat_

        logprob = self._logprob_obs(testing_data.T, mean_pat, var)
        lg, test_ll = self._forward_backward(logprob)
        segments = np.exp(lg)

        return segments, test_ll

    def predict(self, X):
        check_is_fitted(self, ["event_pat_", "event_var_"])
        X = check_array(X)
        segments, test_ll = self.find_events(X)
        return np.argmax(segments, axis=1)

    def calc_weighted_event_var(self, D, weights, event_pat):
        Dz = stats.zscore(D, axis=1, ddof=1)
        ev_var = np.empty(event_pat.shape[1])
        for e in range(event_pat.shape[1]):
            # Only compute variances for weights > 0.1% of max weight
            nz = weights[:, e] > np.max(weights[:, e])/1000
            sumsq = np.dot(weights[nz, e],
                           np.sum(np.square(Dz[nz, :] -
                                  event_pat[:, e]), axis=1))
            ev_var[e] = sumsq/(np.sum(weights[nz, e]) -
                               np.sum(np.square(weights[nz, e])) /
                               np.sum(weights[nz, e]))
        ev_var = ev_var / D.shape[1]
        return ev_var

    def model_prior(self, t):
        lg, test_ll = self._forward_backward(np.zeros((t, self.n_events)))
        segments = np.exp(lg)

        return segments, test_ll

    def _split_merge(self, X, log_gamma, iteration_var, curr_ll):
        # Compute current probabilities and mean patterns
        n_train = len(X)
        n_dim = X[0].shape[0]

        seg_prob = [np.exp(lg) / np.sum(np.exp(lg), axis=0)
                    for lg in log_gamma]
        mean_pat = np.empty((n_train, n_dim, self.n_events))
        for i in range(n_train):
            mean_pat[i, :, :] = X[i].dot(seg_prob[i])
        mean_pat = np.mean(mean_pat, axis=0)

        # For each event, merge its probability distribution
        # with the next event, and also split its probability
        # distribution at its median into two separate events.
        # Use these new event probability distributions to compute
        # merged and split event patterns.
        merge_pat = np.empty((n_train, n_dim, self.n_events))
        split_pat = np.empty((n_train, n_dim, 2 * self.n_events))
        for i, sp in enumerate(seg_prob):  # Iterate over datasets
            m_evprob = np.zeros((sp.shape[0], sp.shape[1]))
            s_evprob = np.zeros((sp.shape[0], 2 * sp.shape[1]))
            cs = np.cumsum(sp, axis=0)
            for e in range(sp.shape[1]):
                # Split distribution at midpoint and normalize each half
                mid = np.where(cs[:, e] >= 0.5)[0][0]
                cs_first = cs[mid, e] - sp[mid, e]
                cs_second = 1 - cs_first
                s_evprob[:mid, 2 * e] = sp[:mid, e] / cs_first
                s_evprob[mid:, 2 * e + 1] = sp[mid:, e] / cs_second

                # Merge distribution with next event distribution
                m_evprob[:, e] = sp[:, e:(e + 2)].mean(1)

            # Weight data by distribution to get event patterns
            merge_pat[i, :, :] = X[i].dot(m_evprob)
            split_pat[i, :, :] = X[i].dot(s_evprob)

        # Average across datasets
        merge_pat = np.mean(merge_pat, axis=0)
        split_pat = np.mean(split_pat, axis=0)

        # Correlate the current event patterns with the split and
        # merged patterns
        merge_corr = np.zeros(self.n_events)
        split_corr = np.zeros(self.n_events)
        for e in range(self.n_events):
            split_corr[e] = np.corrcoef(mean_pat[:, e],
                                        split_pat[:, (2 * e):(2 * e + 2)],
                                        rowvar=False)[0, 1:3].max()
            merge_corr[e] = np.corrcoef(merge_pat[:, e],
                                        mean_pat[:, e:(e + 2)],
                                        rowvar=False)[0, 1:3].min()
        merge_corr = merge_corr[:-1]

        # Find best merge/split candidates
        # A high value of merge_corr indicates that a pair of events are
        # very similar to their merged pattern, and are good candidates for
        # being merged.
        # A low value of split_corr indicates that an event's pattern is
        # very dissimilar from the patterns in its first and second half,
        # and is a good candidate for being split.
        best_merge = np.flipud(np.argsort(merge_corr))
        best_merge = best_merge[:self.split_merge_proposals]
        best_split = np.argsort(split_corr)
        best_split = best_split[:self.split_merge_proposals]

        # For every pair of merge/split candidates, attempt the merge/split
        # and measure the log-likelihood. If any are better than curr_ll,
        # accept this best merge/split
        mean_pat_last = mean_pat.copy()
        return_ll = curr_ll
        return_lg = copy.deepcopy(log_gamma)
        return_mp = mean_pat.copy()
        for m_e, s_e in itertools.product(best_merge, best_split):
            if m_e == s_e or m_e+1 == s_e:
                # Don't attempt to merge/split same event
                continue

            # Construct new set of patterns with merge/split
            mean_pat_ms = np.delete(mean_pat_last, s_e, axis=1)
            mean_pat_ms = np.insert(mean_pat_ms, [s_e, s_e],
                                    split_pat[:, (2 * s_e):(2 * s_e + 2)],
                                    axis=1)
            mean_pat_ms = np.delete(mean_pat_ms,
                                    [m_e + (s_e < m_e), m_e + (s_e < m_e) + 1],
                                    axis=1)
            mean_pat_ms = np.insert(mean_pat_ms, m_e + (s_e < m_e),
                                    merge_pat[:, m_e], axis=1)

            # Measure log-likelihood with these new patterns
            ll_ms = np.zeros(n_train)
            log_gamma_ms = list()
            for i in range(n_train):
                logprob = self._logprob_obs(X[i],
                                            mean_pat_ms, iteration_var)
                lg, ll_ms[i] = self._forward_backward(logprob)
                log_gamma_ms.append(lg)

            # If better than best ll so far, save to return to fit()
            if ll_ms.mean() > return_ll:
                return_mp = mean_pat_ms.copy()
                return_ll = ll_ms
                for i in range(n_train):
                    return_lg[i] = log_gamma_ms[i].copy()

        return return_ll, return_lg, return_mp
