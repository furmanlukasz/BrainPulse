import matplotlib.pyplot as plt
from BrainPulse import (dataset,
                        phase_space,
                        distance_matrix,
                        recurrence_quantification_analysis,
                        plot)


selected_subject = 1
t_start = 0
t_end = 12
fir_filter = [2.0, 50.0]
min_vert_line_len = 30
min_diagonal_line_len = 2
min_white_vert_line_len = 2
cut_freq = 60
win_len = 240
n_fft = 512
eps = 3.85

electrode_name = 'O2'
# epochs.ch_names
epochs = dataset.eegbci_data(tmin=t_start, tmax=t_end,
                             subject=selected_subject,
                             filter_range=fir_filter)

s_rate = epochs.info['sfreq']
electrode_index = epochs.ch_names.index(electrode_name)

electrode_open = epochs.get_data()[0][electrode_index] * 1e6
electrode_close = epochs.get_data()[1][electrode_index] * 1e6

stft_open = phase_space.compute_stft(electrode_open,
                                     n_fft=n_fft, win_len=win_len,
                                     s_rate=epochs.info['sfreq'],
                                     cut_freq=cut_freq)

stft_close = phase_space.compute_stft(electrode_close,
                                     n_fft=n_fft, win_len=win_len,
                                     s_rate=epochs.info['sfreq'],
                                     cut_freq=cut_freq)


# plot.explainer_(electrode_open, stft_open, cut_freq, s_rate)
# plot.explainer_(electrode_close, stft_close, cut_freq, s_rate)
#


# matrix_open = distance_matrix.wasserstein_squereform(stft_open)
matrix_close = distance_matrix.wasserstein_squereform(stft_close)


# matrix_open_binary = distance_matrix.set_epsilon(matrix_open,eps)
matrix_close_binary = distance_matrix.set_epsilon(matrix_close,eps)
#
# result_rqa_open = recurrence_quantification_analysis.get_results(matrix_open_binary, min_diagonal_line_len, min_vert_line_len, min_white_vert_line_len)
# result_rqa_close = recurrence_quantification_analysis.get_results(matrix_close_binary, min_diagonal_line_len, min_vert_line_len, min_white_vert_line_len)


info_args = {"selected_subject":selected_subject,
             "win_len":win_len,"n_fft":n_fft,
             "eps":eps, "electrode_name":electrode_name}

plot.stft_collections(matrix=matrix_close, matrix_binary=matrix_close_binary, stft=stft_close, cut_freq=cut_freq, s_rate=epochs.info['sfreq'], task='closed eyes', info_args=info_args)

# plot.diagnostic2(matrix=matrix_close, matrix_binary=matrix_close_binary, stft=stft_close, cut_freq=cut_freq, s_rate=epochs.info['sfreq'], task='closed eyes', info_args=info_args)
# plot.diagnostic(matrix=matrix_close, matrix_binary=matrix_close_binary, stft=stft_close, cut_freq=cut_freq, s_rate=epochs.info['sfreq'], task='closed eyes', info_args=info_args)

#
# plot.diagnostic(matrix=matrix_close, matrix_binary=matrix_close_binary, stft=stft_close, cut_freq=cut_freq, s_rate=epochs.info['sfreq'], task='closed eyes', info_args=info_args)
# plot.diagnostic(matrix=matrix_open, matrix_binary=matrix_open_binary, stft=stft_open, cut_freq=cut_freq, s_rate=epochs.info['sfreq'], task='open eyes', info_args=info_args)






# plt.figure()
# plt.imshow(stft_open.T, aspect='auto', origin='lower')
# plt.figure()
# plt.imshow(stft_close.T, aspect='auto', origin='lower')

# plt.figure()
# plt.imshow(matrix_open, origin='lower')
# plt.figure()
# plt.imshow(matrix_close, origin='lower')
#
#
# epochs.plot_psd(average=True)
# epochs.plot_psd()