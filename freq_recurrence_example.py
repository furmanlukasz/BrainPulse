from BrainPulse import (dataset,
                        vector_space,
                        distance_matrix,
                        recurrence_quantification_analysis,
                        features_space,
                        frequency_recurrence)


selected_subject = 1
t_start = 0
t_end = 30
fir_filter = [2.0, 50.0]
min_vert_line_len = 30
min_diagonal_line_len = 2
min_white_vert_line_len = 2
cut_freq = 60
win_len = 240
n_fft = 512
percentile = 22
electrode_name = 'O2'
crop=750

info_args = {"selected_subject":selected_subject,
             "win_len":win_len,"n_fft":n_fft,
             "eps":percentile, "electrode_name":electrode_name}

epochs, raw = dataset.eegbci_data(tmin=t_start, tmax=t_end,
                                  subject=selected_subject,
                                  filter_range=fir_filter)


s_rate = epochs.info['sfreq']
electrode_index = epochs.ch_names.index(electrode_name)

electrode_open = epochs.get_data()[0][electrode_index] * 1e6
electrode_close = epochs.get_data()[1][electrode_index] * 1e6


stft_open = vector_space.compute_stft((electrode_open),
                                      n_fft=n_fft, win_len=win_len,
                                      s_rate=epochs.info['sfreq'],
                                      cut_freq=cut_freq)

stft_close = vector_space.compute_stft((electrode_close),
                                       n_fft=n_fft, win_len=win_len,
                                       s_rate=epochs.info['sfreq'],
                                       cut_freq=cut_freq)


color_matrix,raw_c_matrix=frequency_recurrence.freqRP(stft_open,cut_freq,crop=1000)
frequency_recurrence.plot_freqRP_interactive(color_matrix, raw_c_matrix)