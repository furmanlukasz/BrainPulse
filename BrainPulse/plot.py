import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# def diagnostic(matrix, matrix_binary, s_rate, stft, cut_freq, task, info_args):
#
#     # fig, axs = plt.subplots(3,1, figsize=(4,8), gridspec_kw={'height_ratios':[6,2,1]},dpi=120)
#
#     # Set up the axes with gridspec
#     fig = plt.figure(figsize=(6, 6),dpi=120)
#     grid = plt.GridSpec(6, 6, hspace=1.0, wspace=1.0)
#     spectrogram = fig.add_subplot(grid[0:3, 0:3])
#     rp_plot = fig.add_subplot(grid[0:3, 3:])
#     fft_vector = fig.add_subplot(grid[3:,:])
#
#     max_array = np.max(stft, axis=1)
#     max_value_stft = np.max(max_array, axis=0)
#     max_index =  list(max_array).index(max_value_stft)
#
#     min_array = np.min(stft, axis=1)
#     min_value_stft = np.min(min_array, axis=0)
#     min_index = list(min_array).index(min_value_stft)
#
#     # top = np.triu(matrix)
#     # bottom = np.tril(matrix_binary)
#
#     rp_plot.imshow(matrix_binary, cmap='cividis', origin='lower') #interpolation='none'
#     # axs[0].imshow(bottom, cmap='jet', origin='lower') #interpolation='none'
#     rp_plot.plot(max_index,max_index,'orange',marker="o", markersize=7)
#     rp_plot.plot(min_index,min_index,'red',marker="o", markersize=7)
#     # axs[0].set_yticks(axs[0].get_yticks()[1:len(axs[0].get_yticks())-1])
#     # axs[0].set_xticks(axs[0].get_xticks()[1:len(axs[0].get_xticks())-1])
#     rp_plot.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
#     rp_plot.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
#     rp_plot.set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, rp_plot.get_xticks().shape[0])])
#     rp_plot.set_yticklabels([str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, rp_plot.get_yticks().shape[0])])
#     rp_plot.set_xlabel('Time (s)')
#     rp_plot.set_ylabel('Time (s)')
#     rp_plot.set_title('Recurrence Plot', fontsize=10)
#
#
#     # np.linspace(0, stft.shape[1], stft.shape[1]),        np.linspace(0, stft.shape[0], cut_freq),
#
#
#     spectrogram.pcolormesh(stft.T, shading='gouraud') #,vmax=max_value_stft
#     spectrogram.plot(max_index,0,'orange', marker="o", markersize=7)
#     spectrogram.plot(min_index,0,'red', marker="o", markersize=7)
#     spectrogram.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
#     spectrogram.set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, spectrogram.get_xticks().shape[0])])
#     spectrogram.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft.shape[1], 5)))
#     spectrogram.set_yticklabels([str(np.round(x, 1)) for x in np.linspace(0, cut_freq, 5)])
#     spectrogram.set_ylabel('Freq (Hz)')
#     spectrogram.set_xlabel('Time (s)')
#     spectrogram.set_title('Spectrogram', fontsize=10)
#
#     max_index_ = stft[max_index]/stft.shape[1]
#     min_index_ = stft[min_index]/stft.shape[1]
#     fft_vector.plot(max_index_**2,'orange')#,marker="o", markersize=2
#     fft_vector.plot(min_index_**2,'red')#,marker="o", markersize=2
#     fft_vector.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft.shape[1], 9)))
#     fft_vector.set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, cut_freq, 9)])
#     fft_vector.set_xlim([0,100])
#     fft_vector.set_ylabel('Power (µV^2)')
#     fft_vector.set_xlabel('Freq (Hz)')
#     fft_vector.set_title('Frequency Domain', size=10)
#
#     plt.suptitle( 'Condition: '+ task  + '\n' + 'epsilon {},  FFT window size {} '.format(
#                  str(info_args['eps']), str(info_args['win_len'])) + '\n'
#                  + 'Subject {}, electrode {}, n_fft {}'.format(str(info_args['selected_subject']),str(info_args['electrode_name']),str(info_args['n_fft'])),
#                  fontsize=8,ha='left',va='top')
#     plt.tight_layout()


def explainer_(chan, stft, cut_freq, s_rate):

    fig, axs = plt.subplots(4, figsize=(7, 10), dpi=120)  # figsize=(12, 12),
    time_crop = np.linspace(0, int(chan[:400].shape[0]), chan[:400].shape[0])
    axs[0].plot(chan[:400],'k')  # np.linspace(0, int(chan[:400].shape[0]/s_rate), chan[:400].shape[0]),
    axs[0].fill_betweenx(y=[-210, 125], x1=time_crop[0],
                        x2=time_crop[240], color='white', alpha=0.7, edgecolor='red' )

    axs[0].fill_betweenx(y=[-210, 130], x1=time_crop[1],
                         x2=time_crop[241], color='white', alpha=0.7, edgecolor='green')

    axs[0].fill_betweenx(y=[-210, 135], x1=time_crop[2],
                         x2=time_crop[242], color='white', alpha=0.7, edgecolor='blue')

    axs[0].annotate('$fft_{1}$', xy=(.25, 72), xycoords='data',
                xytext=(0.05, 1.45), textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->",facecolor='blue'),
                horizontalalignment='right', verticalalignment='top',
                )

    axs[0].annotate('$fft_{2}$', xy=(.35, 75), xycoords='data',
                    xytext=(0.15, 1.45), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="->",facecolor='blue'),
                    horizontalalignment='right', verticalalignment='top',
                    )

    axs[0].annotate('$fft_{3}$', xy=(.45, 80), xycoords='data',
                    xytext=(0.25, 1.45), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="->",facecolor='blue'),
                    horizontalalignment='right', verticalalignment='top',
                    )
    axs[0].xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, chan[:400].shape[0], 5)))
    axs[0].set_xticklabels(
        [str(np.round(x, 1)) for x in np.linspace(0, int(chan[:400].shape[0] / s_rate), axs[0].get_xticks().shape[0])])
    axs[0].set_ylabel('Amplitude (µV)')
    axs[0].set_xlabel('Time (s)]')
    axs[0].set_title('Time Domain',size=10)


    axs[1].plot((stft[100]/stft.shape[1])**2, 'red',label='$fft_{1}$',marker="o",markersize=3)
    axs[1].legend()
    axs[1].xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft.shape[1], 9)))
    axs[1].set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, cut_freq, 9)])
    axs[1].set_xlim([0, 100])
    # axs[1].set_ylim([0, 250])
    axs[1].set_ylabel('Power ($\mu V^{2}$)')
    axs[1].set_xlabel('Freq (Hz)')
    axs[1].set_title('Frequency Domain ($fft_{1}$, $fft_{2}$, $fft_{3}$)', size=10)

    axs[2].plot((stft[115]/stft.shape[1])**2, 'green', label='$fft_{2}$', marker="o", markersize=3)
    axs[2].legend()
    axs[2].xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft.shape[1], 9)))
    axs[2].set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, cut_freq, 9)])
    axs[2].set_xlim([0, 100])
    # axs[2].set_ylim([0, 250])
    axs[2].set_ylabel('Power ($\mu V^{2}$)')
    axs[2].set_xlabel('Freq (Hz)')
    # axs[2].set_title('Frequency Domain', size=10)

    axs[3].plot((stft[140]/stft.shape[1])**2, 'blue', label='$fft_{3}$', marker="o", markersize=3)
    axs[3].legend()
    axs[3].xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft.shape[1], 9)))
    axs[3].set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, cut_freq, 9)])
    axs[3].set_xlim([0, 100])
    # axs[3].set_ylim([0, 250])
    axs[3].set_ylabel('Power ($\mu V^{2}$)')
    axs[3].set_xlabel('Freq (Hz)')
    # axs[3].set_title('Frequency Domain', size=10)
    plt.tight_layout()

    # axs[4].plot(np.linspace(1,50,150),signal.resample(np.square(list_stft[1].T[100]), 150), 'red', label='$fft_{1}$',marker="o",markersize=3)
    # axs[4].plot(np.linspace(1,50,150),signal.resample(np.square(list_stft[1].T[115]), 150), 'green', label='$fft_{2}$',marker="o",markersize=3)
    # axs[4].plot(np.linspace(1,50,150),signal.resample(np.square(list_stft[1].T[140]), 150), 'blue', label='$fft_{3}$',marker="o",markersize=3)
    # axs[4].legend(loc='right')
    # axs[4].set_ylabel('power [Au]')
    # axs[4].set_xlabel('frequency [Hz]')
    # axs[4].set_title('Frequency Domain resampled',size=10)

    #signal.resample(sft, 150)


def stft_collections(matrix, matrix_binary, s_rate, stft, cut_freq, task, info_args):
    fig = plt.figure(figsize=(6, 6), dpi=140)
    grid = plt.GridSpec(6, 7, hspace=0.0, wspace=3.5)
    spectrogram = fig.add_subplot(grid[0:3, 0:4])
    rp_plot = fig.add_subplot(grid[0:3, 4:])
    fft_vector = fig.add_subplot(grid[4:, :])

    max_array = np.max(stft, axis=1)
    max_value_stft = np.max(max_array, axis=0)
    max_index =  list(max_array).index(max_value_stft)

    min_array = np.min(stft, axis=1)
    min_value_stft = np.min(min_array, axis=0)
    min_index = list(min_array).index(min_value_stft)

    # top = np.triu(matrix)
    # bottom = np.tril(matrix_binary)

    # np.linspace(0, stft.shape[1], stft.shape[1]),        np.linspace(0, stft.shape[0], cut_freq),
    rp_plot.imshow(matrix_binary, cmap='cividis', origin='lower')  # interpolation='none'
    # axs[0].imshow(bottom, cmap='jet', origin='lower') #interpolation='none'
    rp_plot.plot(max_index, max_index, 'orange', marker="o", markersize=7)
    rp_plot.plot(min_index, min_index, 'red', marker="o", markersize=7)
    # axs[0].set_yticks(axs[0].get_yticks()[1:len(axs[0].get_yticks())-1])
    # axs[0].set_xticks(axs[0].get_xticks()[1:len(axs[0].get_xticks())-1])
    rp_plot.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
    rp_plot.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
    rp_plot.set_xticklabels(
        [str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, rp_plot.get_xticks().shape[0])])
    rp_plot.set_yticklabels(
        [str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, rp_plot.get_yticks().shape[0])])
    rp_plot.set_xlabel('Time (s)')
    rp_plot.set_ylabel('Time (s)')
    rp_plot.set_title('Recurrence Plot', fontsize=10)

    spectrogram.pcolormesh(stft.T, shading='gouraud') #,vmax=max_value_stft
    spectrogram.plot(max_index,2,'orange', marker="|", markersize=30)
    spectrogram.plot(min_index,2,'red', marker="|", markersize=30)
    spectrogram.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft.shape[0], 6)))
    spectrogram.set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, stft.shape[0] / s_rate, spectrogram.get_xticks().shape[0])])
    spectrogram.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft.shape[1], 5)))
    spectrogram.set_yticklabels([str(np.round(x, 1)) for x in np.linspace(0, cut_freq, 5)])
    spectrogram.set_ylabel('Freq (Hz)')
    spectrogram.set_xlabel('Time (s)')
    spectrogram.set_title('Spectrogram', fontsize=10)


    max_index_ = stft[max_index]/stft.shape[1]
    min_index_ = stft[min_index]/stft.shape[1]
    fft_vector.plot(max_index_**2,'orange',label='$fft_{t_{1.0}}$')#,marker="o", markersize=2
    fft_vector.plot(min_index_**2,'red',label='$fft_{t_{2.3}}}$')#,marker="o", markersize=2
    fft_vector.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft.shape[1], 9)))
    fft_vector.set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, cut_freq, 9)])
    fft_vector.set_xlim([0,100])
    fft_vector.set_ylabel('Power ($\mu V^{2}$)')
    fft_vector.set_xlabel('Freq (Hz)')
    fft_vector.set_title('Frequency Domain', size=10)
    fft_vector.legend()

    plt.suptitle( 'Condition: '+ task  + '\n' + 'epsilon {},  FFT window size {} '.format(
                 str(info_args['eps']), str(info_args['win_len'])) + '\n'
                 + 'Subject {}, electrode {}, n_fft {}'.format(str(info_args['selected_subject']),str(info_args['electrode_name']),str(info_args['n_fft'])),
                 fontsize=8,ha='left',va='top')
    plt.tight_layout()

    # axs[0].imshow(matrix_binary, cmap='cividis', origin='lower') #interpolation='none'
    # # axs[0].imshow(bottom, cmap='jet', origin='lower') #interpolation='none'
    # axs[0].plot(max_index,max_index,'orange',marker="o", markersize=7)
    # axs[0].plot(min_index,min_index,'red',marker="o", markersize=7)
    # # axs[0].set_yticks(axs[0].get_yticks()[1:len(axs[0].get_yticks())-1])
    # # axs[0].set_xticks(axs[0].get_xticks()[1:len(axs[0].get_xticks())-1])
    # axs[0].xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
    # axs[0].yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
    # axs[0].set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, axs[0].get_xticks().shape[0])])
    # axs[0].set_yticklabels([str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, axs[0].get_yticks().shape[0])])
    # axs[0].set_xlabel('Time (s)')
    # axs[0].set_ylabel('Time (s)')
    # axs[0].set_title('Recurrence Plot', fontsize=12)


def diagnostic(matrix, matrix_binary, s_rate, stft, cut_freq, task, info_args):

    fig, axs = plt.subplots(3,1, figsize=(4,8), gridspec_kw={'height_ratios':[6,2,1]},dpi=120)

    # Set up the axes with gridspec


    max_array = np.max(stft, axis=1)
    max_value_stft = np.max(max_array, axis=0)
    max_index =  list(max_array).index(max_value_stft)

    min_array = np.min(stft, axis=1)
    min_value_stft = np.min(min_array, axis=0)
    min_index = list(min_array).index(min_value_stft)

    # top = np.triu(matrix)
    # bottom = np.tril(matrix_binary)

    axs[0].imshow(matrix_binary, cmap='cividis', origin='lower') #interpolation='none'
    # axs[0].imshow(bottom, cmap='jet', origin='lower') #interpolation='none'

    # axs[0].plot(max_index,max_index,'orange',marker="o", markersize=7)
    # axs[0].plot(min_index,min_index,'red',marker="o", markersize=7)

    # axs[0].set_yticks(axs[0].get_yticks()[1:len(axs[0].get_yticks())-1])
    # axs[0].set_xticks(axs[0].get_xticks()[1:len(axs[0].get_xticks())-1])
    axs[0].xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
    axs[0].yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
    axs[0].set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, axs[0].get_xticks().shape[0])])
    axs[0].set_yticklabels([str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, axs[0].get_yticks().shape[0])])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Time (s)')
    axs[0].set_title('Recurrence Plot', fontsize=10)


    # np.linspace(0, stft.shape[1], stft.shape[1]),        np.linspace(0, stft.shape[0], cut_freq),


    axs[1].pcolormesh(stft.T, shading='gouraud') #,vmax=max_value_stft
    axs[1].plot(max_index,0,'orange', marker="o", markersize=7)
    axs[1].plot(min_index,0,'red', marker="o", markersize=7)
    axs[1].xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
    axs[1].set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, axs[1].get_xticks().shape[0])])
    axs[1].yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft.shape[1], 5)))
    axs[1].set_yticklabels([str(np.round(x, 1)) for x in np.linspace(0, cut_freq, 5)])
    axs[1].set_ylabel('Freq (Hz)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_title('Spectrogram', fontsize=10)

    max_index_ = stft[max_index]/stft.shape[1]
    min_index_ = stft[min_index]/stft.shape[1]
    axs[2].plot(max_index_**2,'orange')#,marker="o", markersize=2
    axs[2].plot(min_index_**2,'red')#,marker="o", markersize=2
    axs[2].xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft.shape[1], 9)))
    axs[2].set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, cut_freq, 9)])
    axs[2].set_xlim([0,100])
    axs[2].set_ylabel('Power (µV^2)')
    axs[2].set_xlabel('Freq (Hz)')
    axs[2].set_title('Frequency Domain', size=10)

    plt.suptitle( 'Condition: '+ task  + '\n' + 'epsilon {},  FFT window size {} '.format(
                 str(info_args['eps']), str(info_args['win_len'])) + '\n'
                 + 'Subject {}, electrode {}, n_fft {}'.format(str(info_args['selected_subject']),str(info_args['electrode_name']),str(info_args['n_fft'])),
                 fontsize=8,ha='left',va='top')
    plt.tight_layout()
