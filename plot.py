import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
import matplotlib
import seaborn as sns
import seaborn as sns
import umap
import umap.plot
import pandas as pd
from .event import EventSegment
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from IPython.display import HTML, Audio, Video, Javascript
sns.set_style("whitegrid")

# plt.rcParams["font.family"] = "cursive"
# plt.rcParams.update({'font.sans-serif':'Times'})
# plt.rcParams.update({'font.family':'sans-serif'})
# plt.rcParams['font.size'] = 14
import matplotlib.font_manager as font_manager
font = font_manager.FontProperties(family='Times')

def explainer_(chan, stft, cut_freq, s_rate):

    fig, axs = plt.subplots(4, figsize=(10, 14), dpi=150)  # figsize=(12, 12),
    time_crop = np.linspace(0, int(chan[:400].shape[0]), chan[:400].shape[0])

    axs[0].plot(chan[:400],'k')  # np.linspace(0, int(chan[:400].shape[0]/s_rate), chan[:400].shape[0]),
    axs[0].fill_betweenx(y=[-210, 125], x1=time_crop[0],
                        x2=time_crop[240], color='white', alpha=0.9, edgecolor='red' )

    axs[0].fill_betweenx(y=[-210, 130], x1=time_crop[2]+20,
                         x2=time_crop[260], color='white', alpha=0.9, edgecolor='green')

    axs[0].fill_betweenx(y=[-210, 135], x1=time_crop[2]+40,
                         x2=time_crop[280], color='white', alpha=0.9, edgecolor='blue')

    axs[0].annotate('$fft_{1}$', xy=(.25, 72), xycoords='data',
                xytext=(0.05, 1.45), textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->",facecolor='black',color='black'),
                horizontalalignment='right', verticalalignment='top',
                )

    axs[0].annotate('$fft_{2}$', xy=(23.35, 85), xycoords='data',
                    xytext=(0.15, 1.45), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="->",facecolor='black',color='black'),
                    horizontalalignment='right', verticalalignment='top',
                    )

    axs[0].annotate('$fft_{3}$', xy=(43.45, 95), xycoords='data',
                    xytext=(0.25, 1.45), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="->",facecolor='black ',color='black'),
                    horizontalalignment='right', verticalalignment='top',
                    )
    axs[0].xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, chan[:400].shape[0], 5)))
    axs[0].set_xticklabels(
        [str(np.round(x, 1)) for x in np.linspace(0, int(chan[:400].shape[0] / s_rate), axs[0].get_xticks().shape[0])])
    axs[0].set_ylabel('Amplitude (µV)', )
    axs[0].set_xlabel('Time (s)', )
    axs[0].set_title('(a)', )
    axs[0].xaxis.grid()
    axs[0].yaxis.grid()


    axs[1].plot((stft[100]/stft.shape[1])**2, 'red',label='$fft_{1}$',marker="o",markersize=3)
    axs[1].legend(prop=font)
    axs[1].xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft.shape[1], 9)))
    axs[1].set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, cut_freq, 9)])
    axs[1].set_xlim([0, 100])
    # axs[1].set_ylim([0, 250])
    axs[1].set_ylabel('Power ($\mu V^{2}$)', )
    axs[1].set_xlabel('Freq (Hz)', )
    # axs[1].set_title('Frequency Domain ($fft_{1}$, $fft_{2}$, $fft_{3}$)', fontsize=10)
    axs[1].set_title('(b)', )
    axs[1].xaxis.grid()
    axs[1].yaxis.grid()


    axs[2].plot((stft[115]/stft.shape[1])**2, 'green', label='$fft_{2}$', marker="o", markersize=3)
    axs[2].legend(prop=font)
    axs[2].xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft.shape[1], 9)))
    axs[2].set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, cut_freq, 9)])
    axs[2].set_xlim([0, 100])
    # axs[2].set_ylim([0, 250])
    axs[2].set_ylabel('Power ($\mu V^{2}$)', )
    axs[2].set_xlabel('Freq (Hz)', )
    axs[2].set_title('(c)', )
    axs[2].xaxis.grid()
    axs[2].yaxis.grid()


    axs[3].plot((stft[140]/stft.shape[1])**2, 'blue', label='$fft_{3}$', marker="o", markersize=3)
    axs[3].legend(prop=font)
    axs[3].xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft.shape[1], 9)))
    axs[3].set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, cut_freq, 9)])
    axs[3].set_xlim([0, 100])
    axs[3].set_ylabel('Power ($\mu V^{2}$)', )
    axs[3].set_xlabel('Freq (Hz)', )
    axs[3].set_title('(d)', )
    axs[3].xaxis.grid()
    axs[3].yaxis.grid()

    # plt.title('Frequency Domain ($fft_{1}$, $fft_{2}$, $fft_{3}$)', fontsize=10)
    plt.tight_layout()

    plt.savefig('fig_4.png')


def stft_collections(matrix, matrix_binary, s_rate, stft, cut_freq, task, info_args, max_indx = None, min_indx = None):
    fig = plt.figure(figsize=(14, 12), dpi=150)
    grid = plt.GridSpec(6, 8, hspace=0.0, wspace=3.5)
    spectrogram = fig.add_subplot(grid[0:3, 0:4])
    rp_plot = fig.add_subplot(grid[0:3, 4:])
    fft_vector = fig.add_subplot(grid[4:, :])

    if max_indx != None and min_indx != None:
        max_index = max_indx
        min_index = min_indx
    else:
        max_array = np.max(stft, axis=1)
        max_value_stft = np.max(max_array, axis=0)
        max_index =  list(max_array).index(max_value_stft)

        min_array = np.min(stft, axis=1)
        min_value_stft = np.min(min_array, axis=0)
        min_index = list(min_array).index(min_value_stft)



    # ręczne ustawienie wskaźników
    # max_index = int(1.52*s_rate)
    # min_index = int(2.4*s_rate)


    # top = np.triu(matrix)
    # bottom = np.tril(matrix_binary)

    # np.linspace(0, stft.shape[1], stft.shape[1]),        np.linspace(0, stft.shape[0], cut_freq),
    rp_plot.imshow(matrix_binary, cmap='Greys', origin='lower')  # interpolation='none'
    # axs[0].imshow(bottom, cmap='jet', origin='lower') #interpolation='none'
    rp_plot.plot(max_index, max_index, 'orange', marker="o", markersize=9)
    rp_plot.plot(min_index, min_index, 'red', marker="o", markersize=9)
    # axs[0].set_yticks(axs[0].get_yticks()[1:len(axs[0].get_yticks())-1])
    # axs[0].set_xticks(axs[0].get_xticks()[1:len(axs[0].get_xticks())-1])
    rp_plot.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
    rp_plot.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
    rp_plot.set_xticklabels(
        [str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, rp_plot.get_xticks().shape[0])])
    rp_plot.set_yticklabels(
        [str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, rp_plot.get_yticks().shape[0])])
    rp_plot.set_xlabel('Time (s)', )
    rp_plot.set_ylabel('Time (s)', )
    rp_plot.set_title('(b) Recurrence Plot', )
    rp_plot.xaxis.grid()
    rp_plot.yaxis.grid()

    spectrogram.pcolormesh(stft.T,cmap='viridis') #,vmax=max_value_stft
    spectrogram.plot(max_index,2,'orange', marker="|", markersize=40)
    spectrogram.plot(min_index,2,'red', marker="|", markersize=40)

    spectrogram.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft.shape[0], 5)))
    spectrogram.set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, stft.shape[0] / s_rate, spectrogram.get_xticks().shape[0])])
    spectrogram.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft.shape[1], 5)))
    spectrogram.set_yticklabels([str(np.round(x, 1)) for x in np.linspace(0, cut_freq, 5)])
    spectrogram.set_ylabel('Freq (Hz)', )
    spectrogram.set_xlabel('Time (s)', )
    spectrogram.set_title('(a) Spectrogram', )
    # spectrogram.xaxis.grid()
    # spectrogram.yaxis.grid()
    # fig.colorbar(im1, cax=spectrogram, orientation='vertical')


    max_index_ = stft[max_index]/stft.shape[1]
    min_index_ = stft[min_index]/stft.shape[1]
    fft_vector.plot(max_index_**2,'orange',label='$fft_{t_{1}}$')#,marker="o", markersize=2
    fft_vector.plot(min_index_**2,'red',label='$fft_{t_{2}}}$')#,marker="o", markersize=2
    fft_vector.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft.shape[1], 9)))
    fft_vector.set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, cut_freq, 9)])
    fft_vector.set_xlim([0,100])
    fft_vector.set_ylabel('Power ($\mu V^{2}$)', )
    fft_vector.set_xlabel('Freq (Hz)', )
    fft_vector.set_title('(c) Frequency Domain', )
    fft_vector.legend(prop=font)
    fft_vector.xaxis.grid()
    fft_vector.yaxis.grid()

    # plt.suptitle( 'Condition: '+ task  + '\n' + 'epsilon {},  FFT window size {} '.format(
    #              str(info_args['eps']), str(info_args['win_len'])) + '\n'
    #              + 'Subject {}, electrode {}, n_fft {}'.format(str(info_args['selected_subject']),str(info_args['electrode_name']),str(info_args['n_fft'])), fontsize=12 ,ha='left',va='top')
    plt.tight_layout()
    plt.savefig('fig_5.png')
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

    fig, axs = plt.subplots(3,1, figsize=(7,12), gridspec_kw={'height_ratios':[6,2,1]},dpi=150)

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

    axs[0].plot(max_index,max_index,'orange',marker="o", markersize=7)
    axs[0].plot(min_index,min_index,'red',marker="o", markersize=7)

    axs[0].set_yticks(axs[0].get_yticks()[1:len(axs[0].get_yticks())-1])
    axs[0].set_xticks(axs[0].get_xticks()[1:len(axs[0].get_xticks())-1])
    axs[0].xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
    axs[0].yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
    axs[0].set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, axs[0].get_xticks().shape[0])])
    axs[0].set_yticklabels([str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, axs[0].get_yticks().shape[0])])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Time (s)')
    axs[0].set_title('Recurrence Plot', )


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
    axs[1].set_title('Spectrogram', )

    max_index_ = stft[max_index]/stft.shape[1]
    min_index_ = stft[min_index]/stft.shape[1]
    axs[2].plot(max_index_**2,'orange')#,marker="o", markersize=2
    axs[2].plot(min_index_**2,'red')#,marker="o", markersize=2
    axs[2].xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft.shape[1], 9)))
    axs[2].set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, cut_freq, 9)])
    axs[2].set_xlim([0,100])
    axs[2].set_ylabel('Power (µV^2)')
    axs[2].set_xlabel('Freq (Hz)')
    axs[2].set_title('Frequency Domain',)

    plt.suptitle( 'Condition: '+ task  + '\n' + 'epsilon {},  FFT window size {} '.format(
                 str(info_args['eps']), str(info_args['win_len'])) + '\n'
                 + 'Subject {}, electrode {}, n_fft {}'.format(str(info_args['selected_subject']),str(info_args['electrode_name']),str(info_args['n_fft'])),
                 ha='left',va='top')
    plt.tight_layout()

def RecurrencePlot(matrix, matrix_binary, s_rate, stft, cut_freq, task, info_args):

    fig, axs = plt.subplots( figsize=(12,12),dpi=200)

    # Set up the axes with gridspec


    max_array = np.max(stft, axis=1)
    max_value_stft = np.max(max_array, axis=0)
    max_index =  list(max_array).index(max_value_stft)

    min_array = np.min(stft, axis=1)
    min_value_stft = np.min(min_array, axis=0)
    min_index = list(min_array).index(min_value_stft)

    # top = np.triu(matrix)
    # bottom = np.tril(matrix_binary)

    axs.imshow(matrix_binary, cmap='cividis', origin='lower') #interpolation='none'
    # axs[0].imshow(bottom, cmap='jet', origin='lower') #interpolation='none'

    # axs[0].plot(max_index,max_index,'orange',marker="o", markersize=7)
    # axs[0].plot(min_index,min_index,'red',marker="o", markersize=7)

    # axs[0].set_yticks(axs[0].get_yticks()[1:len(axs[0].get_yticks())-1])
    # axs[0].set_xticks(axs[0].get_xticks()[1:len(axs[0].get_xticks())-1])
    axs.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
    axs.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
    axs.set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, axs.get_xticks().shape[0])])
    axs.set_yticklabels([str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, axs.get_yticks().shape[0])])
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Time (s)')
    axs.set_title('Recurrence Plot')


    # np.linspace(0, stft.shape[1], stft.shape[1]),        np.linspace(0, stft.shape[0], cut_freq),

def features_hists(df, features_list, condition, dpi = 200):
    fig, axs = plt.subplots(len(features_list),figsize=(6, len(features_list)*3), dpi=dpi)
    abc = ['(a)','(b)','(c)','(d)','(e)','(f)']

    for i,ax in enumerate(axs):
        sns.histplot(data=df, x=features_list[i], hue=condition, alpha=0.8, element="bars", fill=False, ax=ax, kde=True)
        ax.containers[1].remove()
        ax.containers[0].remove()
        ax.xaxis.grid()
        ax.yaxis.grid()
        ax.set_title(abc[i])
        # plt.grid(b=None)

    plt.autoscale(enable=True, axis='both', tight=None)
    fig.tight_layout()

def features_per_subjects_violin(df, features_list, condition, dpi = 200):
    fig, axs = plt.subplots(len(features_list),figsize=(14, len(features_list)*2), dpi=dpi,sharex='col')

    for i,ax in enumerate(axs):
        sns.violinplot(data=df, x=df.Subject, y=features_list[i], hue=condition, ax=ax, split=True,linewidth=0.2)
        ax.legend(loc='lower right')

    axs[len(features_list)-1].set_xticklabels(axs[len(features_list)-1].get_xticklabels(), rotation=90)
    # axs.set_ylim([0,1])


    plt.tick_params(axis='x', which='major', labelsize=16)
    fig.tight_layout()

def umap_on_condition(df,y, title,labels_name,features_list=['TT', 'RR', 'DET', 'LAM', 'L', 'Lentr'], random_state = 70, n_neighbors = 15, min_dist = 0.25, metric = "hamming", df_type=True):


    fig, ax1 = plt.subplots(figsize=(8, 8), dpi=150)

    if df_type:
        stats_data = df
    else:
        stats_data = df[features_list].values

    # Preprocess again
    pipe = make_pipeline(SimpleImputer(strategy="mean"), QuantileTransformer())
    X = pipe.fit_transform(stats_data.copy())

    # Fit UMAP to processed data
    manifold = umap.UMAP(random_state=random_state, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric).fit(X)
    # X_reduced_2 = manifold.transform(X)
    umap.plot.points(manifold, labels=labels_name, ax=ax1, color_key=np.array(
        [(0, 0.35, 0.73), (1, 0.83, 0)]))  # ,color_key=np.array([(1,0.83,0),(0,0.35,0.73)])
    ax1.set_title(title)

def umap_side_by_side_plot(df1, df2, features_list=['TT', 'RR', 'DET', 'LAM', 'L', 'Lentr'], random_state = 70, n_neighbors = 15, min_dist = 0.25, metric = "hamming"):

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(16,8),dpi=150)

    stats_data = df1[features_list].values
    y = df1.Task.values

    # Preprocess again
    pipe = make_pipeline(SimpleImputer(strategy="mean"), QuantileTransformer())
    X = pipe.fit_transform(stats_data.copy())

    # Fit UMAP to processed data
    manifold = umap.UMAP(random_state=random_state, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric).fit(X)
    # X_reduced_2 = manifold.transform(X)
    umap.plot.points(manifold, labels=y, ax=ax1, color_key=np.array(
        [(0, 0.35, 0.73), (1, 0.83, 0)]))  # ,color_key=np.array([(1,0.83,0),(0,0.35,0.73)])
    ax1.set_xlabel('(a) STFT Condition 0 - open eyes, 1 - closed eyes')

    stats_data = df2[features_list].values
    y = df2.Task.values

    # Preprocess again
    pipe = make_pipeline(SimpleImputer(strategy="mean"), QuantileTransformer())
    X = pipe.fit_transform(stats_data.copy())

    # Fit UMAP to processed data
    manifold = umap.UMAP(random_state=random_state, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric).fit(X)
    # X_reduced_2 = manifold.transform(X)
    umap.plot.points(manifold, labels=y, ax=ax2, color_key=np.array(
        [(0, 0.35, 0.73), (1, 0.83, 0)]))  # ,color_key=np.array([(1,0.83,0),(0,0.35,0.73)])
    ax2.set_xlabel('(b) TDEMB Condition 0 - open eyes, 1 - closed eyes')

    return

def SVM_histogram(df, lin, lin_pred,title):
    stats_data = df #[features_list].values
    plt.figure(dpi=150)
    all_cechy=np.dot(stats_data, lin.coef_.T)
    df_all=pd.DataFrame({'vectors':all_cechy.ravel(), 'Task':lin_pred})


    a = sns.histplot(data=df_all, x='vectors', hue='Task', alpha=0.8, element="bars", fill=False,kde=True, kde_kws={'bw_adjust':0.4},palette=np.array([(0.3,0.85,0),(0.8,0.0,0.44)]))
    a.containers[1].remove()
    a.containers[0].remove()
    # a = sns.kdeplot(data=df_all, x='vectors', hue='Task', alpha=0.8, bw_adjust=0.4,palette=np.array([(0.3,0.85,0),(0.8,0.0,0.44)]))
    plt.title(title)
    plt.xlabel('All')
    plt.grid(b=None)
    plt.show()

def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.figure()
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

def SVM_features_importance(lin):


    lebel_ll = np.array([['TT']*int(64)+ ['RR']*int(64)+
              ['DET']*int(64)+ ['LAM']*int(64)+
              ['L']*int(64)+ ['L_entr']*int(64)])

    e_long = "Af3	Af4	Af7	Af8	Afz	C1	C2	C3	C4	C5	C6	CZ	Cp1	Cp2	Cp3	Cp4	Cp5	Cp6	Cpz	F1	F2	F3	F4	F5	F6	F7	F8	Fc1	Fc2	Fc3	Fc4	Fc5	Fc6	Fcz	Fp1	Fp2	Fpz	Ft7	Ft8	Fz	Iz	O1	O2	OZ	P1	P2	P3	P4	P5	P6	P7	P8	Po3	Po4	Po7	Po8	Poz	Pz	T10	T7	T8	T9	Tp7	Tp8	Af3	Af4	Af7	Af8	Afz	C1	C2	C3	C4	C5	C6	CZ	Cp1	Cp2	Cp3	Cp4	Cp5	Cp6	Cpz	F1	F2	F3	F4	F5	F6	F7	F8	Fc1	Fc2	Fc3	Fc4	Fc5	Fc6	Fcz	Fp1	Fp2	Fpz	Ft7	Ft8	Fz	Iz	O1	O2	OZ	P1	P2	P3	P4	P5	P6	P7	P8	Po3	Po4	Po7	Po8	Poz	Pz	T10	T7	T8	T9	Tp7	Tp8	Af3	Af4	Af7	Af8	Afz	C1	C2	C3	C4	C5	C6	CZ	Cp1	Cp2	Cp3	Cp4	Cp5	Cp6	Cpz	F1	F2	F3	F4	F5	F6	F7	F8	Fc1	Fc2	Fc3	Fc4	Fc5	Fc6	Fcz	Fp1	Fp2	Fpz	Ft7	Ft8	Fz	Iz	O1	O2	OZ	P1	P2	P3	P4	P5	P6	P7	P8	Po3	Po4	Po7	Po8	Poz	Pz	T10	T7	T8	T9	Tp7	Tp8	Af3	Af4	Af7	Af8	Afz	C1	C2	C3	C4	C5	C6	CZ	Cp1	Cp2	Cp3	Cp4	Cp5	Cp6	Cpz	F1	F2	F3	F4	F5	F6	F7	F8	Fc1	Fc2	Fc3	Fc4	Fc5	Fc6	Fcz	Fp1	Fp2	Fpz	Ft7	Ft8	Fz	Iz	O1	O2	OZ	P1	P2	P3	P4	P5	P6	P7	P8	Po3	Po4	Po7	Po8	Poz	Pz	T10	T7	T8	T9	Tp7	Tp8	Af3	Af4	Af7	Af8	Afz	C1	C2	C3	C4	C5	C6	CZ	Cp1	Cp2	Cp3	Cp4	Cp5	Cp6	Cpz	F1	F2	F3	F4	F5	F6	F7	F8	Fc1	Fc2	Fc3	Fc4	Fc5	Fc6	Fcz	Fp1	Fp2	Fpz	Ft7	Ft8	Fz	Iz	O1	O2	OZ	P1	P2	P3	P4	P5	P6	P7	P8	Po3	Po4	Po7	Po8	Poz	Pz	T10	T7	T8	T9	Tp7	Tp8	Af3	Af4	Af7	Af8	Afz	C1	C2	C3	C4	C5	C6	CZ	Cp1	Cp2	Cp3	Cp4	Cp5	Cp6	Cpz	F1	F2	F3	F4	F5	F6	F7	F8	Fc1	Fc2	Fc3	Fc4	Fc5	Fc6	Fcz	Fp1	Fp2	Fpz	Ft7	Ft8	Fz	Iz	O1	O2	OZ	P1	P2	P3	P4	P5	P6	P7	P8	Po3	Po4	Po7	Po8	Poz	Pz	T10	T7	T8	T9	Tp7	Tp8".replace('\t',',').split(",")
    y_e_long = np.array(np.unique(e_long, return_inverse=True)[1].tolist())

    df = pd.DataFrame({'feature':lebel_ll[0],
                       'electrode':e_long,
                       'coef':lin.coef_[0],
                       })
    # df = df[(df.coef.values >= 0.15) | (df.coef.values <= -0.15)]

    f_importances(df.coef, df.feature)
    f_importances(df.coef, df.electrode)


    sns.set_theme(style='darkgrid', rc={'figure.dpi': 120},
                  font_scale=1.7)
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title('Weight of features by electrodes')
    sns.barplot(x='feature', y='coef', data=df, ax=ax,
                ci=None,
                hue='electrode')
    ax.legend(bbox_to_anchor=(1, 1), title='electrode',prop={'size': 7})

##### HIDDEN MARKOV MODEL

def soft_bounds(T,seg):

    # Identify soft boundaries at each step of fitting
    bounds_anim = []
    K = seg[0].shape[1]
    for it in range(1,len(seg)):
        sb = np.zeros((T,T))
        for k in range(K-1):
            p_change = np.diff(seg[it][:,(k+1):].sum(1))
            sb[1:,1:] += np.outer(p_change, seg[it][1:,k:(k+2)].sum(1))
        sb = np.maximum(sb,sb.T)
        sb = sb/np.max(sb)
        bounds_anim.append(sb)
    return bounds_anim


def fitting_animation(seg,matrix,s_rate,meta_tick,metastate_id, state_width,color_states_matrix):

    bounds_anim = soft_bounds(matrix.shape[0],seg)

    # Plot timepoint-timepoint correation matrix, with boundaries animated on top

    fig = plt.figure(figsize=(18, 12), dpi=300)
    grid = plt.GridSpec(4, 12, hspace=0.0, wspace=3.5)
    ax1 = fig.add_subplot(grid[:, 0:4])
    ax2 = fig.add_subplot(grid[:, 4:8])
    ax3 = fig.add_subplot(grid[:, 8:])


    # fig, axs = plt.subplots(2,figsize=(8,8), dpi=120)
    datamat = matrix # np.corrcoef(D)
    bk = cm.viridis((datamat-np.min(datamat))/(np.max(datamat)-np.min(datamat)))
    im = ax1.imshow(bk, interpolation='none',origin='lower')
    fg = cm.gray(1-(sum(bounds_anim)/len(bounds_anim)))
    # im.set_array(np.minimum(np.maximum(bk + fg, 0), 1))
    im.set_array(bk * fg)


    ax1.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
    ax1.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
    ax1.set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, ax1.get_xticks().shape[0])])
    ax1.set_yticklabels([str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, ax1.get_yticks().shape[0])])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Metastates plot over recurrence plot', fontsize=10)
    ax1.scatter(meta_tick,meta_tick,s=2)



    ax2.imshow(fg, interpolation='none',origin='lower')
    ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
    ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
    ax2.set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, ax2.get_xticks().shape[0])])
    ax2.set_yticklabels([str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, ax2.get_yticks().shape[0])])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Metastates plot', fontsize=10)
    ax2.scatter(meta_tick,meta_tick,s=2)

    text_kwargs = dict(ha='center', va='center', fontsize=4, color='0')
    for i,mstate in enumerate(metastate_id):
        # ax1.text(meta_tick[i]-35, meta_tick[i]+(state_width[i]/2)+45, 's'+str(mstate)+'| '+ str(int(((1/160)*state_width[i])*1000)) + 'ms', **text_kwargs)
        ax2.annotate('s '+str(mstate)+'| '+ str(int(((1/160)*state_width[i])*1000)) + 'ms', xy=(meta_tick[i], meta_tick[i]+(state_width[i]/2)),
                     xytext =(meta_tick[i], meta_tick[i]+(state_width[i]/2)+70),
                     xycoords='data',
                     textcoords='data',
                     arrowprops=dict(arrowstyle="->",facecolor='blue'),
                     horizontalalignment='right', verticalalignment='top', fontsize=5
                     )


    color_states = color_states_matrix
    ax3.imshow(fg[:,:,:3]*color_states, interpolation='none',origin='lower')
    ax3.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
    ax3.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, matrix.shape[0], 5)))
    ax3.set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, ax3.get_xticks().shape[0])])
    ax3.set_yticklabels([str(np.round(x, 1)) for x in np.linspace(0, matrix.shape[0] / s_rate, ax3.get_yticks().shape[0])])
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Time (s)')
    ax3.set_title('Metastates plot', fontsize=10)
    ax3.scatter(meta_tick,meta_tick,s=2)

    text_kwargs = dict(ha='center', va='center', fontsize=4, color='0')
    for i,mstate in enumerate(metastate_id):
        # ax1.text(meta_tick[i]-35, meta_tick[i]+(state_width[i]/2)+45, 's'+str(mstate)+'| '+ str(int(((1/160)*state_width[i])*1000)) + 'ms', **text_kwargs)
        ax3.annotate('s '+str(mstate)+'| '+ str(int(((1/160)*state_width[i])*1000)) + 'ms', xy=(meta_tick[i], meta_tick[i]+(state_width[i]/2)),
                     xytext =(meta_tick[i], meta_tick[i]+(state_width[i]/2)+70),
                     xycoords='data',
                     textcoords='data',
                     arrowprops=dict(arrowstyle="->",facecolor='blue'),
                     horizontalalignment='right', verticalalignment='top', fontsize=5
                     )


    # def animate_func(i):
    #     fg = cm.Greys(1-bounds_anim[i])
    #     im.set_array(np.minimum(np.maximum(bk + fg,0),1))
    #     return [im]
    #
    # anim = animation.FuncAnimation(fig, animate_func,
    #                                frames = len(bounds_anim), interval = 1)
    #
    #
    plt.savefig('Metastate.png')
    # plt.close("all")

    return fig

    # return HTML(anim.to_jshtml(default_mode='Once'))

def fit_HMM(matrix,n_events):
    return EventSegment(n_events=n_events).fit(matrix)

def metastates(seg,matrix,s_rate,meta_tick,metastate_id, state_width,color_states_matrix):
    fitting_animation(seg,matrix,s_rate,meta_tick,metastate_id, state_width,color_states_matrix)

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

