import matplotlib.pyplot as plt
# %matplotlib inline
# %load_ext autotime
plt.style.use('classic')
import numpy as np



def normalize(tSignal):
    # copy the data if needed, omit and rename function argument if desired
    signal = np.copy(tSignal) # signal is in range [a;b]
    signal -= np.min(signal) # signal is in range to [0;b-a]
    signal /= np.max(signal) # signal is normalized to [0;1]
    signal -= 0.5 # signal is in range [-0.5;0.5]
    signal *=2 # signal is in range [-1;1]
    return signal




def get_max_freqs(stft, cut_freq, crop=False, norm=True):
    if crop==False:
        crop=len(stft)
    id_max_stft = stft[:crop].argmax(axis=1)*(cut_freq/stft.shape[1])
    if norm:
        id_max_norm=id_max_stft*(1/cut_freq)
        return id_max_norm
    else: 
        return id_max_stft


def symmetrize(a):
    return a + np.transpose(a, (1, 0, 2)) - np.diag(a.diagonal())

def symmetrized(m):
    
    import numpy as np
    
    i_lower = np.tril_indices(m.shape[0], -1)
    m[:,:,0][i_lower] = m[:,:,0].T[i_lower] 
    m[:,:,1][i_lower] = m[:,:,1].T[i_lower]
    m[:,:,2][i_lower] = m[:,:,2].T[i_lower] 

    return m

def calc_color(id_max_norm, unique_values,colors,crop):
    if crop==False:
        crop=len(id_max_norm)
    tmp = np.zeros((id_max_norm.shape[0],id_max_norm.shape[0],3), dtype = 'float64')

    for i, v1 in enumerate(id_max_norm):
        for j, v2 in enumerate(id_max_norm):
            tmp[i,j] = v1,v2,0

    tmp_color = np.zeros((id_max_norm.shape[0],id_max_norm.shape[0],3), dtype = 'float64')
    for i in range(crop):
        for j in range(crop):
            x = tmp[i,j][0]
            y = tmp[i,j][1]

            for k in unique_values:

                if x == k:
                    id = (unique_values).index(x)
                    
                    tmp_color[i,j][0] = colors[id][0]
                if y == k:
                    id = (unique_values).index(y)
                    tmp_color[i,j][1] = colors[id][1]
                    
     
    return symmetrized(tmp_color),symmetrized(tmp)

def calc_color_raw(id_max_stft):

    tmp = np.zeros((id_max_stft.shape[0],id_max_stft.shape[0],3), dtype = 'float64')

    for i, v1 in enumerate(id_max_stft):
        for j, v2 in enumerate(id_max_stft):
            tmp[i,j] = v1,v2,0
    return symmetrized(tmp)

def get_unique_colors(cut_freq,unique_values):
    from matplotlib.colors import LinearSegmentedColormap
    
    vmax=cut_freq
    cmap = LinearSegmentedColormap.from_list('mycmap1', [(0 / vmax, "violet"),
                                                        (4 / vmax, 'blue'),
                                                        (8 / vmax, 'green'),
                                                        (15 / vmax, 'yellow'),
                                                        (30 / vmax, 'red'),
                                                        (60 / vmax, 'black')
                                                        ])

    colors=[cmap(each) for each in unique_values]
    
    return cmap, colors


def freqRP(stft, cut_freq, crop=False, norm=True):

    id_max_stft=get_max_freqs(stft,cut_freq,crop,norm)

    unique_values = np.unique(id_max_stft).tolist()

    cmap,colors=get_unique_colors(cut_freq, unique_values)

    color_matrix, raw_c_matrix = calc_color(id_max_stft, unique_values,colors,crop)

    return color_matrix, raw_c_matrix


def plot_freqRP_interactive(color_matrix, raw_c_matrix, filename='', save=False):
    import plotly.express as px
    import numpy as np
    fig=px.imshow(color_matrix, origin='lower')
    fig.update_traces(customdata=np.round((raw_c_matrix*60),2),
        hovertemplate="First frequency: %{customdata[0]}<br>Second frequency: %{customdata[1]}<extra></extra>"
    )
    if save:
        fig.write_html(filename)
    else:
        fig.show()

