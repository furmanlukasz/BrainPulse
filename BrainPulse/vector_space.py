import time
import torch

def compute_stft(epoch_, n_fft, win_len, s_rate, cut_freq):
    # stft_time = time.time()

    signal_tensor = torch.tensor(epoch_, dtype=torch.float)
    stft_tensor = torch.stft(signal_tensor,n_fft=n_fft, win_length=win_len, hop_length=1,return_complex=True,window=torch.hann_window(win_len))

    sft = torch.abs(stft_tensor).numpy()

    freq_to_take = (((n_fft/2)+1)*cut_freq) / ((s_rate/2)+1)

    sft = sft[:int(freq_to_take),::]

    return sft.T