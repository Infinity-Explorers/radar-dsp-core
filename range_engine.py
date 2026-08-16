import numpy as np

def process_range_fft (raw_cube , Tc=1e-4 , B=150e6 , window_type = 'hann'):
    K, M, N = raw_cube.shape
    c=3e8
    window = np.hanning(N) if window_type == 'hann' else np.blackman(N)
    window = window.reshape(1, 1, N)
    windowed_cube = raw_cube * window
    range_fft=np.fft.fft(windowed_cube, axis=2)
    fs = N / Tc
    freq_bins = np.fft.fftfreq(N, d=1/fs)
    pos_mask = freq_bins >= 0
    range_fft = range_fft[:, :, pos_mask]
    range_axis = (c * freq_bins[pos_mask] * Tc) / (2 * B)
    return range_fft, range_axis
