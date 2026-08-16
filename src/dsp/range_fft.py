import numpy as np
from windowing import apply_window

def process_range_fft(raw_cube,TC=1e-4, B=150e6, window_type='hann'):
    window_cube = apply_window(raw_cube, window_type=window_type, axis=2)
    fft_out = np.fft.fft(window_cube, axis=2)
    c=3e8
    N = raw_cube.shape[2]
    fs = N / TC
    freq_axis = np.fft.fftfreq(N, d=1/fs)
    half_N = N // 2
    range_axis = (freq_axis[:half_N] * c *TC)/ (2 * B)
    return fft_out[:, :, :half_N], range_axis
