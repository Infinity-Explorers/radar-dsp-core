import numpy as np
from src.dsp.windowing import apply_window
from src.configs import (
    freqSlopeConst_MHz_usec,
    bandwidth_GHz,
    digOutSampleRate,
    c
)

# Convert all parameters to standard SI units (Hz, s, Hz/s)
B_HZ = bandwidth_GHz * 1e9                         
SLOPE_HZ_PER_SEC = freqSlopeConst_MHz_usec * 1e12  
FS_HZ = digOutSampleRate * 1e3                    
CHIRP_ACTIVE_DURATION = B_HZ / SLOPE_HZ_PER_SEC   

def process_range_fft(raw_cube, Tc=CHIRP_ACTIVE_DURATION, B=B_HZ, Fs=FS_HZ, window_type='hann'):
    windowed_cube = apply_window(raw_cube, window_type=window_type, axis=0)
    
    fft_out = np.fft.fft(windowed_cube, axis=0)
    
    N = raw_cube.shape[0]
    half_N = N // 2
    
    freq_axis = np.fft.fftfreq(N, d=1.0 / Fs)[:half_N]
    
    range_axis = (freq_axis * c * Tc) / (2.0 * B)
    
    return fft_out[:half_N, ...], range_axis