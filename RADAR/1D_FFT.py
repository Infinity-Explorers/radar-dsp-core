import matplotlib.pyplot as plt
import numpy as np
fs = 1000 
t = np.linspace(0,1,fs,endpoint = False) 
f_target = 50
signal = np.sin(2* np.pi*f_target*t)
fft_result = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(signal),d=1/fs)
magnitude = np.abs(fft_result)
pos_mask = freqs >= 0
plt.plot(freqs[pos_mask], magnitude[pos_mask])
plt.xlabel(" frequency (HZ)")
plt.ylabel("magnitude")
plt.title("1D-FFT Spectrum")
plt.grid()
plt.show()

