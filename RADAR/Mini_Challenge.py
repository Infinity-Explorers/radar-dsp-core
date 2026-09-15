import matplotlib.pyplot as plt
import numpy as np
Fs = 1000 #  1000 نقطة ف الثانية 
T = 1.0 # مدة الاشارة  واحد ثانية
N = int(Fs*T) # اجمالي عدد النقط 
t = np.linspace(0,T,N,endpoint=False) #  محور الزمن
target_freq = 50 # 50 (HZ)
clean_signal = np.sin(2*np.pi*target_freq*t)

# adding Gaussian Noise

np.random.normal(loc=0,scale=1,size=len(clean_signal))
noise = np.random.normal(loc=0,scale=1,size=len(clean_signal))
noisy_signal = clean_signal + noise

# appling 1D-FFT and extract freq

fft_result = np.fft.fft(noisy_signal )
freqs = np.fft.fftfreq(len(clean_signal),d=1/Fs)
magnitude = np.abs(fft_result)
pos_mask = freqs>=0
freqs_pos = freqs[pos_mask]
magnitude_pos = magnitude[pos_mask]
# plotting result using matplotlib

plt.figure(figsize=(10,4))

# plot 1 > signal at time domain (before and after noise)

plt.subplot(1,2,1)
plt.plot(t[:100],noisy_signal[:100],label='noisy signal',color='red',alpha=0.7)
plt.plot(t[:100],clean_signal[:100],label='clean signal',color='blue',linewidth=2)
plt.title('Time Domain')
plt.xlabel("Time")
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# plot 2 > signal at freq domain(after fft peak )

plt.subplot(1,2,2)
plt.plot(freqs_pos,magnitude_pos,color='green',linewidth=2)
plt.title("Frequency Domain (FTT Peak)")
plt.xlabel("Frequency (HZ)")
plt.ylabel("Magnitude")
plt.xlim(0,150)
plt.grid(True)

plt.tight_layout()
plt.show()
