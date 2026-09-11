# Range Engine & Windowing Documentation

## 1. Overview
The Range Engine processes the raw 3D FMCW radar ADC data cube $(K, M, N)$ to extract physical target distances via 1D Fast-Time Range FFT.

---

## 2. Spectral Leakage & Windowing Theory

### Spectral Leakage
When continuous radar signals are sampled over a finite time window ($T_c$), the truncation introduces artificial step discontinuities at the boundaries. In the frequency domain, these discontinuities cause energy from a single frequency component to "leak" into adjacent frequency bins, creating high sidelobes that can mask weaker nearby targets.

### Windowing Function
To eliminate boundary discontinuities, a continuous window function (e.g., Hann or Blackman) is applied across the Fast-Time samples (Axis 2) prior to the FFT. The window smoothly attenuates the signal amplitudes toward zero at both ends of the sample interval:

$$\text{Hann Window: } w[n] = 0.5 - 0.5 \cos\left(\frac{2\pi n}{N-1}\right), \quad 0 \le n < N$$

Applying $x_{\text{windowed}}[n] = x[n] \cdot w[n]$ significantly suppresses sidelobes, improving target detection dynamic range.

---

## 3. Range FFT & Distance Mapping Math

### Fast-Time Range FFT
The 1D FFT is computed along Axis 2 (Fast-Time samples, $N$) for each chirp and receiver channel to extract the beat frequencies ($f_b$).

### Distance Formula
The beat frequency $f_b$ obtained from FFT bin index $k$ is directly proportional to the target range $R$:

$$R = \frac{c \cdot f_b \cdot T_c}{2 \cdot B}$$

Where:
* $c$: Speed of light ($3 \times 10^8 \text{ m/s}$)
* $f_b$: Beat frequency ($\text{Hz}$)
* $T_c$: Chirp duration ($\text{seconds}$)
* $B$: Sweep bandwidth ($\text{Hz}$)

### Range Resolution & Max Range
* **Range Resolution ($\Delta R$):** $\Delta R = \frac{c}{2B}$
* **Maximum Unambiguous Range ($R_{\max}$):** $R_{\max} = \frac{c \cdot f_s \cdot T_c}{4B}$
