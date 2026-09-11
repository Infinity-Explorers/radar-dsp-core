import numpy as np
from src.dsp.windowing import apply_window
from src.configs import startFreqConst_GHz, chirpDuration_usec, c

carrier_frequency_Hz = startFreqConst_GHz * 1e9
WAVELENGTH = c / carrier_frequency_Hz
chirp_repetition_time = chirpDuration_usec * 1e-6

def process_doppler_fft(clutter_removed_cube, Tc=chirp_repetition_time, window_type='hann', axis=1):
    """
        1. Perform Doppler FFT across the chirp axis.
            Axis 1 = slow-time / chirps

        2. Calculate target radial velocity from Doppler frequency.
            v = (lambda / 2) * f_d
        """
    windowed = apply_window(clutter_removed_cube, window_type=window_type, axis=axis)

    doppler_spectrum = np.fft.fft(
            windowed,
            axis=axis
        )
    
    doppler_spectrum = np.fft.fftshift(
        doppler_spectrum,
        axes=axis
        )

    wavelength = c / carrier_frequency_Hz
    n_chirps = clutter_removed_cube.shape[axis]
    
    doppler_frequency = np.fft.fftfreq(
        n_chirps,
        d=Tc
    )

    doppler_frequency = np.fft.fftshift(
        doppler_frequency
    )

    velocity = (
        wavelength / 2
    ) * doppler_frequency

    return doppler_spectrum, velocity