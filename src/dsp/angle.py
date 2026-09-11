import numpy as np
from src.dsp.windowing import apply_window
from src.configs import startFreqConst_GHz, c

C = c
CARRIER_FREQUENCY = startFreqConst_GHz * 1e9

def angle_fft(doppler_spectrum, num_angle_bins=64, carrier_frequency=CARRIER_FREQUENCY):
    """
    Perform spatial FFT across the receiver antennas.

    Input shape: (range_bins, chirps, receivers, transmitters)
    Virtual antenna axis = axis 2 (merged receivers and transmitters).
    """
    # Merge 4 Rx and 2 Tx into 8 virtual antenna channels along Axis 2
    r_bins, n_chirps, n_rx, n_tx = doppler_spectrum.shape
    virtual_array = doppler_spectrum.reshape((r_bins, n_chirps, n_rx * n_tx))

    # Apply spatial windowing across virtual antenna elements
    windowed = apply_window(virtual_array, window_type='hann', axis=2)

    angle_spectrum = np.fft.fft(
        windowed,
        n=num_angle_bins,
        axis=2
    )

    angle_spectrum = np.fft.fftshift(
        angle_spectrum,
        axes=2
    )

    """
    Calculate azimuth angles.

    Receiver spacing:
        d = lambda / 2
    """

    wavelength = C / carrier_frequency

    antenna_spacing = wavelength / 2

    spatial_frequency = np.fft.fftfreq(
        num_angle_bins,
        d=antenna_spacing / wavelength
    )

    spatial_frequency = np.fft.fftshift(
        spatial_frequency
    )

    sin_theta = spatial_frequency

    valid = np.abs(sin_theta) <= 1

    azimuth = np.full(
        sin_theta.shape,
        np.nan,
        dtype=float
    )

    azimuth[valid] = np.degrees(
        np.arcsin(sin_theta[valid])
    )

    return angle_spectrum , azimuth