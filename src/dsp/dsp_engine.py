import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Radar Configuration
# ============================================================

C = 3e8

CARRIER_FREQUENCY = 77e9       # 77 GHz
BANDWIDTH = 0.67e9             # 0.67 GHz = 670 MHz
CHIRP_TIME = 60e-6             # 60 us

NUM_SAMPLES = 128
NUM_CHIRPS = 255
NUM_RX = 4
NUM_TX = 2


# ============================================================
# 1. Static Clutter Removal
# ============================================================

def remove_static_clutter(range_matrix):
    """
    Remove stationary reflections by subtracting
    the mean across the chirp axis.

    Input:
        range_matrix shape:
        (receivers, chirps, range_bins)
    """

    mean_clutter = np.mean(
        range_matrix,
        axis=1,
        keepdims=True
    )

    clean_matrix = range_matrix - mean_clutter

    return clean_matrix


# ============================================================
# 2. Doppler FFT
# ============================================================

def doppler_fft(clean_matrix):
    """
    Perform Doppler FFT across the chirp axis.

    Axis 1 = slow-time / chirps
    """

    doppler_spectrum = np.fft.fft(
        clean_matrix,
        axis=1
    )

    doppler_spectrum = np.fft.fftshift(
        doppler_spectrum,
        axes=1
    )

    return doppler_spectrum


# ============================================================
# 3. Velocity Axis
# ============================================================

def calculate_velocity_axis(
    n_chirps,
    chirp_repetition_time=CHIRP_TIME,
    carrier_frequency=CARRIER_FREQUENCY
):
    """
    Calculate target radial velocity from Doppler frequency.

    v = (lambda / 2) * f_d
    """

    wavelength = C / carrier_frequency

    doppler_frequency = np.fft.fftfreq(
        n_chirps,
        d=chirp_repetition_time
    )

    doppler_frequency = np.fft.fftshift(
        doppler_frequency
    )

    velocity = (
        wavelength / 2
    ) * doppler_frequency

    return velocity


# ============================================================
# 4. Range-Doppler Map
# ============================================================

def create_range_doppler_map(doppler_spectrum):
    """
    Average the receiver dimension and create
    a Range-Doppler matrix.

    Input:
        (receivers, chirps, range_bins)

    Output:
        (range_bins, chirps)
    """

    magnitude = np.abs(doppler_spectrum)

    rdm = np.mean(
        magnitude,
        axis=0
    )

    rdm = rdm.T

    return rdm


# ============================================================
# 5. Magnitude to dB
# ============================================================

def magnitude_to_db(data):
    """
    Convert magnitude to decibels.
    """

    return 20 * np.log10(
        data + 1e-12
    )


# ============================================================
# 6. Range-Doppler Plot
# ============================================================

def plot_range_doppler_map(
    rdm_db,
    velocity,
    range_axis
):
    """
    Plot the Range-Doppler Map.

    X-axis -> Velocity
    Y-axis -> Range
    Color -> Signal magnitude in dB
    """

    plt.figure(figsize=(10, 6))

    plt.imshow(
        rdm_db,
        aspect="auto",
        extent=[
            velocity[0],
            velocity[-1],
            range_axis[-1],
            range_axis[0]
        ]
    )

    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Range (m)")
    plt.title("Range-Doppler Map")

    plt.colorbar(
        label="Magnitude (dB)"
    )

    plt.show()


# ============================================================
# 7. Spatial Angle FFT
# ============================================================

def angle_fft(doppler_spectrum):
    """
    Perform spatial FFT across the receiver antennas.

    Axis 0 = receiver antennas.
    """

    angle_spectrum = np.fft.fft(
        doppler_spectrum,
        axis=0
    )

    angle_spectrum = np.fft.fftshift(
        angle_spectrum,
        axes=0
    )

    return angle_spectrum


# ============================================================
# 8. Azimuth Axis
# ============================================================

def calculate_azimuth_axis(
    n_receivers=NUM_RX,
    carrier_frequency=CARRIER_FREQUENCY
):
    """
    Calculate azimuth angles.

    Receiver spacing:
        d = lambda / 2
    """

    wavelength = C / carrier_frequency

    antenna_spacing = wavelength / 2

    spatial_frequency = np.fft.fftfreq(
        n_receivers,
        d=1.0
    )

    spatial_frequency = np.fft.fftshift(
        spatial_frequency
    )

    sin_theta = (
        wavelength * spatial_frequency
        / antenna_spacing
    )

    valid = np.abs(sin_theta) <= 1

    azimuth = np.full(
        sin_theta.shape,
        np.nan,
        dtype=float
    )

    azimuth[valid] = np.degrees(
        np.arcsin(sin_theta[valid])
    )

    return azimuth


# ============================================================
# 9. Polar Scope Plot
# ============================================================

def plot_polar_scope(
    angle_spectrum,
    azimuth,
    range_axis
):
    """
    Create a polar radar-style plot.

    Radius -> Range
    Angle  -> Azimuth
    """

    magnitude = np.abs(angle_spectrum)

    # Average over Doppler dimension
    angle_range_map = np.mean(
        magnitude,
        axis=1
    )

    # Convert receiver-angle dimension to
    # range x angle representation
    angle_range_map = angle_range_map.T

    angle_rad = np.radians(
        azimuth
    )

    valid = ~np.isnan(angle_rad)

    theta = angle_rad[valid]
    data = angle_range_map[:, valid]

    theta_grid, range_grid = np.meshgrid(
        theta,
        range_axis
    )

    plt.figure(figsize=(8, 8))

    ax = plt.subplot(
        111,
        projection="polar"
    )

    ax.pcolormesh(
        theta_grid,
        range_grid,
        data,
        shading="auto"
    )

    ax.set_title(
        "Azimuth-Range Polar Scope"
    )

    plt.show()


# ============================================================
# 10. Complete DSP Pipeline
# ============================================================

def process_dsp_and_plot(
    range_matrix,
    range_axis,
    chirp_repetition_time=CHIRP_TIME,
    carrier_frequency=CARRIER_FREQUENCY
):
    """
    Complete DSP processing pipeline.

    Input:
        range_matrix:
            Range Spectrum from Member 2.

        Expected shape:
            (receivers, chirps, range_bins)

        range_axis:
            Range axis from Member 2.

    Returns:
        clean_matrix
        doppler_spectrum
        velocity
        rdm
        rdm_db
        angle_spectrum
        azimuth
    """

    # --------------------------------------------------------
    # Step 1: Static clutter removal
    # --------------------------------------------------------

    clean_matrix = remove_static_clutter(
        range_matrix
    )

    # --------------------------------------------------------
    # Step 2: Doppler FFT
    # --------------------------------------------------------

    doppler_spectrum = doppler_fft(
        clean_matrix
    )

    # --------------------------------------------------------
    # Step 3: Velocity axis
    # --------------------------------------------------------

    n_chirps = range_matrix.shape[1]

    velocity = calculate_velocity_axis(
        n_chirps,
        chirp_repetition_time,
        carrier_frequency
    )

    # --------------------------------------------------------
    # Step 4: Range-Doppler Map
    # --------------------------------------------------------

    rdm = create_range_doppler_map(
        doppler_spectrum
    )

    # --------------------------------------------------------
    # Step 5: Convert to dB
    # --------------------------------------------------------

    rdm_db = magnitude_to_db(
        rdm
    )

    # --------------------------------------------------------
    # Step 6: Plot RDM
    # --------------------------------------------------------

    plot_range_doppler_map(
        rdm_db,
        velocity,
        range_axis
    )

    # --------------------------------------------------------
    # Step 7: Spatial Angle FFT
    # --------------------------------------------------------

    angle_spectrum = angle_fft(
        doppler_spectrum
    )

    # --------------------------------------------------------
    # Step 8: Azimuth
    # --------------------------------------------------------

    n_receivers = range_matrix.shape[0]

    azimuth = calculate_azimuth_axis(
        n_receivers,
        carrier_frequency
    )

    return (
        clean_matrix,
        doppler_spectrum,
        velocity,
        rdm,
        rdm_db,
        angle_spectrum,
        azimuth
    )