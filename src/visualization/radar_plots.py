import numpy as np
import matplotlib.pyplot as plt

def magnitude_to_db(data):
    """
    Convert magnitude to decibels.
    """

    return 20 * np.log10(
        data + 1e-12
    )

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
