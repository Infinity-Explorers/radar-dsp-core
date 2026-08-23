import numpy as np

def remove_static_clutter(range_matrix):
    """
    Remove stationary reflections by subtracting
    the mean across the chirp axis.

    Input:
        range_matrix shape:
        (range_bins, chirps, receivers, transmitters)
    """

    mean_clutter = np.mean(
        range_matrix,
        axis=1,
        keepdims=True
    )

    clean_matrix = range_matrix - mean_clutter

    return clean_matrix