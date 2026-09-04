import numpy as np
from scipy.ndimage import uniform_filter, rank_filter

#===================
#2D CFAR Detectors
#===================

def ca_cfar_2d(power_map, num_train_r, num_train_c, num_guard_r, num_guard_c, pfa=1e-4):
    """
    2D Cell-Averaging CFAR (CA-CFAR) Detector using vectorized uniform moving average.
    """
    # Calculate full sliding window dimensions (Training + Guard + CUT)
    kernel_r_full = 2 * (num_train_r + num_guard_r) + 1
    kernel_c_full = 2 * (num_train_c + num_guard_c) + 1

    # Calculate inner guard window dimensions (Guard + CUT)
    kernel_r_guard = 2 * num_guard_r + 1
    kernel_c_guard = 2 * num_guard_c + 1

    n_full = kernel_r_full * kernel_c_full
    n_guard = kernel_r_guard * kernel_c_guard
    n_train = n_full - n_guard

    # Compute total integrated power using uniform moving average filters
    sum_full = uniform_filter(power_map, size=(kernel_r_full, kernel_c_full), mode='constant') * n_full
    sum_guard = uniform_filter(power_map, size=(kernel_r_guard, kernel_c_guard), mode='constant') * n_guard

    noise_floor = (sum_full - sum_guard) / n_train

    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)

    threshold = noise_floor * alpha
    detection_mask = power_map > threshold

    return detection_mask, noise_floor

def os_cfar_2d(power_map, num_train_r, num_train_c, num_guard_r, num_guard_c, k_rank=None, pfa=1e-4):
    """
    2D Ordered-Statistic CFAR (OS-CFAR) Detector using rank filtering.
    """
    kernel_r_full = 2 * (num_train_r + num_guard_r) + 1
    kernel_c_full = 2 * (num_train_c + num_guard_c) + 1

    n_full = kernel_r_full * kernel_c_full
    n_guard = (2 * num_guard_r + 1) * (2 * num_guard_c + 1)
    n_train = n_full - n_guard

    # Default rank index k to 75th percentile of training cells if not provided
    if k_rank is None:
        k_rank = int(0.75 * n_train)

    # Estimate noise floor using k-th ordered statistic
    noise_floor = rank_filter(power_map, rank=k_rank, size=(kernel_r_full, kernel_c_full), mode='constant')

    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)

    threshold = noise_floor * alpha
    detection_mask = power_map > threshold

    return detection_mask, noise_floor