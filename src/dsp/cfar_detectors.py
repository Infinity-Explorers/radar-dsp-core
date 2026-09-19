import numpy as np
from scipy.ndimage import uniform_filter, rank_filter
from scipy.optimize import brentq
#===================
#2D CFAR Detectors
#===================
def get_os_alpha(pfa, n_train, k_rank):
    # Product formula: prod{i=0}^{k-1} (N - i) / (N - i + alpha) = Pfa
    i_vals = np.arange(k_rank)

    def objective(alpha):
        log_terms = np.log(n_train - i_vals) - np.log(n_train - i_vals + alpha)
        return np.exp(np.sum(log_terms)) - pfa

    # Find root alpha where objective == 0
    return brentq(objective, 0.01, 1e4)
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
    sum_full = uniform_filter(power_map, size=(kernel_r_full, kernel_c_full), mode='nearest') * n_full
    sum_guard = uniform_filter(power_map, size=(kernel_r_full, kernel_c_full), mode='nearest') * n_guard

    noise_floor = (sum_full - sum_guard) / n_train
    noise_floor = np.maximum(noise_floor, 1e-9)

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

    kernel_r_guard = 2 * num_guard_r + 1
    kernel_c_guard = 2 * num_guard_c + 1

    n_full = kernel_r_full * kernel_c_full
    n_guard = (2 * num_guard_r + 1) * (2 * num_guard_c + 1)
    n_train = n_full - n_guard

    # Default rank index k to 75th percentile of training cells if not provided
    if k_rank is None or isinstance(k_rank, float):
        ratio = k_rank if isinstance(k_rank, float) else 0.75
        k_rank_idx = int(ratio * (n_train - 1))
    else:
        k_rank_idx = int(k_rank)

    # ضمان إن الـ rank جوه حدود الـ training cells
    k_rank_idx = max(0, min(k_rank_idx, n_train - 1))

    footprint = np.ones((kernel_r_full, kernel_c_full), dtype=bool)
    r_start = num_train_r
    r_end = r_start + kernel_r_guard
    c_start = num_train_c
    c_end = c_start + kernel_c_guard
    footprint[r_start:r_end, c_start:c_end] = False

    # Estimate noise floor using k-th ordered statistic
    noise_floor = rank_filter(power_map, rank=k_rank_idx, footprint=footprint, mode='nearest')
    noise_floor = np.maximum(noise_floor, 1e-9)
    
    alpha = get_os_alpha(pfa, n_train, k_rank_idx)

    threshold = noise_floor * alpha
    detection_mask = power_map > threshold

    margin_r = num_train_r + num_guard_r
    margin_c = num_train_c + num_guard_c
    detection_mask[:margin_r, :] = False
    detection_mask[-margin_r:, :] = False
    detection_mask[:, :margin_c] = False
    detection_mask[:, -margin_c:] = False

    return detection_mask, noise_floor