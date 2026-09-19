"""2D Constant False Alarm Rate (CFAR) Detectors.

Provides vectorized Cell-Averaging (CA-CFAR) via uniform moving sums
and Ordered-Statistic (OS-CFAR) via rank filtering.
"""

from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import rank_filter, uniform_filter
from scipy.optimize import brentq


def get_os_alpha(pfa: float, n_train: int, k_rank: int) -> float:
    """Calculates the scaling factor alpha for OS-CFAR using Brent's root-finding method.

    Solves: prod_{i=0}^{k-1} (N - i) / (N - i + alpha) = Pfa
    """
    # Enforce 1-based order index (1 <= k <= N)
    k = max(1, min(int(k_rank), n_train))
    i_vals = np.arange(k)

    def objective(alpha: float) -> float:
        log_terms = np.log(n_train - i_vals) - np.log(n_train - i_vals + alpha)
        return np.exp(np.sum(log_terms)) - pfa

    try:
        return float(brentq(objective, 1e-4, 1e5))
    except ValueError:
        # Fallback to analytical CA-CFAR approximation if root search fails on extreme edges
        return float(n_train * (pfa ** (-1.0 / n_train) - 1.0))


def ca_cfar_2d(
        power_map: np.ndarray,
        num_train_r: int,
        num_train_c: int,
        num_guard_r: int,
        num_guard_c: int,
        pfa: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized 2D Cell-Averaging CFAR detector.

    Returns:
        detection_mask: Boolean array indicating detections.
        noise_floor: Estimated background noise power.
        threshold: Final cutoff threshold (noise_floor * alpha).
    """
    kernel_r_full = 2 * (num_train_r + num_guard_r) + 1
    kernel_c_full = 2 * (num_train_c + num_guard_c) + 1

    kernel_r_guard = 2 * num_guard_r + 1
    kernel_c_guard = 2 * num_guard_c + 1

    n_full = kernel_r_full * kernel_c_full
    n_guard = kernel_r_guard * kernel_c_guard
    n_train = n_full - n_guard

    # Vectorized moving averages
    sum_full = (
        uniform_filter(
            power_map, size=(kernel_r_full, kernel_c_full), mode="nearest"
        )
        * n_full
    )
    sum_guard = (
        uniform_filter(
            power_map, size=(kernel_r_guard, kernel_c_guard), mode="nearest"
        )
        * n_guard
    )


    noise_floor = (sum_full - sum_guard) / n_train
    noise_floor = np.maximum(noise_floor, 1e-9)

    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)
    threshold = noise_floor * alpha
    detection_mask = power_map > threshold

    # Zero out image borders where the sliding window extends outside the matrix
    margin_r = num_train_r + num_guard_r
    margin_c = num_train_c + num_guard_c
    detection_mask[:margin_r, :] = False
    detection_mask[-margin_r:, :] = False
    detection_mask[:, :margin_c] = False
    detection_mask[:, -margin_c:] = False

    return detection_mask, noise_floor, threshold


def os_cfar_2d(
    power_map: np.ndarray,
    num_train_r: int,
    num_train_c: int,
    num_guard_r: int,
    num_guard_c: int,
    k_rank: Optional[float] = None,
    pfa: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized 2D Ordered-Statistic CFAR detector using SciPy rank filtering.

    Returns:
        detection_mask: Boolean array indicating detections.
        noise_floor: Estimated background noise power (k-th rank).
        threshold: Final cutoff threshold (noise_floor * alpha).
    """
    kernel_r_full = 2 * (num_train_r + num_guard_r) + 1
    kernel_c_full = 2 * (num_train_c + num_guard_c) + 1

    kernel_r_guard = 2 * num_guard_r + 1
    kernel_c_guard = 2 * num_guard_c + 1

    n_full = kernel_r_full * kernel_c_full
    n_guard = kernel_r_guard * kernel_c_guard
    n_train = n_full - n_guard

    # Determine k rank index (0-indexed for SciPy rank_filter)
    if k_rank is None or isinstance(k_rank, float):
        ratio = k_rank if isinstance(k_rank, float) else 0.75
        k_rank_idx = int(round(ratio * (n_train - 1)))
    else:
        k_rank_idx = int(k_rank)

    k_rank_idx = max(0, min(k_rank_idx, n_train - 1))
    
    # Build hollow window footprint
    footprint = np.ones((kernel_r_full, kernel_c_full), dtype=bool)
    r_start = num_train_r
    r_end = r_start + kernel_r_guard
    c_start = num_train_c
    c_end = c_start + kernel_c_guard
    footprint[r_start:r_end, c_start:c_end] = False

    noise_floor = rank_filter(
        power_map, rank=k_rank_idx, footprint=footprint, mode="nearest"
    )

    noise_floor = np.maximum(noise_floor, 1e-9)

    # Pass 1-based order k to root-finder
    alpha = get_os_alpha(pfa, n_train, k_rank_idx + 1)
    threshold = noise_floor * alpha
    detection_mask = power_map > threshold

    # Zero out margins
    margin_r = num_train_r + num_guard_r
    margin_c = num_train_c + num_guard_c
    detection_mask[:margin_r, :] = False
    detection_mask[-margin_r:, :] = False
    detection_mask[:, :margin_c] = False
    detection_mask[:, -margin_c:] = False

    return detection_mask, noise_floor, threshold