"""
Core 3D CFAR detection pipeline and peak extraction engine.
Handles dual-plane projections (RD, RA), 2D thresholding, range-gate association,
and 3D ghost target suppression.
"""

from typing import Any, Dict, List
import numpy as np
from cfar_detectors import ca_cfar_2d, os_cfar_2d


def detect_3d_peaks(
    cube: np.ndarray,
    res_params: Dict[str, Any],
    cfar_params: Dict[str, Any],
    algorithm: str = "CA"
) -> List[Dict[str, Any]]:
    """
    Executes Stage 1 peak detection across dual projections and extracts physical coordinates.
    """
    if np.iscomplexobj(cube):
        power_cube = np.abs(cube) ** 2
    else:
        power_cube = cube

    nr, nd, na = power_cube.shape

    # Dual-plane max-power projections
    power_rd = np.max(power_cube, axis=2)  # Range-Doppler: (Nr, Nd)
    power_ra = np.max(power_cube, axis=1)  # Range-Azimuth: (Nr, Na)

    pfa = cfar_params.get("pfa", 1e-4)
    k_rank = cfar_params.get("k_rank", 0.75)
    algo = algorithm.upper()

    if algo == "CA":
        mask_rd, _, thresh_rd = ca_cfar_2d(
            power_rd,
            cfar_params["num_train_r"], cfar_params["num_train_d"],
            cfar_params["num_guard_r"], cfar_params["num_guard_d"],
            pfa=pfa
        )
        mask_ra, _, _ = ca_cfar_2d(
            power_ra,
            cfar_params["num_train_r"], cfar_params["num_train_a"],
            cfar_params["num_guard_r"], cfar_params["num_guard_a"],
            pfa=pfa
        )
    elif algo == "OS":
        mask_rd, _, thresh_rd = os_cfar_2d(
            power_rd,
            cfar_params["num_train_r"], cfar_params["num_train_d"],
            cfar_params["num_guard_r"], cfar_params["num_guard_d"],
            k_rank=k_rank, pfa=pfa

        )
        mask_ra, _, _ = os_cfar_2d(
            power_ra,
            cfar_params["num_train_r"], cfar_params["num_train_a"],
            cfar_params["num_guard_r"], cfar_params["num_guard_a"],
            k_rank=k_rank, pfa=pfa

        )
    else:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose 'CA' or 'OS'.")

    # Range-gate association
    rd_r_indices, rd_d_indices = np.where(mask_rd)
    ra_r_indices, ra_az_indices = np.where(mask_ra)

    azimuth_axis = res_params.get("azimuth_axis", None)
    range_axis = res_params.get("range_axis", None)
    range_res = res_params.get("range_res", 0.2238)
    vel_res = res_params["vel_res"]
    az_res = res_params.get("az_res", 1.2)

    candidate_peaks = []
    visited_bins = set()

    for r_rd, d_idx in zip(rd_r_indices, rd_d_indices):
        matched_az_indices = ra_az_indices[np.abs(ra_r_indices - r_rd) <= 1]

        for az_idx in matched_az_indices:
            bin_tuple = (int(r_rd), int(d_idx), int(az_idx))
            if bin_tuple in visited_bins:
                continue

            candidate_power = power_cube[r_rd, d_idx, az_idx]
            threshold_cutoff = thresh_rd[r_rd, d_idx]

            if candidate_power <= threshold_cutoff:
                continue

            # 3D local maximum filter for ghost/sidelobe rejection
            r_slice = slice(max(0, r_rd - 1), min(nr, r_rd + 2))
            d_slice = slice(max(0, d_idx - 1), min(nd, d_idx + 2))
            a_slice = slice(max(0, az_idx - 1), min(na, az_idx + 2))
            
            if candidate_power < np.max(power_cube[r_slice, d_slice, a_slice]):
                continue

            visited_bins.add(bin_tuple)

            # Physical coordinates
            if range_axis is not None:
                range_m = float(range_axis[r_rd])
            else:
                range_m = float(r_rd * range_res)

            velocity_m_s = float((d_idx - nd // 2) * vel_res)
            
            if azimuth_axis is not None:
                azimuth_deg = float(azimuth_axis[az_idx])
            else:
                azimuth_deg = float((az_idx - na // 2) * az_res)

            candidate_peaks.append({
                "r_bin": int(r_rd),
                "v_bin": int(d_idx),
                "az_bin": int(az_idx),
                "range_m": range_m,
                "velocity_m_s": velocity_m_s,
                "azimuth_deg": azimuth_deg,
                "power": float(candidate_power),
                "noise_floor": float(threshold_cutoff)
            })

    return candidate_peaks