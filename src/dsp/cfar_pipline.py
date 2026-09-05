import numpy as np
from cfar_detectors import ca_cfar_2d, os_cfar_2d

def detect_3d_peaks(cube, res_params, cfar_params, algorithm='CA'):
    """
    Stage 1 Deliverable: Core CFAR Algorithms & Peak Extraction Engine
    
    Inputs:
        - cube: 3D complex radar cube tensor of shape (64, 255, 64)
        - res_params: dict containing 'range_res', 'vel_res', 'az_res'
        - cfar_params: dict containing training/guard window bounds and pfa
        - algorithm: 'CA' for Cell-Averaging or 'OS' for Ordered-Statistic
        
    Returns:
        - candidate_peaks: List of dictionaries containing candidate detection attributes
    """
    if np.iscomplexobj(cube):
        power_cube = np.abs(cube) ** 2
    else:
        power_cube = cube

    Nr, Nd, Na = power_cube.shape
    power_rd = np.max(power_cube, axis=2)  # Max over azimuth dimension
    power_ra = np.max(power_cube, axis=1)  # Max over Doppler dimension

    if algorithm.upper() == 'CA':
        mask_rd, noise_rd = ca_cfar_2d(
            power_rd,
            cfar_params['num_train_r'], cfar_params['num_train_d'],
            cfar_params['num_guard_r'], cfar_params['num_guard_d'],
            pfa=cfar_params.get('pfa', 1e-4)
        )
        mask_ra, noise_ra = ca_cfar_2d(
            power_ra,
            cfar_params['num_train_r'], cfar_params['num_train_a'],
            cfar_params['num_guard_r'], cfar_params['num_guard_a'],
            pfa=cfar_params.get('pfa', 1e-4)
        )
    elif algorithm.upper() == 'OS':
        mask_rd, noise_rd = os_cfar_2d(
            power_rd,
            cfar_params['num_train_r'], cfar_params['num_train_d'],
            cfar_params['num_guard_r'], cfar_params['num_guard_d'],
            pfa=cfar_params.get('pfa', 1e-4)
        )
        mask_ra, noise_ra = os_cfar_2d(
            power_ra,
            cfar_params['num_train_r'], cfar_params['num_train_a'],
            cfar_params['num_guard_r'], cfar_params['num_guard_a'],
            pfa=cfar_params.get('pfa', 1e-4)
        )
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Supported options: 'CA', 'OS'.")

    rd_r_indices, rd_d_indices = np.where(mask_rd)
    ra_r_indices, ra_az_indices = np.where(mask_ra)

    for r_rd, d_idx in zip(rd_r_indices, rd_d_indices):
     
     # Match RD and RA detections with range gate tolerance <= 1
     matching_mask = np.abs(ra_r_indices - r_rd) <= 1
     matched_az_indices = ra_az_indices[matching_mask]

    for az_idx in matched_az_indices:
        candidate_peaks = []
        candidate_power = power_cube[r_rd, d_idx, az_idx]
        local_noise_floor = noise_rd[r_rd, d_idx]
        if candidate_power > local_noise_floor:
            if candidate_power > local_noise_floor:
                # -------------------------------------------------------------
                # 5. Coordinate Transformation to Physical Units
                # -------------------------------------------------------------
                range_m = r_rd * res_params['range_res']
                velocity_m_s = (d_idx - Nd // 2) * res_params['vel_res']
                azimuth_deg = (az_idx - Na // 2) * res_params['az_res']

                candidate_peaks.append({
                    "r_bin": int(r_rd),
                    "v_bin": int(d_idx),
                    "az_bin": int(az_idx),
                    "range_m": float(range_m),
                    "velocity_m_s": float(velocity_m_s),
                    "azimuth_deg": float(azimuth_deg),
                    "power": float(candidate_power),
                    "noise_floor": float(local_noise_floor)
                })

    return candidate_peaks
if __name__ == "__main__":

    # 1. Setup Resolution and CFAR Parameters

    res_params = {
            'range_res': 0.2238,   # Range resolution in meters
            'vel_res': 0.15,       # Velocity resolution in m/s
            'az_res': 1.2          # Azimuth resolution in degrees
        }
    
    cfar_params = {
            'num_train_r': 4,
            'num_guard_r': 2,
            'num_train_d': 4,
            'num_guard_d': 2,
            'num_train_a': 2,
            'num_guard_a': 1,
            'pfa': 1e-4
        }

    # 2. Generate Dummy 3D Radar Cube Tensor of Shape (64, 255, 64)

    np.random.seed(42)  # For reproducible results
    noise_real = np.random.normal(0, 1, size=(64, 255, 64))
    noise_imag = np.random.normal(0, 1, size=(64, 255, 64))
    dummy_cube = noise_real + 1j * noise_imag


    # 3. Run Pipeline Execution

    print("Executing Stage 1 CFAR Pipeline on Dummy 3D Radar Cube...")
    detected_targets = detect_3d_peaks(dummy_cube, res_params, cfar_params, algorithm='CA')

    # 4. Display Extracted Candidates

    print(f"\n[+] Total Candidate Targets Extracted: {len(detected_targets)}")
    for idx, target in enumerate(detected_targets, start=1):
        print(f"\nTarget {idx}:")
        print(f"  Bins (r, v, az)      : ({target['r_bin']}, {target['v_bin']}, {target['az_bin']})")
        print(f"  Range (m)            : {target['range_m']:.3f} m")
        print(f"  Velocity (m/s)       : {target['velocity_m_s']:.3f} m/s")
        print(f"  Azimuth (deg)        : {target['azimuth_deg']:.3f}°")
        print(f"  Power / Noise Floor  : {target['power']:.2f} / {target['noise_floor']:.2f}")
    