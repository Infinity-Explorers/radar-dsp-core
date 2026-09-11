import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

CUBE_SHAPE = (64, 255, 64)
PATCH_SIZE = 9
HALF_PATCH = PATCH_SIZE // 2
MATCH_DISTANCE_THRESHOLD = 1.5
MAX_BACKGROUND_RATIO = 3
EPSILON = 1e-6

CLASS_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}


def validate_radar_cube(radar_cube):
    radar_cube = np.asarray(radar_cube)
    if radar_cube.shape != CUBE_SHAPE:
        raise ValueError(
            f"Expected radar cube shape {CUBE_SHAPE}, "
            f"but got {radar_cube.shape}"
        )
    return radar_cube


def extract_patch(radar_cube, peak):
    radar_cube = validate_radar_cube(radar_cube)
    r_bin, v_bin, az_bin = peak

    padded_cube = np.pad(
        radar_cube,
        pad_width=HALF_PATCH,
        mode="constant",
        constant_values=0
    )

    r = r_bin + HALF_PATCH
    v = v_bin + HALF_PATCH
    az = az_bin + HALF_PATCH

    patch = padded_cube[
        r - HALF_PATCH : r + HALF_PATCH + 1,
        v - HALF_PATCH : v + HALF_PATCH + 1,
        az - HALF_PATCH : az + HALF_PATCH + 1
    ]

    return patch


def log_power_scaling(patch):
    power = np.abs(patch) ** 2
    log_power = np.log10(power + EPSILON)
    return log_power


def calculate_distance(candidate_range, candidate_azimuth, gt_range, gt_azimuth):
    distance = np.sqrt(
        (candidate_range - gt_range) ** 2 +
        (candidate_azimuth - gt_azimuth) ** 2
    )
    return distance


def match_ground_truth(
    candidate_range,
    candidate_azimuth,
    metadata,
    frame_id=None,
    sequence=None
):
    gt = metadata.copy()

    if sequence is not None and "sequence" in gt.columns:
        gt = gt[gt["sequence"] == sequence]

    if frame_id is not None and "frame_id" in gt.columns:
        gt = gt[gt["frame_id"].astype(str) == str(frame_id)]

    if len(gt) == 0:
        return 0

    distances = calculate_distance(
        candidate_range,
        candidate_azimuth,
        gt["range_m"],
        gt["azimuth_deg"]
    )

    nearest_index = distances.idxmin()
    nearest_distance = distances.loc[nearest_index]

    if nearest_distance < MATCH_DISTANCE_THRESHOLD:
        original_class = int(gt.loc[nearest_index, "class"])
        return CLASS_MAP.get(original_class, 0)
    
    return 0


def subsample_clutter(
    patches,
    labels,
    max_background_ratio=MAX_BACKGROUND_RATIO,
    random_seed=42
):
    rng = np.random.default_rng(random_seed)
    patches = np.asarray(patches)
    labels = np.asarray(labels)

    if len(patches) == 0:
        return patches, labels

    foreground_indices = np.where(labels != 0)[0]
    background_indices = np.where(labels == 0)[0]

    # If testing with dummy data where no foreground targets match
    if len(foreground_indices) == 0:
        selected_indices = background_indices
    else:
        max_background = len(foreground_indices) * max_background_ratio

        if len(background_indices) > max_background:
            background_indices = rng.choice(
                background_indices,
                size=max_background,
                replace=False
            )

        selected_indices = np.concatenate([foreground_indices, background_indices])

    rng.shuffle(selected_indices)

    return patches[selected_indices], labels[selected_indices]


def generate_dummy_peaks(num_peaks=20, seed=42):
    rng = np.random.default_rng(seed)
    r_bins = rng.integers(0, CUBE_SHAPE[0], size=num_peaks)
    v_bins = rng.integers(0, CUBE_SHAPE[1], size=num_peaks)
    az_bins = rng.integers(0, CUBE_SHAPE[2], size=num_peaks)
    return list(zip(r_bins, v_bins, az_bins))


def build_roi_dataset(
    radar_cube,
    peaks,
    metadata,
    range_axis,
    azimuth_axis,
    frame_id=None,
    sequence=None
):
    patches = []
    labels = []

    for peak in peaks:
        r_bin, v_bin, az_bin = peak

        patch = extract_patch(radar_cube, peak)
        patch = log_power_scaling(patch)
        candidate_range = range_axis[r_bin]
        candidate_azimuth = azimuth_axis[az_bin]

        label = match_ground_truth(
            candidate_range,
            candidate_azimuth,
            metadata,
            frame_id=frame_id,
            sequence=sequence
        )

        patches.append(patch)
        labels.append(label)

    patches = np.asarray(patches)
    labels = np.asarray(labels, dtype=np.int64)

    return patches, labels


def process_sequence(
    sequence_cubes,
    sequence_peaks,
    metadata,
    range_axis,
    azimuth_axis,
    sequence_name,
    output_dir="data/roi_patches"
):
    all_patches = []
    all_labels = []

    for frame_id, radar_cube in enumerate(sequence_cubes):
        peaks = sequence_peaks.get(frame_id, generate_dummy_peaks(num_peaks=20))

        patches, labels = build_roi_dataset(
            radar_cube=radar_cube,
            peaks=peaks,
            metadata=metadata,
            range_axis=range_axis,
            azimuth_axis=azimuth_axis,
            frame_id=frame_id,
            sequence=sequence_name
        )

        if len(patches) > 0:
            all_patches.append(patches)
            all_labels.append(labels)

    if len(all_patches) > 0:
        all_patches = np.concatenate(all_patches, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

    # Subsample clutter once across the full sequence
    all_patches, all_labels = subsample_clutter(all_patches, all_labels)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_path = out_dir / f"{sequence_name}.npz"

    np.savez_compressed(
        export_path,
        patches=all_patches,
        labels=all_labels
    )
    print(f"Saved: {export_path} (Patches: {all_patches.shape}, Labels: {all_labels.shape})")


if __name__ == "__main__":
    range_axis = np.linspace(0, 100, CUBE_SHAPE[0])
    azimuth_axis = np.linspace(-60, 60, CUBE_SHAPE[2])
    
    dummy_cube = np.random.randn(*CUBE_SHAPE) + 1j * np.random.randn(*CUBE_SHAPE)
    dummy_metadata = pd.DataFrame({
        "sequence": ["seq_01"],
        "frame_id": [0],
        "range_m": [25.0],
        "azimuth_deg": [10.0],
        "class": [1]
    })

    dummy_peaks_dict = {0: generate_dummy_peaks(num_peaks=20)}

    process_sequence(
        sequence_cubes=[dummy_cube],
        sequence_peaks=dummy_peaks_dict,
        metadata=dummy_metadata,
        range_axis=range_axis,
        azimuth_axis=azimuth_axis,
        sequence_name="seq_01"
    )   