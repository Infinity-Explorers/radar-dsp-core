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

# Name of the column in metadata.parquet that holds the ground-truth class.
# Adjust this if the actual parquet schema uses a different column name.
GT_CLASS_COLUMN = "class_id"


def validate_radar_cube(radar_cube):
    radar_cube = np.asarray(radar_cube)
    if radar_cube.shape != CUBE_SHAPE:
        raise ValueError(
            f"Expected radar cube shape {CUBE_SHAPE}, "
            f"but got {radar_cube.shape}"
        )
    return radar_cube


def extract_patch(radar_cube, peak):
    """Extract a 9x9x9 sub-volume centered on `peak`, with symmetric
    zero-padding applied whenever the window would exceed the cube
    boundaries."""
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
    """log10(|patch|^2 + eps) normalization."""
    power = np.abs(patch) ** 2
    log_power = np.log10(power + EPSILON)
    return log_power


def polar_to_cartesian(range_val, azimuth_deg):
    """Convert (range, azimuth) polar coordinates to Cartesian (x, y)
    so distances are computed in a single, consistent unit (meters)."""
    azimuth_rad = np.radians(azimuth_deg)
    x = range_val * np.sin(azimuth_rad)
    y = range_val * np.cos(azimuth_rad)
    return x, y


def calculate_distance(candidate_range, candidate_azimuth, gt_range, gt_azimuth):
    """Euclidean distance in meters between a candidate and a ground-truth
    center, after converting both from (range, azimuth) polar coordinates
    to Cartesian (x, y). Mixing range (meters) and azimuth (degrees)
    directly in a single Euclidean formula is physically meaningless,
    so both points are projected into the same Cartesian plane first.

    `gt_range` / `gt_azimuth` are typically pandas Series pulled from a
    filtered DataFrame, so their index may be non-contiguous or
    duplicated. To keep this a purely positional, vectorized computation
    (and avoid pandas trying to align on that index, which can silently
    introduce NaNs or mismatches), we strip both inputs down to raw
    NumPy arrays with `.to_numpy()` / `np.asarray()` before computing.
    The caller is responsible for mapping the resulting positional
    argmin back to the original DataFrame index.
    """
    gt_range = gt_range.to_numpy() if hasattr(gt_range, "to_numpy") else np.asarray(gt_range)
    gt_azimuth = gt_azimuth.to_numpy() if hasattr(gt_azimuth, "to_numpy") else np.asarray(gt_azimuth)

    cand_x, cand_y = polar_to_cartesian(candidate_range, candidate_azimuth)
    gt_x, gt_y = polar_to_cartesian(gt_range, gt_azimuth)

    distance = np.sqrt((cand_x - gt_x) ** 2 + (cand_y - gt_y) ** 2)
    return distance


def match_ground_truth(
    candidate_range,
    candidate_azimuth,
    metadata,
    frame_id=None,
    sequence=None
):
    """Match a CFAR candidate to the nearest ground-truth object.
    Returns class_id (1..6) if a match is found within
    MATCH_DISTANCE_THRESHOLD, otherwise returns 0 (Clutter/False Alarm)."""
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

    # distances is now a plain NumPy array (positional, 0-based), so we
    # find the nearest match positionally with argmin, then translate
    # that position back to the original (possibly non-contiguous)
    # DataFrame index via gt.index. This avoids relying on pandas index
    # alignment, which could otherwise produce NaNs or mismatches after
    # upstream filtering.
    nearest_pos = np.argmin(distances)
    nearest_distance = distances[nearest_pos]
    nearest_index = gt.index[nearest_pos]

    if nearest_distance < MATCH_DISTANCE_THRESHOLD:
        original_class = int(gt.loc[nearest_index, GT_CLASS_COLUMN])
        return CLASS_MAP.get(original_class, 0)

    return 0


def subsample_clutter(
    patches,
    labels,
    max_background_ratio=MAX_BACKGROUND_RATIO,
    random_seed=42
):
    """Cap the background(clutter):foreground ratio at max_background_ratio:1."""
    rng = np.random.default_rng(random_seed)
    patches = np.asarray(patches)
    labels = np.asarray(labels)

    if len(patches) == 0:
        return patches, labels

    foreground_indices = np.where(labels != 0)[0]
    background_indices = np.where(labels == 0)[0]

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


def build_roi_dataset(
    radar_cube,
    peaks,
    metadata,
    range_axis,
    azimuth_axis,
    frame_id=None,
    sequence=None
):
    """Build (patches, labels) arrays for a single frame."""
    patches = []
    labels = []

    for peak_info in peaks:
        if isinstance(peak_info, dict):
            peak = (peak_info["r_bin"], peak_info["v_bin"], peak_info["az_bin"])
        else:
            peak = peak_info

        patch = extract_patch(radar_cube, peak)
        patch = log_power_scaling(patch)
        patch = patch[np.newaxis, ...]  # Add channel dim -> shape becomes (1, 9, 9, 9)

        r_bin, _, az_bin = peak

        # Prefer Member 1's precomputed physical coordinates when available,
        # to avoid sign/offset mismatches with the ground-truth metadata.
        # Fall back to axis lookups only if peak_info doesn't carry them.
        if isinstance(peak_info, dict):
            candidate_range = peak_info.get("range_m", range_axis[r_bin])
            candidate_azimuth = peak_info.get("azimuth_deg", azimuth_axis[az_bin])
        else:
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

    if len(patches) == 0:
        # Preserve a well-defined shape even when a frame has no peaks,
        # so downstream concatenation / consumers never see shape (0,).
        patches = np.empty((0, 1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE))
        labels = np.empty((0,), dtype=np.int64)
    else:
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
    """Process a single sequence: extract patches/labels for every frame,
    subsample clutter, and export to data/roi_patches/<sequence_name>.npz."""
    all_patches = []
    all_labels = []

    for frame_id, radar_cube in enumerate(sequence_cubes):
        peaks = sequence_peaks.get(frame_id, [])

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
    else:
        # No peaks in the whole sequence: keep a well-defined empty shape
        # instead of collapsing to shape (0,).
        all_patches = np.empty((0, 1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE))
        all_labels = np.empty((0,), dtype=np.int64)

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

    return export_path


def process_all_sequences(
    sequences,
    metadata,
    range_axis,
    azimuth_axis,
    output_dir="data/roi_patches"
):
    """Batch-export entry point: iterate over every sequence and export
    each one to its own .npz file under `output_dir`.

    `sequences` is expected to be a dict mapping:
        sequence_name -> {
            "cubes": list/array of radar cubes for that sequence,
            "peaks": dict mapping frame_id -> list of peak (r_bin, v_bin, az_bin)
        }
    """
    exported_paths = []

    for sequence_name, sequence_data in sequences.items():
        sequence_cubes = sequence_data["cubes"]
        sequence_peaks = sequence_data["peaks"]

        export_path = process_sequence(
            sequence_cubes=sequence_cubes,
            sequence_peaks=sequence_peaks,
            metadata=metadata,
            range_axis=range_axis,
            azimuth_axis=azimuth_axis,
            sequence_name=sequence_name,
            output_dir=output_dir
        )
        exported_paths.append(export_path)

    return exported_paths
