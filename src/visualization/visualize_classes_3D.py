import sys
from pathlib import Path

# Add repo root and src/dsp to sys.path
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent.parent.parent
DSP_DIR = ROOT_DIR / "src" / "dsp"

for p in [ROOT_DIR, DSP_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.io import loadmat
from tqdm import tqdm

# Project imports
from src.dsp.pipeline import run_dsp_pipeline
from src.dsp.cfar_pipline import detect_3d_peaks

# Try importing roi_extractor
try:
    from src.data_processing.roi_extractor import match_ground_truth, polar_to_cartesian
except ModuleNotFoundError:
    from src.dsp.roi_extractor_2 import match_ground_truth, polar_to_cartesian

# Local data paths
DATA_DIR = ROOT_DIR / "data"
AUTOMOTIVE_DIR = DATA_DIR / "Automotive"
LOCAL_METADATA_PATH = DATA_DIR / "metadata.parquet"
CACHE_FILE = DATA_DIR / "extracted_3d_detections.parquet"
HTML_EXPORT_PATH = DATA_DIR / "radar_detections_3d.html"

# CFAR Parameters
try:
    import src.configs as configs
    BASE_CFAR_PARAMS = getattr(configs, "cfar_params", {
        "num_train_r": 4, "num_guard_r": 2,
        "num_train_d": 4, "num_guard_d": 2,
        "num_train_a": 2, "num_guard_a": 1,
        "pfa": 1e-3, "k_rank": 0.75, "algorithm": "CA"
    })
except Exception:
    BASE_CFAR_PARAMS = {
        "num_train_r": 4, "num_guard_r": 2,
        "num_train_d": 4, "num_guard_d": 2,
        "num_train_a": 2, "num_guard_a": 1,
        "pfa": 1e-3, "k_rank": 0.75, "algorithm": "CA"
    }

INV_CLASS_MAP = {
    0: "Background / Clutter",
    1: "Person / Pedestrian",
    2: "Cyclist",
    3: "Car",
    4: "Motorbike",
    5: "Bus",
    6: "Truck"
}

CLASS_COLORS = {
    0: "rgba(160, 160, 160, 0.35)",  # Translucent clutter
    1: "#1f77b4",                    # Blue
    2: "#2ca02c",                    # Green
    3: "#d62728",                    # Red
    4: "#9467bd",                    # Purple
    5: "#8c564b",                    # Brown
    6: "#ff7f0e"                     # Orange
}


def scan_local_frames() -> list[Path]:
    """Finds all raw radar .mat frames in data/Automotive recursively."""
    if not AUTOMOTIVE_DIR.exists():
        raise FileNotFoundError(f"Local Automotive directory not found at: {AUTOMOTIVE_DIR}")
    
    print(f"Scanning local files in {AUTOMOTIVE_DIR}...")
    frame_files = sorted(AUTOMOTIVE_DIR.rglob("*.mat"))
    print(f"Found {len(frame_files)} local .mat frames.")
    return frame_files


def load_local_frame(file_path: Path) -> dict:
    """Reads a local .mat frame and parses sequence name and frame_id."""
    mat = loadmat(str(file_path))
    raw_frame = mat["adcData"]  # (samples, chirps, receivers, transmitters)

    # Stem format: '000003' or 'radar_raw_frame_000003'
    clean_stem = file_path.stem.replace("radar_raw_frame_", "").lstrip("0")
    frame_id = clean_stem.zfill(10) if clean_stem else "0000000000"

    # Identify sequence name from parent folders (e.g., '2019_04_09_bms1000')
    sequence = "unknown"
    for part in file_path.parts:
        if part.startswith("2019_"):
            sequence = part
            break

    return {
        "radar_raw_frame": raw_frame,
        "frame_id": frame_id,
        "sequence": sequence
    }


def extract_and_label_dataset(max_frames: int = None, force_recompute: bool = False) -> pd.DataFrame:
    """
    Runs DSP + CFAR locally over your disk frames with a live progress bar.
    Caches the results to data/extracted_3d_detections.parquet.
    """
    if CACHE_FILE.exists() and not force_recompute:
        print(f"Loading cached detections from: {CACHE_FILE}")
        return pd.read_parquet(CACHE_FILE)

    if not LOCAL_METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing {LOCAL_METADATA_PATH}. Place metadata.parquet in {DATA_DIR}")

    df_metadata = pd.read_parquet(LOCAL_METADATA_PATH)
    df_metadata["frame_id"] = df_metadata["frame_id"].astype(str).str.zfill(10)

    frame_files = scan_local_frames()
    if max_frames is not None:
        frame_files = frame_files[:max_frames]

    all_detections = []
    print(f"Processing {len(frame_files)} frames directly from disk...")

    with tqdm(total=len(frame_files), desc="DSP & CFAR Pipeline", unit="frame") as pbar:
        for file_path in frame_files:
            frame_data = load_local_frame(file_path)
            raw_adc = frame_data["radar_raw_frame"]
            seq_name = frame_data["sequence"]
            frame_id = frame_data["frame_id"]

            # 1. Range-Doppler-Azimuth DSP Engine
            r_axis, v_axis, rd_angle_cube, az_axis = run_dsp_pipeline(raw_adc)

            # Two-way radar propagation factor correction
            corrected_r_axis = r_axis / 2.0

            res_params = {
                "range_res": float(corrected_r_axis[1] - corrected_r_axis[0]),
                "range_axis": corrected_r_axis,
                "vel_res": float(v_axis[1] - v_axis[0]),
                "azimuth_axis": az_axis
            }

            # 2. 3D CFAR peak detection
            peaks = detect_3d_peaks(
                rd_angle_cube,
                res_params,
                BASE_CFAR_PARAMS,
                algorithm=BASE_CFAR_PARAMS.get("algorithm", "CA")
            )

            # 3. Ground truth target association
            for p in peaks:
                cand_range = p["range_m"]
                cand_azimuth = p["azimuth_deg"]
                cand_velocity = p["velocity_m_s"]

                class_id = match_ground_truth(
                    candidate_range=cand_range,
                    candidate_azimuth=cand_azimuth,
                    metadata=df_metadata,
                    frame_id=frame_id,
                    sequence=seq_name
                )

                cand_x, cand_y = polar_to_cartesian(cand_range, cand_azimuth)

                all_detections.append({
                    "x_m": cand_x,
                    "y_m": cand_y,
                    "velocity_mps": cand_velocity,
                    "range_m": cand_range,
                    "azimuth_deg": cand_azimuth,
                    "power_db": 10 * np.log10(p["power"] + 1e-6),
                    "class_id": class_id,
                    "class_name": INV_CLASS_MAP.get(class_id, "Unknown"),
                    "frame_id": frame_id,
                    "sequence": seq_name
                })

            pbar.update(1)

    df = pd.DataFrame(all_detections)

    if not df.empty:
        df.to_parquet(CACHE_FILE, index=False)
        print(f"\nSaved {len(df)} total detections to cache: {CACHE_FILE}")

    return df


def plot_classes_3d(df_detections: pd.DataFrame, max_background_display: int = 5000):
    """Plots the 3D space and exports a standalone HTML file."""
    if df_detections.empty:
        print("No detections found.")
        return

    print("Building 3D visualization...")
    fig = go.Figure()

    for class_id, class_name in INV_CLASS_MAP.items():
        subset = df_detections[df_detections["class_id"] == class_id]
        if subset.empty:
            continue

        is_clutter = (class_id == 0)

        # Cap background clutter visual count so WebGL stays responsive
        if is_clutter and len(subset) > max_background_display:
            subset = subset.sample(n=max_background_display, random_state=42)

        fig.add_trace(go.Scatter3d(
            x=subset["x_m"],
            y=subset["y_m"],
            z=subset["velocity_mps"],
            mode="markers",
            name=f"{class_name} (N={len(subset)})",
            marker=dict(
                size=2.5 if is_clutter else 6.0,
                color=CLASS_COLORS.get(class_id, "black"),
                opacity=0.25 if is_clutter else 0.85,
                symbol="circle" if not is_clutter else "x"
            ),
            hovertemplate=(
                f"<b>Class: {class_name}</b><br>"
                "X (Lateral): %{x:.2f} m<br>"
                "Y (Longitudinal): %{y:.2f} m<br>"
                "Velocity: %{z:.2f} m/s<br>"
                "Power: %{customdata[0]:.1f} dB<br>"
                "Seq: %{customdata[1]} | Frame: %{customdata[2]}<extra></extra>"
            ),
            customdata=subset[["power_db", "sequence", "frame_id"]].values
        ))

    fig.update_layout(
        title=dict(
            text="3D Automotive Radar Detections (Cartesian X-Y-V)",
            font=dict(size=18)
        ),
        scene=dict(
            xaxis=dict(title="Lateral Distance X (m)", zeroline=True, showgrid=True),
            yaxis=dict(title="Longitudinal Distance Y (m)", zeroline=True, showgrid=True),
            zaxis=dict(title="Doppler Velocity (m/s)", zeroline=True, showgrid=True),
            aspectmode="data"
        ),
        legend=dict(
            title=dict(text="Class Labels"),
            itemsizing="constant",
            yanchor="top",
            y=0.9,
            xanchor="left",
            x=1.02
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    HTML_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(HTML_EXPORT_PATH))
    print(f"Standalone HTML file saved: {HTML_EXPORT_PATH.resolve()}")
    fig.show()


if __name__ == "__main__":
    # max_frames=None processes all local files found across all sequences
    # Or set e.g. max_frames=200 for a test run
    df_results = extract_and_label_dataset(max_frames=None, force_recompute=False)
    plot_classes_3d(df_results)