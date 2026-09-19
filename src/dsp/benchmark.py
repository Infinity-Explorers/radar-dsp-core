"""
Stage 2 Benchmark Suite: CFAR Sensitivity & Parameter Tuning.
Evaluates detection recall vs. false alarm rate (Pfa) on real radar sequence frames,
profiles execution latency, and exports tuned parameters to config/cfar_config.yaml.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import yaml

FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent.parent.parent
OUTPUT_CONFIG_PATH = ROOT_DIR / "config" / "cfar_config.yaml"
sys.path.extend([str(FILE_PATH.parent), str(FILE_PATH.parent.parent), str(ROOT_DIR)])

try:
    import configs
except ModuleNotFoundError:
    import src.configs as configs

from src.data_access.hf_client import frame_stream
from src.dsp.pipeline import run_dsp_pipeline
from src.dsp.cfar_pipline import detect_3d_peaks

from huggingface_hub import hf_hub_download

REPO_ID = "hany34/raw-adc-data-77ghz-mmwave-radar-automotive-object-detection"
LOCAL_METADATA_PATH = ROOT_DIR / "data" / "metadata.parquet"


def load_ground_truth() -> pd.DataFrame:
    """Loads ground truth from local disk if present; otherwise caches it via Hugging Face Hub."""
    print("Loading ground-truth metadata...")

    if LOCAL_METADATA_PATH.exists():
        parquet_path = LOCAL_METADATA_PATH
        print(f"Using local metadata cache: {parquet_path}")
    else:
        print("Downloading metadata.parquet once to persistent cache...")
        LOCAL_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        parquet_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="metadata.parquet",
            repo_type="dataset",
            local_dir=str(LOCAL_METADATA_PATH.parent),
        )

    df = pd.read_parquet(parquet_path)
    df["frame_id"] = df["frame_id"].astype(str).str.zfill(10)
    print(f"Metadata ready: {len(df)} total bounding records.")
    return df


def get_gt_for_frame(frame: Dict[str, Any], df_metadata: pd.DataFrame) -> List[Dict[str, float]]:
    if not isinstance(frame, dict):
        return []

    seq = frame.get("sequence")
    fid = str(frame.get("frame_id", "")).zfill(10)

    matched = df_metadata[(df_metadata["sequence"] == seq) & (df_metadata["frame_id"] == fid)]
    if matched.empty:
        return []

    return [
        {
            "range": float(row["range_m"]),
            "azimuth": float(row["azimuth_deg"]),
            "class": row["class_name"]
        }
        for _, row in matched.iterrows()
    ]


def evaluate_detections(
    detected_peaks: List[Dict[str, Any]],
    gt_targets: List[Dict[str, float]],
    range_tolerance: float = 1.5,
    azimuth_tolerance: float = 5.0
) -> Tuple[float, int]:
    if not gt_targets:
        return 0.0, 0

    matched_gt = 0
    for gt in gt_targets:
        for det in detected_peaks:
            r_diff = abs(det["range_m"] - gt["range"])
            az_diff = abs(det["azimuth_deg"] - gt["azimuth"])
            if r_diff <= range_tolerance and az_diff <= azimuth_tolerance:
                matched_gt += 1
                break

    recall = matched_gt / len(gt_targets)
    return recall, matched_gt


def main():
    print("=" * 70)
    print("AUTOMOTIVE RADAR DSP BENCHMARK SUITE")
    print("=" * 70)

    df_gt = load_ground_truth()

    num_frames_to_bench = 25
    print(f"\nStreaming first {num_frames_to_bench} sequence frames...")
    cached_stream = []
    for i, frame in enumerate(frame_stream(realtime=False)):
        cached_stream.append(frame)
        if i + 1 >= num_frames_to_bench:
            break
    print(f"Cached {len(cached_stream)} frames successfully.")

    print("Running DSP Range-Doppler-Azimuth engine across frames...")
    processed_cubes = []
    for frame in cached_stream:
        raw_adc = frame["radar_raw_frame"] if isinstance(frame, dict) else frame
        r_axis, v_axis, cube, az_axis = run_dsp_pipeline(raw_adc)
        
        gt_targets = get_gt_for_frame(frame, df_gt)
        
        # NOTE: Division by 2.0 corrects for missing two-way radar propagation factor in pipeline axis
        corrected_range_axis = r_axis / 2.0
        
        res_params = {
            "range_res": float(corrected_range_axis[1] - corrected_range_axis[0]),
            "range_axis": corrected_range_axis,
            "vel_res": float(v_axis[1] - v_axis[0]),
            "azimuth_axis": az_axis
        }
        processed_cubes.append((cube, res_params, gt_targets))

    annotated_frames_count = sum(1 for _, _, gt in processed_cubes if len(gt) > 0)
    print(f"Frames with ground-truth targets: {annotated_frames_count}/{len(processed_cubes)}")

    base_cfar_params = getattr(configs, "cfar_params", {
        "num_train_r": 4, "num_guard_r": 2,
        "num_train_d": 4, "num_guard_d": 2,
        "num_train_a": 2, "num_guard_a": 1,
        "k_rank": 0.75, "algorithm": "CA"
    })

    pfa_candidates = [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    best_pfa = None
    best_recall = -1.0
    best_stats = {}

    print("\nBenchmarking Sensitivity Across Pfa Candidates:")
    print(f"{'Pfa':<10} | {'Avg Recall':<12} | {'Proposals/Frame':<16} | {'CFAR Latency':<12}")
    print("-" * 58)

    for pfa in pfa_candidates:
        cfar_params = base_cfar_params.copy()
        cfar_params["pfa"] = pfa

        total_recall = 0.0
        total_proposals = 0
        total_cfar_ms = 0.0

        for cube, res_params, gt_targets in processed_cubes:
            t0 = time.perf_counter()
            peaks = detect_3d_peaks(cube, res_params, cfar_params, algorithm=cfar_params["algorithm"])
            t1 = time.perf_counter()

            total_cfar_ms += (t1 - t0) * 1000.0
            total_proposals += len(peaks)

            if gt_targets:
                recall, _ = evaluate_detections(peaks, gt_targets)
                total_recall += recall

        avg_recall = total_recall / max(1, annotated_frames_count)
        avg_proposals = total_proposals / len(processed_cubes)
        avg_latency = total_cfar_ms / len(processed_cubes)

        print(f"{pfa:<10.1e} | {avg_recall:<12.4f} | {avg_proposals:<16.1f} | {avg_latency:<9.2f} ms")

        if avg_recall > best_recall:
            best_recall = avg_recall
            best_pfa = pfa
            best_stats = {
                "recall": avg_recall,
                "proposals": avg_proposals,
                "latency_ms": avg_latency
            }

    print("-" * 58)
    if best_pfa is not None:
        print(f"\nOptimal Operating Point: Pfa = {best_pfa:.1e}")
        print(f"Target Recall: {best_stats['recall']:.2%}")
        print(f"Proposals/Frame: {best_stats['proposals']:.1f}")
        print(f"Average CFAR Latency: {best_stats['latency_ms']:.2f} ms")

        final_config = base_cfar_params.copy()
        final_config["pfa"] = float(best_pfa)

        OUTPUT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump({"cfar_params": final_config}, f, default_flow_style=False)
        print(f"Exported configuration to: {OUTPUT_CONFIG_PATH.resolve()}")
    else:
        print("Benchmark completed with no valid proposals extracted.")


if __name__ == "__main__":
    main()