import sys
from pathlib import Path
file_path = Path(__file__).resolve()
sys.path.extend([
    str(file_path.parent),                      # src/dsp
    str(file_path.parent.parent),               # src
    str(file_path.parent.parent.parent)         # radar-dsp-core (Root)
])

try:
    import configs
except ModuleNotFoundError:
    import src.configs as configs

import time
import numpy as np
from src.data_access.hf_client import frame_stream
from src.dsp.pipeline import run_dsp_pipeline  
from src.dsp.cfar_pipline import detect_3d_peaks
import pandas as pd
df_gt = pd.read_parquet(
    "hf://datasets/hany34/raw-adc-data-77ghz-mmwave-radar-automotive-object-detection/metadata.parquet"
)
unique_frames = df_gt['frame_id'].unique()
def get_gt_for_frame(frame_idx, df_meatadata:pd.DataFrame) -> list[dict]:
    if frame_idx >= len(unique_frames):
        return []
    target_frame_id = unique_frames[frame_idx]
    frame_data = df_meatadata[df_meatadata['frame_id'] == target_frame_id]
    gt_targets = []
    for _, row in frame_data.iterrows():
        gt_targets.append({
            'range': row['range_m'],
            'azimuth': row['azimuth_deg'],
            'class': row['class_name']

        })
    return gt_targets
def evaluate_detections(detected_peaks, gt_targets, range_tolerance=1.0, azimuth_tolerance=5.0):
    if not gt_targets:
        return 0.0
    matched_gt = 0
    for gt in gt_targets:
        for det in detected_peaks:
            det_range = det.get('range_m', det.get('range', 0.0))
            det_azimuth = det.get('azimuth_deg', det.get('azimuth', 0.0))
            r_diff = abs(det_range - gt['range'])
            az_diff = abs(det_azimuth - gt['azimuth'])
            if r_diff <= range_tolerance and az_diff <= azimuth_tolerance:
                matched_gt += 1
                break
    return matched_gt / len(gt_targets)
if __name__ == "__main__":
    print("Starting DSP benchmark...")
    cached_frames = []
    for i, frame in enumerate(frame_stream(realtime=False)):
        cached_frames.append(frame)
        if i >= 100:  
            break
    base_cfar_params = getattr(configs, 'cfar_params', {
        'num_train_r': 4,'num_guard_r': 2,
        'num_train_d': 4,'num_guard_d': 2,
        'num_train_a': 2,'num_guard_a': 1,
        'k_rank': 0.75,'algorithm': 'CA'
    })
    pfa_candidates = [1e-6,1e-5,1e-4,1e-3,1e-2]
    best_pfa = None
    best_recall = 0.0
    for pfa in pfa_candidates:
        cfar_params = base_cfar_params.copy()
        cfar_params['pfa'] = pfa
        total_recall = 0.0
        total_latency_ms = 0.0
        total_proposals = 0
        for frame_idx, frame in enumerate(cached_frames):
            raw_adc_data = frame['radar_raw_frame'] if isinstance(frame, dict) else frame
            range_axis, velocity_axis, rd_angle_cube, azimuth_axis = run_dsp_pipeline(raw_adc_data)
            res_params = {
                'range_res': range_axis[1] - range_axis[0],
                'vel_res': velocity_axis[1] - velocity_axis[0],
                'az_res': azimuth_axis[1] - azimuth_axis[0]
            }
            t0 = time.perf_counter()
            detected_peaks = detect_3d_peaks(rd_angle_cube, res_params, cfar_params, algorithm=cfar_params['algorithm'])
            t1 = time.perf_counter()
            total_latency_ms += (t1 - t0) * 1000.0
            gt_targets = get_gt_for_frame(frame_idx, df_gt)
            recall = evaluate_detections(detected_peaks, gt_targets)
            total_recall += recall
            total_proposals += len(detected_peaks)
        avg_recall = total_recall / len(cached_frames)
        avg_latency_ms = total_latency_ms / len(cached_frames)
        avg_proposals = total_proposals / len(cached_frames)
        print(f"PFA: {pfa:.1e}, Avg Recall: {avg_recall:.4f}, Avg Latency: {avg_latency_ms:.2f} ms, Avg Proposals: {avg_proposals:.2f}")
        if avg_recall > best_recall:
            best_recall = avg_recall
            best_pfa = pfa
    if best_pfa is not None:
        print(f"Best PFA so far: {best_pfa:.1e} with Avg Recall: {best_recall:.4f}")
    else:
        print(f"No optimal PFA found with recall > 0. (Best Recall: {best_recall:.4f})") 