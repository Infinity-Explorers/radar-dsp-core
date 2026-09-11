# this file is for extracting the data frames

from huggingface_hub import list_repo_files, hf_hub_download
from scipy.io import loadmat
import numpy as np
import time

REPO_ID = "hany34/raw-adc-data-77ghz-mmwave-radar-automotive-object-detection"
REPO_TYPE = "dataset"

_frame_paths = None 

def _load_frame_index() -> list[str]:
    global _frame_paths
    if _frame_paths is None:
        all_files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
        _frame_paths = sorted(f for f in all_files if "radar_raw_frame" in f and f.endswith(".mat"))
    return _frame_paths

# total number of frames available in the data
def num_frames() -> int:
    return len(_load_frame_index())

# extracting a single frame as an array of shape (samples, chirps, receivers, transmitters)
def get_frame(index: int) -> np.ndarray:
    paths = _load_frame_index()
    local_path = hf_hub_download(repo_id=REPO_ID, filename=paths[index], repo_type=REPO_TYPE)
    mat = loadmat(local_path)
    key = "adcData"
    if key not in mat:
        raise KeyError(f"Expected key '{key}' not found in {paths[index]}. Found: {list(mat.keys())}")
    return mat[key]

FRAME_PERIOD_S = 33.33333 / 1000  # from dataset config: framePeriodicity_msec, 30 fps
def frame_stream(realtime: bool = False, frame_period_s: float = FRAME_PERIOD_S):
    # yields a stream of frames
    # realtime=False (default): yields as fast as possible, for DSP testing.
    # realtime=True: yields the data with delays in between, simulating a live feed
    for i in range(num_frames()):
        yield get_frame(i)
        if realtime:
            time.sleep(frame_period_s)

# this can be used as the next example

# from src.data_access.hf_client import frame_stream

# for frame in frame_stream():
#     result = my_dsp_pipeline(frame)   # windowing -> range FFT -> etc.