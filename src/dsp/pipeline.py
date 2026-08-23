
import numpy as np

from src.data_access.hf_client import get_frame , frame_stream 
from src.dsp.range_fft import process_range_fft
from src.dsp.clutter_removal import remove_static_clutter
from src.dsp.doppler_fft import process_doppler_fft
from src.dsp.angle import angle_fft
from src.configs import bandwidth_GHz , chirpDuration_usec ,startFreqConst_GHz

def run_dsp_pipeline(raw_frame):

    B = bandwidth_GHz * 1e9          # Bandwidth in Hz
    Tc = chirpDuration_usec * 1e-6    # Chirp duration in seconds
    carrier_freq = startFreqConst_GHz * 1e9  # starting frequency (fc) in Hz

    # Step 1: Range FFT 
    range_spectrum, range_axis = process_range_fft(
        raw_frame, 
        Tc=Tc, 
        B=B, 
        window_type='hann', 
    )

    # Step 2: Static Clutter Removal
    clean_range_spectrum = remove_static_clutter(range_spectrum)

    # Step 3: Doppler FFT
    doppler_spectrum, velocity_axis = process_doppler_fft(
        clean_range_spectrum, 
        Tc=Tc, 
        window_type='hann', 
        axis=1
    )

    # Step 4: Angle FFT 
    rd_angle_cube, azimuth_axis = angle_fft(
        doppler_spectrum, 
        num_angle_bins=64,
        carrier_frequency=carrier_freq
    )

    return  range_axis, velocity_axis , rd_angle_cube , azimuth_axis


if __name__ == "__main__":
    print("Starting DSP pipeline...")

    for i, frame in enumerate(frame_stream(realtime=False)):
        print(f"Processing frame {i+1}...")
        range_axis, velocity_axis, rd_angle_cube, azimuth_axis = run_dsp_pipeline(frame)
        print(f"Frame {i+1} processed. Range axis shape: {range_axis.shape}, Velocity axis shape: {velocity_axis.shape}, RD-Angle cube shape: {rd_angle_cube.shape}, Azimuth axis shape: {azimuth_axis.shape}")
        if i >= 4:  # Process only the first 5 frames for demonstration
            break