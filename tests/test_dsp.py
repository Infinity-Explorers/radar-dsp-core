import pytest
import numpy as np

from src.dsp.windowing import apply_window
from src.dsp.range_fft import process_range_fft
from src.dsp.clutter_removal import remove_static_clutter
from src.dsp.doppler_fft import process_doppler_fft
from src.dsp.angle import angle_fft
from src.dsp.pipeline import run_dsp_pipeline


# =====================================================================
# 1. Tensor Shape & Type Verification
# =====================================================================
def test_tensor_shape_and_type():
    # Input format: (num_adc_samples=510, num_chirps=64, num_rx=4, num_tx=2)
    n_adc, n_chirps, n_rx, n_tx = 510, 64, 4, 2
    raw_frame = np.random.randn(n_adc, n_chirps, n_rx, n_tx) + 1j * np.random.randn(n_adc, n_chirps, n_rx, n_tx)

    range_axis, velocity_axis, rd_angle_cube, azimuth_axis = run_dsp_pipeline(raw_frame)

    # Half-spectrum from 510 ADC samples yields 255 range bins
    # Shape: (range_bins=255, chirps=64, angle_bins=64)
    assert rd_angle_cube.shape == (255, 64, 64) # [source: 2, 6]
    assert range_axis.shape == (255,) # [source: 6]
    assert velocity_axis.shape == (64,) # [source: 4]
    assert azimuth_axis.shape == (64,) # [source: 2]

    # Validate finite numerical content
    assert not np.isnan(rd_angle_cube).any(), "Output radar cube contains NaN values"
    assert not np.isinf(rd_angle_cube).any(), "Output radar cube contains Inf values"


# =====================================================================
# 2. DC Offset & Clutter Removal
# =====================================================================
def test_static_clutter_removal():
    r_bins, n_chirps, n_rx, n_tx = 255, 64, 4, 2
    # Create dynamic signal with strong static baseline across chirps (axis 1)
    dynamic_signal = np.random.randn(r_bins, n_chirps, n_rx, n_tx)
    static_dc_component = np.ones((r_bins, 1, n_rx, n_tx)) * 50.0
    range_matrix = dynamic_signal + static_dc_component

    clean_matrix = remove_static_clutter(range_matrix)

    # Mean across chirp axis (axis=1) must evaluate to 0
    mean_across_slow_time = np.mean(clean_matrix, axis=1)
    assert np.allclose(mean_across_slow_time, 0.0, atol=1e-6)
    assert clean_matrix.shape == range_matrix.shape


# =====================================================================
# 3. Synthetic Single-Tone Verification (Peak Localization)
# =====================================================================
def test_synthetic_single_tone_peak_localization():
    n_adc, n_chirps, n_rx, n_tx = 512, 64, 4, 2
    target_range_bin = 45
    target_doppler_bin = 16

    n = np.arange(n_adc)[:, None, None, None]
    k = np.arange(n_chirps)[None, :, None, None]

    # Synthesize pure complex tone across fast-time (range) and slow-time (Doppler)
    fast_time_signal = np.exp(1j * 2 * np.pi * target_range_bin * n / n_adc)
    slow_time_signal = np.exp(1j * 2 * np.pi * target_doppler_bin * k / n_chirps)
    synthetic_cube = np.tile(fast_time_signal * slow_time_signal, (1, 1, n_rx, n_tx))

    # 1. Test Range FFT Peak
    range_spectrum, _ = process_range_fft(synthetic_cube, window_type='hann') # [source: 6]
    detected_range_bin = np.argmax(np.abs(range_spectrum[:, 0, 0, 0])) # [source: 6]
    assert detected_range_bin == target_range_bin, f"Range peak mismatch: expected {target_range_bin}, got {detected_range_bin}"

    # 2. Test Doppler FFT Peak (without fftshift to check positive bin directly)
    clean_cube = remove_static_clutter(range_spectrum)
    doppler_spectrum, _ = process_doppler_fft(clean_cube, window_type='hann', axis=1)

    # Shifted center bin offset check
    center_bin = n_chirps // 2
    expected_shifted_bin = center_bin + target_doppler_bin if target_doppler_bin < center_bin else target_doppler_bin - center_bin
    detected_doppler_bin = np.argmax(np.abs(doppler_spectrum[target_range_bin, :, 0, 0]))
    assert detected_doppler_bin == expected_shifted_bin, f"Doppler peak mismatch: expected {expected_shifted_bin}, got {detected_doppler_bin}"


# =====================================================================
# 4. Windowing Verification (Sidelobe Attenuation & Peak Preservation)
# =====================================================================
@pytest.mark.parametrize("window_type", ["hann", "blackman"])
def test_windowing_attenuation_and_peak_preservation(window_type):
    n_samples = 256
    # Use an off-grid fractional bin to induce spectral leakage
    target_bin = 30.5
    n = np.arange(n_samples)
    pure_tone = np.exp(1j * 2 * np.pi * target_bin * n / n_samples)

    # Apply window via src.dsp.windowing
    windowed_signal = apply_window(pure_tone, window_type=window_type, axis=0)

    fft_rect = np.abs(np.fft.fft(pure_tone))
    fft_win = np.abs(np.fft.fft(windowed_signal))

    # 1. Peak location must remain at the closest integer bin
    assert np.argmax(fft_win) == np.argmax(fft_rect) == int(round(target_bin))

    # 2. Far sidelobe (8 bins away from mainlobe) must be attenuated relative to mainlobe
    sidelobe_idx = int(round(target_bin)) + 8
    pslr_rect = fft_rect[sidelobe_idx] / np.max(fft_rect)
    pslr_win = fft_win[sidelobe_idx] / np.max(fft_win)

    assert pslr_win < pslr_rect, f"Window {window_type} failed to attenuate sidelobes"