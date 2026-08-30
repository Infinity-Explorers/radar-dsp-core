"""
Unit tests for the radar DSP pipeline (src/dsp/*).

Design notes
------------
* All tests use synthetic, in-memory ADC data. Nothing here touches the
  network or the Hugging Face dataset, so the suite is fully hermetic and
  deterministic (a fixed RNG seed is used wherever randomness is needed).

* `src/dsp/pipeline.py` is intentionally NOT imported directly. It imports
  `src.data_access.hf_client`, which pulls in `huggingface_hub`/`scipy.io`
  purely to *fetch* frames from the Hugging Face Hub -- a dependency that
  has nothing to do with the correctness of the DSP math and that would
  make these tests fail on a machine without that package installed / with
  no network access. Instead, `_run_full_pipeline()` below chains the same
  four processing functions, in the same order and with the same
  parameters, that `run_dsp_pipeline()` uses -- so the shape/NaN tests
  below still exercise the exact pipeline logic, just without the network
  coupling.

* Raw ADC buffer layout: the task brief describes it as
  `(num_chirps, num_rx, num_adc_samples)`. This project's actual
  convention -- as returned by `src.data_access.hf_client.get_frame()` and
  as consumed by `process_range_fft` (windows/FFTs axis 0) and
  `process_doppler_fft` (windows/FFTs axis 1) -- is instead
  `(num_adc_samples, num_chirps, num_rx, num_tx)`. The tests below verify
  the layout the code actually relies on, and this docstring flags the
  discrepancy explicitly rather than silently testing something the
  pipeline doesn't use.

* Real-data integration test: a `.bin` sample frame was requested, but this
  project has no `.bin` data path anywhere. Its only real-data source is a
  Hugging Face Hub dataset of MATLAB `.mat` files, accessed via
  `src.data_access.hf_client.get_frame()`. `TestRealDataIntegration` (at the
  bottom of this file) exercises that actual real-data path end-to-end and
  skips -- explicitly, with a reason -- if the dataset can't be reached in
  the current environment, instead of fabricating a fake `.bin` file.
"""

import numpy as np
import pytest

from src.configs import (
    numAdcSamples,
    numLoops,
    bandwidth_GHz,
    chirpDuration_usec,
    startFreqConst_GHz,
)
from src.dsp.windowing import apply_window
from src.dsp.range_fft import process_range_fft
from src.dsp.clutter_removal import remove_static_clutter
from src.dsp.doppler_fft import process_doppler_fft
from src.dsp.angle import angle_fft

# --------------------------------------------------------------------------
# Project-wide constants
# --------------------------------------------------------------------------
# Antenna counts aren't exposed in src/configs.py, but they're fixed by the
# dataset / src/dsp/angle.py ("Merge 4 Rx and 2 Tx into 8 virtual antenna
# channels") and by docs/clutter_velocity_and_angle.md.
NUM_RX = 4
NUM_TX = 2
NUM_ANGLE_BINS = 64

EXPECTED_RANGE_BINS = numAdcSamples // 2          # 64
EXPECTED_CHIRPS = numLoops                        # 255
EXPECTED_CUBE_SHAPE = (EXPECTED_RANGE_BINS, EXPECTED_CHIRPS, NUM_ANGLE_BINS)  # (64, 255, 64)

RNG_SEED = 1234


def _run_full_pipeline(raw_cube):
    """Mirror src/dsp/pipeline.run_dsp_pipeline() step-for-step, without
    routing through the network-backed src.data_access.hf_client import."""
    B = bandwidth_GHz * 1e9
    Tc = chirpDuration_usec * 1e-6
    carrier_freq = startFreqConst_GHz * 1e9

    range_spectrum, range_axis = process_range_fft(
        raw_cube, Tc=Tc, B=B, window_type="hann"
    )
    clean_range_spectrum = remove_static_clutter(range_spectrum)
    doppler_spectrum, velocity_axis = process_doppler_fft(
        clean_range_spectrum, Tc=Tc, window_type="hann", axis=1
    )
    rd_angle_cube, azimuth_axis = angle_fft(
        doppler_spectrum, num_angle_bins=NUM_ANGLE_BINS, carrier_frequency=carrier_freq
    )
    return range_spectrum, clean_range_spectrum, doppler_spectrum, range_axis, velocity_axis, rd_angle_cube, azimuth_axis


@pytest.fixture
def synthetic_raw_frame():
    """A realistically-sized synthetic raw ADC cube, matching the shape the
    pipeline actually expects: (num_adc_samples, num_chirps, num_rx, num_tx)."""
    rng = np.random.default_rng(RNG_SEED)
    shape = (numAdcSamples, numLoops, NUM_RX, NUM_TX)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


# ==========================================================================
# 1. Tensor Shape & Type Verification
# ==========================================================================
class TestShapeAndType:
    def test_raw_adc_buffer_matches_expected_input_layout(self, synthetic_raw_frame):
        assert synthetic_raw_frame.ndim == 4
        assert synthetic_raw_frame.shape == (numAdcSamples, numLoops, NUM_RX, NUM_TX)
        assert np.iscomplexobj(synthetic_raw_frame)

    def test_final_radar_cube_shape_is_64_255_64(self, synthetic_raw_frame):
        *_ , rd_angle_cube, _ = _run_full_pipeline(synthetic_raw_frame)
        assert rd_angle_cube.shape == EXPECTED_CUBE_SHAPE

    def test_final_radar_cube_has_no_nan_or_inf(self, synthetic_raw_frame):
        *_ , rd_angle_cube, _ = _run_full_pipeline(synthetic_raw_frame)
        assert np.all(np.isfinite(rd_angle_cube))
        assert not np.any(np.isnan(rd_angle_cube))
        assert not np.any(np.isinf(rd_angle_cube))

    def test_intermediate_stage_shapes_are_consistent(self, synthetic_raw_frame):
        range_spectrum, clean_range_spectrum, doppler_spectrum, range_axis, velocity_axis, rd_angle_cube, azimuth_axis = (
            _run_full_pipeline(synthetic_raw_frame)
        )
        expected_rd_shape = (EXPECTED_RANGE_BINS, EXPECTED_CHIRPS, NUM_RX, NUM_TX)
        assert range_spectrum.shape == expected_rd_shape
        assert clean_range_spectrum.shape == expected_rd_shape
        assert doppler_spectrum.shape == expected_rd_shape

        assert range_axis.shape == (EXPECTED_RANGE_BINS,)
        assert velocity_axis.shape == (EXPECTED_CHIRPS,)
        assert azimuth_axis.shape == (NUM_ANGLE_BINS,)


# ==========================================================================
# 2. DC Offset & Clutter Removal
# ==========================================================================
class TestClutterRemoval:
    def test_output_has_zero_mean_across_chirp_axis(self):
        rng = np.random.default_rng(RNG_SEED)
        shape = (6, 40, NUM_RX, NUM_TX)
        range_matrix = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

        clean = remove_static_clutter(range_matrix)

        assert clean.shape == range_matrix.shape
        assert np.allclose(clean.mean(axis=1), 0.0, atol=1e-9)

    def test_dc_spike_is_suppressed_while_moving_target_is_preserved(self):
        """A strong, constant (zero-Doppler / DC) reflection combined with a
        weaker moving target at a known nonzero Doppler bin. Clutter removal
        should crush the DC bin's power while leaving the moving target's
        power essentially untouched."""
        M = 64
        static_amplitude = 50.0 + 0j       # strong static clutter component
        moving_bin = 6                     # exact FFT bin -> nonzero Doppler
        moving_amplitude = 1.0 + 0j

        m = np.arange(M)
        signal = static_amplitude + moving_amplitude * np.exp(1j * 2 * np.pi * moving_bin * m / M)
        range_matrix = signal.reshape(1, M, 1, 1)

        clean = remove_static_clutter(range_matrix)

        raw_spectrum = np.fft.fft(range_matrix[0, :, 0, 0])
        clean_spectrum = np.fft.fft(clean[0, :, 0, 0])

        dc_power_before = np.abs(raw_spectrum[0]) ** 2
        dc_power_after = np.abs(clean_spectrum[0]) ** 2
        target_power_before = np.abs(raw_spectrum[moving_bin]) ** 2
        target_power_after = np.abs(clean_spectrum[moving_bin]) ** 2

        # DC/zero-Doppler spike should be suppressed by many orders of magnitude.
        assert dc_power_after < dc_power_before * 1e-6
        assert dc_power_after == pytest.approx(0.0, abs=1e-6)

        # The moving target's Doppler bin should be essentially unaffected.
        assert target_power_after == pytest.approx(target_power_before, rel=1e-6)


# ==========================================================================
# 3. Synthetic Single-Tone Verification (Peak Localization)
# ==========================================================================
class TestSyntheticTonePeakLocalization:
    N_SAMPLES = 64   # fast-time (must be even so half_N = N_SAMPLES // 2)
    M_CHIRPS = 32    # slow-time
    R_BIN = 7        # target's range FFT bin (< N_SAMPLES // 2)
    D_BIN = 5        # target's Doppler FFT bin (pre-fftshift)

    def _make_synthetic_target_cube(self):
        """A complex sinusoid simulating a single target at a known range
        bin (fast-time frequency) and known Doppler bin (slow-time
        frequency), broadcast identically across all Rx/Tx channels."""
        n = np.arange(self.N_SAMPLES).reshape(self.N_SAMPLES, 1, 1, 1)
        m = np.arange(self.M_CHIRPS).reshape(1, self.M_CHIRPS, 1, 1)
        tone = (
            np.exp(1j * 2 * np.pi * self.R_BIN * n / self.N_SAMPLES)
            * np.exp(1j * 2 * np.pi * self.D_BIN * m / self.M_CHIRPS)
        )
        return np.broadcast_to(tone, (self.N_SAMPLES, self.M_CHIRPS, 2, 1)).copy()

    def test_range_fft_peak_lands_on_expected_bin(self):
        raw_cube = self._make_synthetic_target_cube()
        range_spectrum, _ = process_range_fft(
            raw_cube, Tc=1e-6, B=1e9, Fs=self.N_SAMPLES * 1e6, window_type="hann"
        )
        power = np.abs(range_spectrum[:, 0, 0, 0]) ** 2
        assert range_spectrum.shape[0] == self.N_SAMPLES // 2
        assert np.argmax(power) == self.R_BIN

    def test_range_doppler_fft_peak_lands_on_expected_bin(self):
        raw_cube = self._make_synthetic_target_cube()
        range_spectrum, _ = process_range_fft(
            raw_cube, Tc=1e-6, B=1e9, Fs=self.N_SAMPLES * 1e6, window_type="hann"
        )
        doppler_spectrum, velocity_axis = process_doppler_fft(
            range_spectrum, Tc=1e-6, window_type="hann", axis=1
        )

        power_map = np.abs(doppler_spectrum[:, :, 0, 0]) ** 2
        peak_range_idx, peak_doppler_idx = np.unravel_index(
            np.argmax(power_map), power_map.shape
        )

        # Doppler axis is fftshifted by process_doppler_fft; derive the
        # expected post-shift bin index the same way the function does,
        # rather than re-deriving the shift formula by hand.
        expected_doppler_idx = int(
            np.where(np.fft.fftshift(np.arange(self.M_CHIRPS)) == self.D_BIN)[0][0]
        )

        assert peak_range_idx == self.R_BIN
        assert peak_doppler_idx == expected_doppler_idx
        assert velocity_axis.shape == (self.M_CHIRPS,)


# ==========================================================================
# 4. Windowing Verification
# ==========================================================================
class TestWindowing:
    def test_apply_window_preserves_input_shape(self):
        data = np.ones((16, 5, 3), dtype=complex)
        for window_type in ("hann", "blackman"):
            windowed = apply_window(data, window_type=window_type, axis=0)
            assert windowed.shape == data.shape

    def test_apply_window_invalid_type_raises_value_error(self):
        data = np.ones((16, 3), dtype=complex)
        with pytest.raises(ValueError):
            apply_window(data, window_type="not_a_real_window", axis=0)

    @pytest.mark.parametrize("window_type", ["hann", "blackman"])
    def test_window_does_not_shift_peak_and_attenuates_far_sidelobes(self, window_type):
        """Use a tone that does NOT land exactly on an FFT bin, so the
        rectangular (unwindowed) case exhibits real spectral leakage. The
        main-lobe peak bin should be identical with and without windowing,
        while a bin further out in the sidelobe region should show clearly
        lower power once windowed."""
        N = 64
        k0 = 12.25  # non-integer bin -> genuine leakage in the unwindowed FFT
        n = np.arange(N)
        tone = np.exp(1j * 2 * np.pi * k0 * n / N).reshape(N, 1)

        unwindowed_power = np.abs(np.fft.fft(tone, axis=0)[:, 0]) ** 2
        windowed = apply_window(tone, window_type=window_type, axis=0)
        windowed_power = np.abs(np.fft.fft(windowed, axis=0)[:, 0]) ** 2

        peak_unwindowed = np.argmax(unwindowed_power)
        peak_windowed = np.argmax(windowed_power)
        assert peak_windowed == peak_unwindowed

        far_bin = (int(peak_unwindowed) + 6) % N
        assert windowed_power[far_bin] < unwindowed_power[far_bin]


# ==========================================================================
# 5. Angle FFT -- Virtual Antenna Array Phase Synthesis & Peak Localization
# ==========================================================================
class TestAngleFFTPeakLocalization:
    """Dedicated unit test for src.dsp.angle.angle_fft.

    angle_fft's real signature is:
        angle_fft(doppler_spectrum, num_angle_bins=64, carrier_frequency=...)
    with doppler_spectrum shape (range_bins, chirps, receivers, transmitters).
    Internally it reshapes the last two axes into 8 virtual antenna channels
    (n_rx * n_tx, row-major: virtual index v = rx * n_tx + tx), applies a
    Hann window across that virtual-antenna axis, then FFTs across it
    (axis=2) and fftshifts. There is no `axis=` argument to pass -- the
    spatial axis is fixed internally by the function, so this test drives
    the function through its actual, real parameters only.
    """

    NUM_ANGLE_BINS = 64
    ANGLE_BIN = 20  # arbitrary target bin in [0, NUM_ANGLE_BINS - 1]

    def _make_synthetic_virtual_array(self):
        """Simulate a single far-field target arriving with a linear phase
        progression across the NUM_RX x NUM_TX = 8 virtual antenna channels:
        element v's phase = 2*pi * ANGLE_BIN * v / NUM_ANGLE_BINS. This is
        the discrete analogue of a planar wavefront hitting the array at the
        angle that maps exactly to bin ANGLE_BIN."""
        doppler_spectrum = np.zeros((1, 1, NUM_RX, NUM_TX), dtype=complex)
        for rx in range(NUM_RX):
            for tx in range(NUM_TX):
                v = rx * NUM_TX + tx  # matches angle_fft's row-major reshape
                doppler_spectrum[0, 0, rx, tx] = np.exp(
                    1j * 2 * np.pi * self.ANGLE_BIN * v / self.NUM_ANGLE_BINS
                )
        return doppler_spectrum

    def test_angle_fft_output_shape(self):
        doppler_spectrum = self._make_synthetic_virtual_array()
        angle_spectrum, azimuth_axis = angle_fft(
            doppler_spectrum, num_angle_bins=self.NUM_ANGLE_BINS
        )
        assert angle_spectrum.shape == (1, 1, self.NUM_ANGLE_BINS)
        assert azimuth_axis.shape == (self.NUM_ANGLE_BINS,)

    def test_angle_fft_peak_lands_on_expected_angle_bin(self):
        """For a virtual-array phase progression that exactly matches one of
        the num_angle_bins DFT frequencies, the windowed spatial FFT is
        mathematically guaranteed to peak at that exact bin: the aligned bin
        maximizes |sum_v w[v] * exp(-j*theta*v)| by the triangle inequality,
        for any nonnegative real window w[v]. So this also confirms the Hann
        window applied inside angle_fft does not shift the angle peak."""
        doppler_spectrum = self._make_synthetic_virtual_array()
        angle_spectrum, azimuth_axis = angle_fft(
            doppler_spectrum, num_angle_bins=self.NUM_ANGLE_BINS
        )

        power = np.abs(angle_spectrum[0, 0, :]) ** 2
        peak_bin = int(np.argmax(power))

        # angle_fft fftshifts the spatial FFT axis; derive the expected
        # post-shift bin the same way the function does, rather than
        # re-deriving the shift formula by hand.
        expected_bin = int(
            np.where(np.fft.fftshift(np.arange(self.NUM_ANGLE_BINS)) == self.ANGLE_BIN)[0][0]
        )
        assert peak_bin == expected_bin

        # The peak should clearly dominate an unrelated, distant bin.
        far_bin = (peak_bin + self.NUM_ANGLE_BINS // 2) % self.NUM_ANGLE_BINS
        assert power[peak_bin] > power[far_bin]

        # The azimuth angle reported at the peak must be a real, finite
        # value within [-90, 90] degrees (azimuth is NaN only for spatial
        # frequencies outside the visible |sin(theta)| <= 1 range).
        assert np.isfinite(azimuth_axis[peak_bin])
        assert -90.0 <= azimuth_axis[peak_bin] <= 90.0


# ==========================================================================
# 6. Real-Data Integration Test
# ==========================================================================
# This project has no ".bin" frames anywhere. Its only real-data path is
# src.data_access.hf_client.get_frame(), which downloads a MATLAB ".mat"
# file (key "adcData") from the Hugging Face Hub dataset
# "hany34/raw-adc-data-77ghz-mmwave-radar-automotive-object-detection" and
# returns an array shaped (samples, chirps, receivers, transmitters). No
# sample frame of any kind is checked into this repository. Rather than
# fabricate a fake ".bin" file and call that "real data", this test drives
# the project's actual real-data entry point end-to-end
# (get_frame -> run_dsp_pipeline) and skips -- explicitly, with a reason --
# if the dataset can't be reached in this environment (huggingface_hub not
# installed, no network access, dataset unavailable/gated, etc.), rather
# than silently substituting synthetic data for it.
class TestRealDataIntegration:
    def test_end_to_end_pipeline_on_real_dataset_frame(self):
        try:
            from src.data_access.hf_client import get_frame
        except ImportError as exc:
            pytest.skip(f"huggingface_hub not installed; cannot reach real dataset: {exc}")

        try:
            raw_frame = get_frame(0)
        except Exception as exc:  # network / auth / dataset-availability errors
            pytest.skip(f"Real dataset unavailable in this environment: {exc}")

        from src.dsp.pipeline import run_dsp_pipeline

        assert raw_frame.ndim == 4
        assert raw_frame.shape[0] == numAdcSamples
        assert raw_frame.shape[1] == numLoops

        range_axis, velocity_axis, rd_angle_cube, azimuth_axis = run_dsp_pipeline(raw_frame)

        assert rd_angle_cube.shape == EXPECTED_CUBE_SHAPE
        assert not np.any(np.isnan(rd_angle_cube))
        assert not np.any(np.isinf(rd_angle_cube))
        assert np.any(np.abs(rd_angle_cube) > 0)  # valid, non-trivial spectral data

        assert range_axis.shape == (EXPECTED_RANGE_BINS,)
        assert velocity_axis.shape == (EXPECTED_CHIRPS,)
        assert azimuth_axis.shape == (NUM_ANGLE_BINS,)