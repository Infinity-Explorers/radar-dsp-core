# `tests/test_dsp.py` — Documentation

This document explains, function by function, **what the DSP pipeline code does**, **what the
test suite calls / expects / asserts**, and **why each assertion is the correct thing to check**.
It's meant to let any team member read the tests without re-deriving the math, and to make it
safe to extend or modify either the tests or the pipeline itself.

There are two halves to this document:

1. **The functions under test** (`src/dsp/*.py`, `src/data_access/hf_client.py`) — signatures,
   inputs, outputs, side effects.
2. **The tests themselves** (`tests/test_dsp.py`) — for every test: what it builds, what it
   calls, what it checks, and what "correct" means for that check.

---

## 1. Pipeline architecture (what's being tested)

```
raw ADC cube                 (num_adc_samples, num_chirps, num_rx, num_tx)
      │
      ▼  process_range_fft()      windows + FFTs axis 0 (fast-time), keeps positive half
range_spectrum                (num_adc_samples/2, num_chirps, num_rx, num_tx)
      │
      ▼  remove_static_clutter()  subtracts the mean across axis 1 (chirps)
clean_range_spectrum          (num_adc_samples/2, num_chirps, num_rx, num_tx)
      │
      ▼  process_doppler_fft()    windows + FFTs axis 1 (slow-time), fftshifts it
doppler_spectrum              (num_adc_samples/2, num_chirps, num_rx, num_tx)
      │
      ▼  angle_fft()              reshapes (rx,tx) -> 8 virtual channels, windows + FFTs + fftshifts that axis
rd_angle_cube                 (num_adc_samples/2, num_chirps, num_angle_bins)
```

With this project's configured values (`src/configs.py`: `numAdcSamples=128`, `numLoops=255`)
and `num_angle_bins=64`, the final cube is **exactly `(64, 255, 64)`**. That number is not
hard-coded by hand anywhere in the tests — it's computed from `numAdcSamples // 2`, `numLoops`,
and `NUM_ANGLE_BINS`, so if a config value ever changes, the expected shape updates with it
automatically (see `EXPECTED_CUBE_SHAPE` in the test file, §3 below).

`run_dsp_pipeline()` in `src/dsp/pipeline.py` is just this chain wired together with the
project's real constants (bandwidth, chirp duration, carrier frequency) pulled from
`src/configs.py`.

---

## 2. Functions under test — signatures, inputs, outputs

### 2.1 `src.dsp.windowing.apply_window(data, window_type='hann', axis=0)`

| | |
|---|---|
| **Input** | `data`: any-shape numpy array (real or complex). `window_type`: `'hann'` or `'blackman'` (case-insensitive). `axis`: which axis to window along. |
| **What it does** | Builds a 1-D window of length `data.shape[axis]` (`np.hanning` or `np.blackman`), reshapes it to broadcast against `axis`, and returns `data * window` element-wise. |
| **Output** | Same shape and dtype-family as `data`. |
| **Failure mode** | Raises `ValueError("Unsupported window type: <name>")` for anything other than `'hann'`/`'blackman'`. |
| **Used by** | `process_range_fft` (axis 0), `process_doppler_fft` (axis 1), `angle_fft` (axis 2, hard-coded). |

### 2.2 `src.dsp.range_fft.process_range_fft(raw_cube, Tc=CHIRP_ACTIVE_DURATION, B=B_HZ, Fs=FS_HZ, window_type='hann')`

| | |
|---|---|
| **Input** | `raw_cube`: complex array, axis 0 = fast-time (ADC samples). `Tc`, `B`, `Fs`: chirp duration, bandwidth, sample rate (only affect the returned `range_axis` physical units, not the FFT itself). |
| **What it does** | 1. Windows `raw_cube` along axis 0. 2. `np.fft.fft(..., axis=0)`. 3. Keeps only the first half of the spectrum (`N // 2` bins) — the positive-frequency half, since beat-frequency signals from a real radar are one-sided. 4. Converts the corresponding frequency bins into physical range (meters) via `range_axis = freq_axis * c * Tc / (2*B)`. |
| **Output** | `(fft_out[:half_N, ...], range_axis)` — spectrum shape `(N//2, ...same trailing dims as input...)`, `range_axis` shape `(N//2,)`. |
| **Note** | Truncation to half the bins is why the final cube's first dimension is `numAdcSamples // 2 = 64`, not `128`. |

### 2.3 `src.dsp.clutter_removal.remove_static_clutter(range_matrix)`

| | |
|---|---|
| **Input** | `range_matrix` shape `(range_bins, chirps, receivers, transmitters)` (or any shape with chirps on axis 1). |
| **What it does** | `mean_clutter = mean(range_matrix, axis=1, keepdims=True)`; returns `range_matrix - mean_clutter`. This is a classic MTI (moving target indication) filter: anything that doesn't change chirp-to-chirp (i.e. zero-Doppler / stationary clutter) is exactly the per-range-bin mean, and subtracting it removes it. |
| **Output** | Same shape as input. By construction, the output's mean along axis 1 is (numerically) zero. |

### 2.4 `src.dsp.doppler_fft.process_doppler_fft(clutter_removed_cube, Tc=chirp_repetition_time, window_type='hann', axis=1)`

| | |
|---|---|
| **Input** | `clutter_removed_cube`: output of `remove_static_clutter`. `axis`: which axis is the chirp axis (project always passes `axis=1`). |
| **What it does** | 1. Windows along `axis`. 2. `np.fft.fft(..., axis=axis)`. 3. `np.fft.fftshift(..., axes=axis)` — re-centers so bin 0 (DC/zero-Doppler) moves to the middle of the axis instead of sitting at index 0. 4. Computes `velocity = (wavelength/2) * fftshift(fftfreq(n_chirps, d=Tc))`, i.e. the physical velocity axis matching each (now-shifted) bin. |
| **Output** | `(doppler_spectrum, velocity)` — spectrum same shape as input; `velocity` shape `(n_chirps,)`. |
| **Important gotcha** | Because of the `fftshift`, **bin index `k` in the raw FFT does *not* land at index `k` in the returned array.** Any test that plants a tone at a known pre-shift bin must re-derive the post-shift index the same way the function does — see §3.4/§3.5 below. Don't hand-derive the shift formula; use `np.fft.fftshift(np.arange(N))` and look up where your bin landed. |

### 2.5 `src.dsp.angle.angle_fft(doppler_spectrum, num_angle_bins=64, carrier_frequency=CARRIER_FREQUENCY)`

| | |
|---|---|
| **Input** | `doppler_spectrum` shape `(range_bins, chirps, receivers, transmitters)`. **No `axis` parameter** — the spatial axis is fixed internally. |
| **What it does** | 1. Reshapes the last two axes into one: `(r_bins, n_chirps, n_rx * n_tx)`. **Row-major**, so virtual channel index `v = rx * n_tx + tx` (this matters if you ever build a synthetic array by hand — get the index order wrong and your "known" phase progression won't be at the position you think). 2. Windows along the new axis 2 (hard-coded `'hann'`, not configurable through this function). 3. `np.fft.fft(..., n=num_angle_bins, axis=2)` — zero-pads/truncates to `num_angle_bins`. 4. `np.fft.fftshift(..., axes=2)`. 5. Computes `azimuth` (degrees) from `arcsin(spatial_frequency)`; bins whose implied `|sin(theta)| > 1` (non-physical angles) are set to `NaN` by design — this is expected, not a bug. |
| **Output** | `(angle_spectrum, azimuth)` — spectrum shape `(range_bins, chirps, num_angle_bins)`; `azimuth` shape `(num_angle_bins,)`, may contain `NaN` at the edges. |
| **Same fftshift gotcha as §2.4** applies here — tests must derive expected bin index via `fftshift`, not by hand. |

### 2.6 `src.dsp.pipeline.run_dsp_pipeline(raw_frame)`

Wires §2.2 → §2.3 → §2.4 → §2.5 together using this project's real constants from
`src/configs.py` (`B`, `Tc`, `carrier_freq`). Returns `(range_axis, velocity_axis,
rd_angle_cube, azimuth_axis)`.

**Not imported directly by most tests** — see §3.0 for why.

### 2.7 `src.data_access.hf_client.get_frame(index)` / `frame_stream(...)`

Downloads frame `index` (a `.mat` file, key `"adcData"`) from the Hugging Face Hub dataset
`hany34/raw-adc-data-77ghz-mmwave-radar-automotive-object-detection` and returns it as a numpy
array shaped `(samples, chirps, receivers, transmitters)`. `frame_stream()` is a generator that
yields every frame in the dataset, optionally paced in real time. **Requires `huggingface_hub`
and network access to `huggingface.co`.** This is the *only* real-sensor-data entry point in the
whole project — there is no `.bin` file format anywhere in this codebase.

---

## 3. `tests/test_dsp.py` — test-by-test reference

### 3.0 Module-level design decisions (read this before editing anything)

* **`_run_full_pipeline(raw_cube)`** (helper function, not a test) manually chains
  §2.2 → §2.3 → §2.4 → §2.5 with the same parameters `run_dsp_pipeline` uses, **instead of**
  calling `run_dsp_pipeline` directly. Reason: `pipeline.py` imports `hf_client.py`, which
  imports `huggingface_hub` at module load time. If that package isn't installed, *every* test
  in the file — even ones that have nothing to do with real data — would fail to collect. All
  shape/NaN/DC tests route through this helper so they stay hermetic (no network, no optional
  dependency). If you change `run_dsp_pipeline`'s step order or parameters, **update this
  helper to match**, or the tests will silently stop reflecting production behavior.
* **`NUM_RX = 4`, `NUM_TX = 2`, `NUM_ANGLE_BINS = 64`** are hard-coded constants at the top of
  the file because they aren't exposed in `src/configs.py` — they come from the dataset itself
  and from `angle_fft`'s default parameter / docstring. **If the real dataset's antenna count
  ever changes, update these constants.**
* **`EXPECTED_RANGE_BINS`, `EXPECTED_CHIRPS`, `EXPECTED_CUBE_SHAPE`** are *derived* from
  `src.configs.numAdcSamples` / `numLoops`, not hard-coded numbers — so `(64, 255, 64)` is a
  computed consequence of the config, not a magic literal duplicated in the test file.
* **`RNG_SEED = 1234`** — all randomized fixtures use `np.random.default_rng(RNG_SEED)` so test
  runs are reproducible. Don't switch to unseeded `np.random` calls.
* **Raw layout discrepancy**: the original task brief described the raw ADC buffer as
  `(num_chirps, num_rx, num_adc_samples)`. The code actually uses
  `(num_adc_samples, num_chirps, num_rx, num_tx)` (confirmed from `hf_client.get_frame`'s
  docstring and from which axis `process_range_fft`/`process_doppler_fft` operate on). The tests
  verify the layout the code *actually* relies on. If you ever see this project's raw layout
  described differently elsewhere, this file's docstring is the source of truth for what the
  code does today.
* **`.bin` vs `.mat` discrepancy**: a `.bin` integration test was requested at one point, but
  this project has no `.bin` data path anywhere — only `.mat` files via `hf_client.get_frame()`.
  §3.7 documents how that's handled.

### 3.1 Fixture: `synthetic_raw_frame()`

* **Builds**: a random complex array, shape `(numAdcSamples, numLoops, NUM_RX, NUM_TX)` =
  `(128, 255, 4, 2)`, via `rng.standard_normal(shape) + 1j*rng.standard_normal(shape)`.
* **Used by**: every test in `TestShapeAndType`.
* **Why random (not a tone)**: these tests only care about *shape* and *finiteness*, not about
  where energy ends up — random data is the right tool because it can't accidentally hide a
  shape bug behind a lucky cancellation the way a specially-constructed signal might.

### 3.2 `class TestShapeAndType`

| Test | Calls | Expects | Comparing against |
|---|---|---|---|
| `test_raw_adc_buffer_matches_expected_input_layout` | nothing (just inspects the fixture) | `synthetic_raw_frame.shape == (numAdcSamples, numLoops, NUM_RX, NUM_TX)`, `ndim==4`, complex dtype | The layout the real pipeline functions actually consume (see §3.0 discrepancy note) |
| `test_final_radar_cube_shape_is_64_255_64` | `_run_full_pipeline()` | `rd_angle_cube.shape == EXPECTED_CUBE_SHAPE` | `EXPECTED_CUBE_SHAPE`, derived from config (§3.0) — not the literal `(64,255,64)` typed twice |
| `test_final_radar_cube_has_no_nan_or_inf` | `_run_full_pipeline()` | `np.isfinite` true everywhere; no NaN; no Inf | IEEE-754 finiteness — catches divide-by-zero / log-of-zero / overflow bugs anywhere upstream |
| `test_intermediate_stage_shapes_are_consistent` | `_run_full_pipeline()` | `range_spectrum`, `clean_range_spectrum`, `doppler_spectrum` all `(64, 255, 4, 2)`; `range_axis` `(64,)`; `velocity_axis` `(255,)`; `azimuth_axis` `(64,)` | Shapes computed from the same config constants — regression-proofs every intermediate stage, not just the final one, so a shape bug is caught at the stage that introduced it |

### 3.3 `class TestClutterRemoval`

| Test | Builds | Calls | Expects | Comparing against |
|---|---|---|---|---|
| `test_output_has_zero_mean_across_chirp_axis` | random complex `(6, 40, NUM_RX, NUM_TX)` | `remove_static_clutter()` | `clean.mean(axis=1) ≈ 0` (`atol=1e-9`) | The mathematical definition of the function itself — mean subtraction by construction zeroes the mean, for *any* input, not just a hand-picked one |
| `test_dc_spike_is_suppressed_while_moving_target_is_preserved` | `signal = 50 (static/DC) + 1·exp(j·2π·6·m/64) (moving target)` reshaped to `(1, 64, 1, 1)` | `remove_static_clutter()`, then an independent `np.fft.fft` on both the raw and cleaned signal (**not** `process_doppler_fft` — a plain, un-windowed, un-shifted FFT is used deliberately here so the "DC bin" is unambiguously index 0, with no fftshift bookkeeping to get in the way) | DC-bin power drops by more than 6 orders of magnitude (`< before * 1e-6`) and is `≈ 0`; moving-target-bin power is unchanged (`rel=1e-6`) | Before/after comparison of the *same* signal — isolates exactly what clutter removal is supposed to do (kill zero-Doppler, leave everything else alone) |

### 3.4 `class TestSyntheticTonePeakLocalization`

Class constants: `N_SAMPLES=64` (fast-time), `M_CHIRPS=32` (slow-time), `R_BIN=7` (target range
bin), `D_BIN=5` (target Doppler bin, pre-shift). These are deliberately smaller/different from
the project's real `numAdcSamples`/`numLoops` — this class is testing FFT *math*, not
production shapes, so small convenient numbers keep the test fast and easy to reason about by
hand. (Production shapes are covered separately in §3.2.)

* **`_make_synthetic_target_cube()`** (helper): builds
  `exp(j·2π·R_BIN·n/N_SAMPLES) · exp(j·2π·D_BIN·m/M_CHIRPS)`, a 2-D complex tone representing a
  single point target at an exact range bin and an exact Doppler bin, broadcast identically
  across 2 Rx × 1 Tx channels (channel count doesn't matter for this test, only range/chirp
  axes do).

| Test | Calls | Expects | Comparing against |
|---|---|---|---|
| `test_range_fft_peak_lands_on_expected_bin` | `process_range_fft()` on the synthetic cube | `argmax(power) == R_BIN`; output has `N_SAMPLES//2` range bins | Ground truth: the tone was *constructed* at bin `R_BIN`, so the FFT must recover exactly that bin (an integer-cycle tone produces a perfect, non-leaking spectral line) |
| `test_range_doppler_fft_peak_lands_on_expected_bin` | `process_range_fft()` then `process_doppler_fft()` | 2-D `argmax` lands at `(R_BIN, expected_doppler_idx)` | `expected_doppler_idx` is **not** hand-computed from a shift formula — it's derived by asking `np.fft.fftshift` itself where bin `D_BIN` ends up: `np.where(np.fft.fftshift(np.arange(M_CHIRPS)) == D_BIN)[0][0]`. This means the test can never silently drift out of sync with however `np.fft.fftshift` actually behaves (e.g. across numpy versions or even/odd `M_CHIRPS`) |

### 3.5 `class TestAngleFFTPeakLocalization`

Class constants: `NUM_ANGLE_BINS=64`, `ANGLE_BIN=20` (arbitrary target angle bin).

* **`_make_synthetic_virtual_array()`** (helper): builds a `(1, 1, NUM_RX, NUM_TX)` array where
  element `[0,0,rx,tx] = exp(j·2π·ANGLE_BIN·v/NUM_ANGLE_BINS)`, with `v = rx*NUM_TX + tx` — this
  **must** match `angle_fft`'s internal reshape order (§2.5), or the "known" phase progression
  won't actually be linear across the function's virtual-channel axis and the peak won't land
  where expected.

| Test | Calls | Expects | Comparing against |
|---|---|---|---|
| `test_angle_fft_output_shape` | `angle_fft()` | `angle_spectrum.shape == (1,1,NUM_ANGLE_BINS)`; `azimuth_axis.shape == (NUM_ANGLE_BINS,)` | `angle_fft`'s documented output shape |
| `test_angle_fft_peak_lands_on_expected_angle_bin` | `angle_fft()` | `argmax(power) == expected_bin` (again derived via `fftshift(arange(...))`, same technique as §3.4); peak power `>` power at the diametrically opposite bin; `azimuth_axis[peak_bin]` is finite and in `[-90, 90]` | **Mathematical guarantee, not a numerical coincidence**: for a virtual-array phase exactly aligned to one DFT bin, `\|Σ_v w[v]·exp(-jθv)\|` is maximized at that bin for *any* nonnegative real window `w[v]`, by the triangle inequality. This is why the test can assert an *exact* bin match rather than "approximately near" — and it's also an implicit proof that the Hann window `angle_fft` applies internally doesn't shift the peak |

### 3.6 `class TestWindowing`

| Test | Calls | Expects | Comparing against |
|---|---|---|---|
| `test_apply_window_preserves_input_shape` | `apply_window()` with `'hann'` and `'blackman'` on `(16,5,3)` ones | output shape unchanged | Trivial invariant — windowing must never resize the array |
| `test_apply_window_invalid_type_raises_value_error` | `apply_window(..., window_type="not_a_real_window")` | raises `ValueError` | The function's documented failure mode (§2.1) |
| `test_window_does_not_shift_peak_and_attenuates_far_sidelobes[hann/blackman]` | plain `np.fft.fft` on an **off-bin** tone (`k0=12.25`, deliberately *not* an integer bin — see note below) with and without `apply_window()` | peak bin identical windowed vs. unwindowed; power at a "far" bin (`peak+6`) is lower when windowed | The unwindowed spectrum itself, computed in the same test — a true before/after comparison |

**Why `k0=12.25` and not an integer bin:** an *exact*-bin tone has essentially zero energy in any
other bin already (a perfect delta function) — there'd be nothing for a window to "attenuate,"
and windowing such a tone actually *adds* sidelobe energy from a numerical-zero floor (this was
verified by hand while writing the test and is *not* what "windowing reduces leakage" means in
practice). Using a **non-integer**-bin tone creates genuine spectral leakage in the unwindowed
case, which is the realistic scenario windowing is meant to help with. If you ever "simplify"
this test back to an integer-bin tone, the sidelobe-attenuation assertion will likely start
failing (or worse, pass for the wrong reason) — keep it non-integer.

### 3.7 `class TestRealDataIntegration`

| Test | Calls | Expects | Comparing against |
|---|---|---|---|
| `test_end_to_end_pipeline_on_real_dataset_frame` | `hf_client.get_frame(0)`, then `pipeline.run_dsp_pipeline(raw_frame)` (the **real** `run_dsp_pipeline`, not the local helper — this test is specifically meant to exercise production wiring end-to-end) | final cube `== EXPECTED_CUBE_SHAPE`; no NaN/Inf; `np.any(abs(cube) > 0)` (not silently all-zero); `range_axis`/`velocity_axis`/`azimuth_axis` shapes correct | Same expectations as §3.2, but against a real downloaded frame instead of synthetic data |

**Skip behavior (by design, not a bug):**
* `ImportError` on `from src.data_access.hf_client import get_frame` (i.e. `huggingface_hub` not
  installed) → `pytest.skip()` with that reason.
* Any other `Exception` while calling `get_frame(0)` (no network, `huggingface.co` unreachable,
  dataset gated/renamed, HTTP 403, etc.) → `pytest.skip()` with that exception's message.

This test **will show as `SKIPPED`, not `PASSED`, in any environment without real network
access to `huggingface.co`** (this is the case in the current CI/sandbox environment — confirmed
skip reason: `Host not in allowlist: huggingface.co`). A `SKIPPED` result here is expected and
correct; it is **not** the same as the assertions having been verified. To get a real `PASSED`
here, run the suite from a machine with `huggingface_hub` installed and outbound access to
`huggingface.co` (and a valid HF token/permissions if the dataset requires one).

---

## 4. How to extend this file

* **Adding a new DSP stage** (e.g. a CFAR detector): add its function to §2 of this doc first
  (signature/input/output/failure modes), then add a corresponding test class following the
  pattern in §3 — a shape test, a "does it do the one thing it claims to do" test with a
  hand-constructed input where you know the correct answer analytically, and (if it touches an
  FFT axis) a peak-localization test using the `fftshift(arange(N))`-lookup technique from
  §3.4/§3.5 rather than a hand-derived shift formula.
* **Changing a config value** (`numAdcSamples`, `numLoops`, antenna counts): the shape tests in
  §3.2 will automatically track a `numAdcSamples`/`numLoops` change since they're derived, not
  hard-coded. `NUM_RX`/`NUM_TX` in the test file are **not** derived from `configs.py` (that
  module doesn't expose them) — update the constants at the top of `tests/test_dsp.py` by hand
  if the real antenna configuration changes.
* **Changing `run_dsp_pipeline`'s step order/parameters**: update `_run_full_pipeline()` (§3.0)
  to match, or the hermetic shape/NaN tests will keep testing the *old* wiring while production
  runs the new one.
* **Running the suite**: `pytest tests/test_dsp.py -v` from the project root (relies on
  `pythonpath = ["."]` in `pyproject.toml`). Add `-rs` to see skip reasons
  (`pytest tests/test_dsp.py -v -rs`).
