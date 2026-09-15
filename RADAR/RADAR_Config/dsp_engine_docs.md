# DSP Engine Documentation

## 1. Overview

The DSP processing stage receives the Range Spectrum matrix and performs
the signal processing required to detect moving targets.

The main processing stages are:

1. Static clutter removal
2. Doppler FFT
3. Velocity calculation
4. Range-Doppler Map generation
5. Azimuth angle estimation using spatial FFT
6. Visualization using Range-Doppler and polar plots

The goal is to extract the target's:

- Range
- Velocity
- Azimuth angle

## 2. Static Clutter Removal

Static clutter refers to reflections from stationary objects such as walls,
buildings, or other objects that do not have radial velocity relative to the radar.

Stationary targets have approximately zero Doppler frequency, corresponding
to 0 m/s.

To reduce these unwanted stationary reflections, the mean value is calculated
across the slow-time chirp axis.

The input Range Spectrum matrix is assumed to have the following dimensions:

(rx, chirps, range)

Therefore, Axis 1 represents the slow-time chirp dimension.

The mean clutter is calculated as:

mean_clutter = mean(range_matrix, axis=1, keepdims=True)

The clutter-reduced signal is then obtained by:

clean_matrix = range_matrix - mean_clutter

This removes the component that remains approximately constant across chirps,
which helps suppress stationary reflections around zero velocity.

### Why Axis 1?

The Range Spectrum matrix is organized as:
Axis 0 → Receiver antennas
Axis 1 → Chirps (slow time)
Axis 2 → Range bins
Since static clutter is identified by its behavior across successive chirps,
the mean must be calculated along Axis 1.

## 3. Doppler Processing

After static clutter removal, a Doppler FFT is applied along the slow-time
chirp axis.

The Doppler FFT analyzes the phase changes between successive chirps.

A moving target produces a Doppler frequency shift. This frequency shift is
related to the target's radial velocity by:
fd = 2v / λ

where:
fd = Doppler frequency
v  = target radial velocity
λ  = radar wavelength

The wavelength is calculated from:
λ = c / fc

where:
c  = speed of light
fc = radar carrier frequency

### Velocity Axis

The Doppler frequency bins are generated using the FFT sampling interval,
which is the Chirp Repetition Time (CRT).

The Doppler frequency axis is converted to velocity using:
v = (λ / 2) fd

The FFT output is shifted using fftshift so that zero velocity is located
at the center of the velocity axis.

## 4. Range-Doppler Map (RDM)

The Range-Doppler Map (RDM) is a 2D representation of the target energy
over Range and Velocity.

It allows us to determine:

- Where a target is located in range.
- How fast the target is moving.
- The strength of the received target signal.

The RDM is generated from the Doppler spectrum after applying the Doppler FFT.

### RDM Data Dimensions

The input Range Spectrum is organized as:

(rx, chirps, range)

After the Doppler FFT, the dimensions remain:

(rx, doppler bins, range bins)

To combine the information from the receiver antennas, the magnitude of
the Doppler spectrum is averaged across the receiver axis:

rdm = mean(abs(doppler_spectrum), axis=0)

This produces a 2D matrix with the dimensions:

(doppler bins, range bins)

The matrix is then transposed:

rdm = rdm.T

so that the final layout is:

(rows, columns) = (range bins, velocity bins)

This arrangement is convenient for visualization because:

- Rows represent the Y-axis → Range.
- Columns represent the X-axis → Velocity.

The transpose operation does not change the signal values. It only changes
their orientation in the matrix to match the desired plot layout.

---

## 5. RDM Visualization

The RDM is displayed as a 2D heatmap using Matplotlib's imshow() function.

The plot uses:

- X-axis → Velocity in m/s.
- Y-axis → Range in meters.
- Color → Signal magnitude.

Each point in the RDM represents the signal strength at a specific
combination of range and velocity.

For example, a target located at approximately:

Range = 20 m
Velocity = 5 m/s

should appear as a strong peak near the corresponding position on the RDM.

### Why Use a Heatmap?

The heatmap provides a visual representation of the RDM and makes strong
target reflections easier to identify.

However, the heatmap itself is only a visualization.

The actual RDM is the numerical 2D matrix produced by the DSP processing,
and it can also be processed directly to detect target peaks without plotting it.

---

## 6. Magnitude Conversion to dB

The magnitude of the Doppler spectrum is converted to a logarithmic decibel
scale before visualization.

The conversion used is:

magnitude_dB = 20 * log10(magnitude)

A small value is added before the logarithm to avoid taking the logarithm
of zero.

The dB scale is useful because radar signals can contain a large dynamic
range between strong target reflections and weak signals or noise.

Using a logarithmic scale makes weaker components easier to visualize while
still showing strong reflections.

The dB conversion is mainly used for visualization and does not represent
a different physical target measurement.

---

## 7. RDM Colorbar

The colorbar on the right side of the heatmap represents the magnitude of
the signal in dB.

Different colors correspond to different signal-strength levels.

Therefore:

- Stronger signal → higher magnitude.
- Weaker signal → lower magnitude.

The colorbar allows the displayed colors to be interpreted quantitatively
in terms of signal magnitude.

---

## 8. RDM Test with Synthetic Data

Before receiving the real Range Spectrum matrix, a synthetic target was
used to verify the DSP processing.

The test target was defined as:

Target Range = 20 m
Target Velocity = 5 m/s

A Doppler phase progression was generated across the slow-time chirps to
simulate a moving target.

The signal was inserted into the corresponding range bin and a small amount
of noise was added.

The resulting RDM was then inspected to verify that the target appeared
near the expected:

Range ≈ 20 m
Velocity ≈ 5 m/s

This test is only for development and validation. The synthetic data will
be removed when the real Range Spectrum matrix is provided.