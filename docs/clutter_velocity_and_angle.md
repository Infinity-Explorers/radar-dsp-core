# DSP Radar Processing

## 1. Task Overview

The DSP part of the radar project is responsible for processing the Range Spectrum data received from Member 2 and extracting useful target information.

The main objectives are:

1. Remove static background/clutter.
2. Apply Doppler FFT to estimate target velocity.
3. Generate a Range-Doppler Map (RDM).
4. Apply spatial FFT across the receiver antennas.
5. Estimate the target azimuth angle.
6. Generate a polar radar-style Range-Azimuth visualization.

---

## 2. Input Data from Member 2

The DSP pipeline does not generate the Range Spectrum itself.

Member 2 provides the processed Range Spectrum / range-domain data.

The original radar frame was observed with the shape:

`text
(128, 255, 4, 2)

For the DSP processing pipeline, the data is arranged as:

(receivers, chirps, range_bins)

The main input to the DSP stage is therefore the Range Spectrum matrix produced by the previous processing stage.


---

3. Radar Parameters

The radar configuration used in the project contains:

Parameter Value

Carrier frequency 77 GHz
Bandwidth 0.67 GHz
Chirp duration 60 us
ADC samples 128
Number of chirps 255
Number of RX antennas 4
Number of TX antennas 2
Frequency slope 21 MHz/us
ADC sampling rate 4000 Ksps
Frame periodicity 33.33333 ms


The DSP processing mainly uses:

Carrier frequency = 77 GHz

Chirp repetition time = 60 us

Number of chirps = 255

Number of RX antennas = 4


The receiver antenna spacing is:

d = lambda / 2

where:

lambda = c / fc


---

4. Static Clutter Removal

The first DSP operation is static clutter removal.

Stationary objects produce approximately zero-Doppler reflections. To suppress this background, the mean across the slow-time/chirp dimension is calculated and subtracted.

def remove_static_clutter(range_matrix):
    mean_clutter = np.mean(
        range_matrix,
        axis=1,
        keepdims=True
    )

    clean_matrix = range_matrix - mean_clutter

    return clean_matrix

Axis used

Axis 1 = chirps / slow-time

Therefore:

np.mean(range_matrix, axis=1)

estimates the stationary component over multiple chirps.

The result is the cleaned Range Spectrum matrix.


---
5. Doppler FFT

After clutter removal, Doppler processing is performed along the chirp axis.

doppler_spectrum = np.fft.fft(
    clean_matrix,
    axis=1
)

Then fftshift is applied:

doppler_spectrum = np.fft.fftshift(
    doppler_spectrum,
    axes=1
)

This places zero Doppler in the center of the spectrum.

The Doppler spectrum contains information about target motion.


---

6. Velocity Calculation

The Doppler frequency is converted into radial velocity using:

v = (lambda / 2) * fd

where:

v = radial velocity

lambda = radar wavelength

fd = Doppler frequency


The wavelength is calculated using:

wavelength = C / carrier_frequency

with:

C = 3 x 10^8 m/s
fc = 77 GHz

The Doppler frequency bins are generated using:

np.fft.fftfreq(
    n_chirps,
    d=chirp_repetition_time
)

and shifted using fftshift.

The result is the velocity axis in m/s.


---

7. Range-Doppler Map

The Doppler spectrum contains a receiver dimension.

The magnitude is calculated and averaged across the receivers:

magnitude = np.abs(doppler_spectrum)

rdm = np.mean(
    magnitude,
    axis=0
)

rdm = rdm.T

The resulting matrix has the form:

(range bins, velocity bins)

Therefore:

Y-axis -> Range (m)
X-axis -> Velocity (m/s)
Color  -> Signal magnitude


---

8. Conversion to dB

The signal magnitude is converted to decibels using:

20 * np.log10(data + 1e-12)

The small value 1e-12 prevents taking the logarithm of zero.

The dB representation makes differences in signal strength easier to visualize.


---

9. Range-Doppler Map Visualization

The RDM is displayed using imshow().

The plot contains:

X-axis -> Velocity (m/s)
Y-axis -> Range (m)
Color  -> Magnitude (dB)

The generated plot is titled:

Range-Doppler Map

The map is useful for visually identifying targets according to:

Range

Radial velocity

Signal strength


The RDM is therefore not just decorative; it is a useful diagnostic and visualization of the Doppler processing results.


---

10. Spatial / Angle FFT

After Doppler processing, the receiver antenna dimension is used to estimate target direction.

The radar has:

4 RX antennas

The spatial FFT is applied across the receiver axis:

angle_spectrum = np.fft.fft(
    doppler_spectrum,
    axis=0
)

Then:

angle_spectrum = np.fft.fftshift(
    angle_spectrum,
    axes=0
)

The important axis mapping is:

Axis 0 -> RX antennas
Axis 1 -> chirps / Doppler
Axis 2 -> range


---

11. Azimuth Calculation

For the receiver array, the antenna spacing is assumed to be:

d = lambda / 2

The spatial frequency bins are calculated using:

np.fft.fftfreq(
    n_receivers,
    d=1.0
)

After shifting the spatial frequency bins, they are converted to an angle.

The calculation uses:

sin(theta) = lambda * spatial_frequency / d

and:

theta = arcsin(sin(theta))

The result is the azimuth angle axis in degrees.


---

12. Polar Scope

The angle spectrum is converted into a range-angle representation.

The magnitude is averaged over the Doppler dimension:

angle_range_map = np.mean(
    magnitude,
    axis=1
)

The result is arranged as:

(range, angle)

A polar plot is then generated.

The polar visualization represents:

Radius          -> Range
Angular position -> Azimuth
Color/intensity -> Signal magnitude

The resulting plot is an Azimuth-Range Polar Scope.


---

13. Complete DSP Pipeline

The complete processing sequence is:

Member 2 Range Spectrum
          |
          v
Static Clutter Removal
          |
          v
Doppler FFT
          |
          v
Velocity Axis
          |
          v
Range-Doppler Map
          |
          v
Magnitude -> dB
          |
          +-----------------> RDM Plot
          |
          v
Spatial FFT across RX antennas
          |
          v
Azimuth Calculation
          |
          v
Range-Azimuth Polar Scope


---

14. Main DSP Function

All DSP processing stages are connected through:

process_dsp_and_plot(
    range_matrix,
    range_axis,
    chirp_repetition_time=CHIRP_TIME,
    carrier_frequency=CARRIER_FREQUENCY
)

The function performs:

1. Static clutter removal.


2. Doppler FFT.


3. Velocity calculation.


4. Range-Doppler Map creation.


5. dB conversion.


6. RDM plotting.


7. Spatial angle FFT.


8. Azimuth calculation.



The function returns:

clean_matrix
doppler_spectrum
velocity
rdm
rdm_db
angle_spectrum
azimuth


---

15. Testing

The testing notebook is:

testing/doppler_angle_testing.ipynb

The notebook was used to:

Load radar data using the existing data-access module.

Check the number of available frames.

Inspect the raw frame shape.

Prepare the Range Spectrum input.

Run the DSP processing pipeline.

Generate the Range-Doppler Map.

Generate the Azimuth-Range Polar Scope.


The data-access test successfully returned:

Number of frames: 19754

The raw frame shape was:

(128, 255, 4, 2)

The raw data was complex-valued.

The DSP input was rearranged into:

(4, 255, 128)

which corresponds to:

4   -> RX antennas
255 -> chirps
128 -> range samples/bins before the positive-frequency selection


---

16. Project Files

src/dsp_engine.py

Contains the DSP processing functions:

remove_static_clutter()

doppler_fft()

calculate_velocity_axis()

create_range_doppler_map()

magnitude_to_db()

plot_range_doppler_map()

angle_fft()

calculate_azimuth_axis()

plot_polar_scope()

process_dsp_and_plot()


testing/doppler_angle_testing.ipynb

Used to test the DSP pipeline using the available radar data and visualize the results.


---

17. Input / Output Interface

The DSP module receives:

Range Spectrum matrix
Range axis
Radar timing information
Carrier frequency

The main input is:

range_matrix

with the expected structure:

(receivers, chirps, range_bins)

The DSP module then produces:
Clean Range Spectrum
        |
        v
Doppler Spectrum
        |
        v
Velocity Axis
        |
        v
Range-Doppler Map
        |
        v
Angle Spectrum
        |
        v
Azimuth Axis


---

18. Important Separation Between Team Members

The DSP module starts from the Range Spectrum produced by the previous stage.

Therefore:

Raw ADC Data
     |
     v
Range Processing / Range FFT
     |
     |  Member 2 output
     v
Range Spectrum
     |
     |  DSP task
     v
Static Clutter Removal
     |
     v
Doppler FFT
     |
     v
Velocity
     |
     v
RDM
     |
     v
Angle FFT
     |
     v
Azimuth
     |
     v
Polar Scope

The DSP module does not replace the Range FFT stage.


---

19. Current Status

The DSP pipeline was successfully executed on the available radar data.

The following outputs were generated successfully:

Range-Doppler Map

Shows:

Range vs Velocity

with signal magnitude represented by the color scale.

Azimuth-Range Polar Scope

Shows:

Range vs Azimuth

with signal magnitude represented spatially.

The complete DSP flow therefore runs from the Range Spectrum input through:

Clutter Removal
-> Doppler FFT
-> Velocity
-> RDM
-> Angle FFT
-> Azimuth
-> Polar Scope


---

20. Conclusion

The DSP task implements the processing required after the Range FFT stage.

The core responsibilities are:

Remove stationary clutter.

Extract Doppler information.

Calculate target radial velocity.

Build and visualize the Range-Doppler Map.

Use the RX antenna array to estimate azimuth.

Generate a Range-Azimuth polar visualization.


The module is designed to receive the Range Spectrum from the previous radar-processing stage and continue the processing pipeline from that point.

