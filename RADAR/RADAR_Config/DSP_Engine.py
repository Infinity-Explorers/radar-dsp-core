import numpy as np
import matplotlib.pyplot as plt
def remove_static_clutter(range_matrix):
    mean_clutter = np.mean(range_matrix,axis = 1,keepdims = True)
    clean_matrix = range_matrix - mean_clutter 
    return clean_matrix

'''
range_matrix = np.array ([
 [[10,20,30],[12,20,35],[11,20,40],[1,20,45]   ]
 ,[[5,15,25],[5,18,30],[5,21,35],[5,24,40]] 
    ])
print(range_matrix.shape)
clean_matrix =remove_static_clutter(range_matrix)
print(clean_matrix)
'''
def doppler_fft(clean_matrix):
    doppler_spectrum = np.fft.fft(clean_matrix,axis = 1)
    doppler_spectrum = np.fft.fftshift(doppler_spectrum,axes = 1)
    return(doppler_spectrum)

def calculate_velocity_axis(n_chirp,chirp_repetition_time,carrier_frequency):
    c = 3e8
    wavelength = c/carrier_frequency
    doppler_frequency = np.fft.fftfreq(n_chirp,d = chirp_repetition_time)
    doppler_frequency = np.fft.fftshift(doppler_frequency)
    velocity = (wavelength/2)*doppler_frequency
    return(velocity)

def create_range_doppler_map(doppler_spectrum):
    magnitude = np.abs(doppler_spectrum)
    rdm =np.mean(magnitude,axis = 0)
    rdm = rdm.T
    return(rdm)

def magnitude_to_db(data):
    return 20*np.log10(data + 1e-12)

def plot_range_doppler_map(rdm_db,velocity,range_axis):
    plt.figure(figsize = (10,6))
    plt.imshow(rdm_db,aspect = "auto",extent = [velocity[0],velocity[-1],range_axis[-1],range_axis[0]])
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Range (m)")
    plt.title("Range-Doppler Map")
    plt.colorbar(label = "Magnitude (dB)")
    plt.show()

'''
# =========================
# TESTING ONLY
# Uses synthetic data to verify
# the DSP processing and RDM plot.
# =========================

if __name__ == "__main__":
    print("test started")

    # Test radar parameters
    fc = 77e9
    chirp_repetition_time = 0.0001

    n_rx = 4
    n_chirps = 128
    n_range = 256



     # =========================
     # Synthetic Target Test
     # =========================

    target_range = 20.0       # meters
    target_velocity = 5.0     # m/s
    range_resolution = 0.2    # meters

    # Create empty complex radar data
    range_matrix = np.zeros((n_rx, n_chirps, n_range),
    dtype=complex
)

    # Target range bin
    target_range_bin = int(target_range / range_resolution)

    # Radar wavelength
    c = 3e8
    wavelength = c / fc

    # Doppler frequency produced by the target
    doppler_frequency = 2 * target_velocity / wavelength

    # Slow-time samples
    chirp_index = np.arange(n_chirps)

    # Target phase change from chirp to chirp
    target_signal = np.exp(1j * 2 * np.pi * doppler_frequency
    * chirp_index * chirp_repetition_time
)

   # Put the target at the chosen range bin
    for rx in range(n_rx):
        range_matrix[rx, :, target_range_bin] = 10 * target_signal

    # Add a small amount of noise
    noise = (np.random.randn(n_rx, n_chirps, n_range)
    + 1j * np.random.randn(n_rx, n_chirps, n_range)
) * 0.1

    range_matrix += noise
    

    # 1. Static clutter removal
    clean_matrix = remove_static_clutter(range_matrix)

    # 2. Doppler FFT
    doppler_spectrum = doppler_fft(clean_matrix)

    # 3. Velocity axis
    velocity = calculate_velocity_axis(
        n_chirps,
        chirp_repetition_time,
        fc
    )

    # 4. Range-Doppler Map
    rdm = create_range_doppler_map(doppler_spectrum)

    # 5. Convert to dB
    rdm_db = magnitude_to_db(rdm)

    # Test range axis
    range_resolution = 0.2
    range_axis = np.arange(n_range) * range_resolution

    # 6. Plot RDM
    plot_range_doppler_map(
        rdm_db,
        velocity,
        range_axis
    )
'''






    
    

