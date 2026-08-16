import numpy as np

def apply_window(data, window_type='hann', axis=-1):
    N = data.shape[axis]
    if window_type.lower() == 'hann':
        win = np.hanning(N)
    elif window_type.lower() == 'blackman':
        win = np.blackman(N)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")
    shape = [1] * data.ndim
    shape[axis] = N
    win = win.reshape(shape)

    return data * win
