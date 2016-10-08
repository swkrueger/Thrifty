"""Various utility methods."""

import numpy as np


def snr(peak_ampl, noise_rms):
    """Calculate SNR given amplitude and noise RMS."""
    return 20 * np.log10(np.divide(peak_ampl, noise_rms))


def fft_bin(idx, fft_len):
    """Convert index of FFT array to a signed frequency bin.

    The result of a DFT is generally stored in "standard" order: the first half
    contains the positive-frequency terms and the last half the negative-
    frequency terms. This method converts an index of an array in "standard"
    order to a signed frequency term.
    """
    if idx < 0 or idx <= (2*fft_len - 1) / 4:
        return idx
    else:
        return idx - fft_len


def find_nearest(array, values):
    """For each value in values, find the index of the nearest value in
    array."""
    indices = np.searchsorted(array, values)
    for i, idx in enumerate(indices):
        value = values[i]
        if idx > 0 and (idx == len(array) or
                        np.abs(value - array[idx-1]) <
                        np.abs(value - array[idx])):
            indices[i] = idx - 1
    return indices
