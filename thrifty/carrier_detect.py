"""
Detect the presence of a carrier in a block of data.

Essentially checks for the frequency bin with the highest energy and tests it
against the threshold.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.signal


def fft_range_index(start, stop, length):
    """Convert a range of frequency bins to FFT indices.

    The range [start, stop] is a closed interval.

    Parameters
    ----------
    start : int
    stop : int
    length : int
        FFT length

    Returns
    -------
    start_idx : int
    stop_idx : int
        Note that `stop` may be >= length, in which case `np.take` should be
        used with the `mode=wrap` argument when taking elements from the FFT.

    Examples
    --------
    >>> fft_range_index(50, 100, 1024)
    (50, 100)
    >>> fft_range_index(0, -1, 1024)
    (0, 1023)
    >>> fft_range_index(-10, 10, 1024)
    (1014, 1034)
    >>> fft_range_index(-1, 0, 1024)
    (1023, 1024)
    """
    if abs(start) >= length or abs(stop) >= length:
        raise ValueError("Frequency window out of range: {} - {}"
                         .format(start, stop))
    if start < 0 and stop >= 0:
        start, stop = length+start, length+stop
    if start < 0:
        start = length+start
    if stop < 0:
        stop = length+stop
    return start, stop


def detect(fft_mag, thresh_coeffs, window=None, peak_filter=None):
    """Detect the presence of a carrier in a FFT.

    Parameters
    ----------
    fft_mag : :class:`numpy.ndarray`
        Magnitude of FFT.
    thresh_coeffs : (float, float, float) tuple
        Coefficients of threshold formula: (constant, snr, stddev).
    window : (int, int) tuple
        Limit detection to the frequency bins [start, stop].
    peak_filter : :class:`numpy.ndarray`
        Coefficients of FIR filter (weights) applied to window in order to
        match the shape of the peak for a better estimate of the peak's energy.
        The window is effectively correlated with the peak_filter array.
        The coefficients will be normalised to `sum(peak_filter)`.
        It is assumed that the peak is located at the largest coefficient.
        Note that `window` should compensate for the decreased window size.

    Returns
    -------
    detected : bool
    peak_idx : int
    peak_energy : float
    threshold : float
    """

    threshold = _calculate_threshold(fft_mag, thresh_coeffs)
    peak_idx, peak_energy = _window_peak(fft_mag, window, peak_filter)
    detected = (peak_energy > threshold)
    return detected, peak_idx, peak_energy, threshold


def _calculate_threshold(fft_mag, thresh_coeffs):
    thresh_const, thresh_snr, thresh_stddev = thresh_coeffs
    stddev = np.std(fft_mag) if thresh_stddev else 0
    noise = np.mean(fft_mag)
    thresh = thresh_const + thresh_snr * noise + thresh_stddev * stddev
    return thresh


def _get_window(array, window):
    if window is None:
        start, stop = 0, -1
    else:
        start, stop = window
    start_idx, stop_idx = fft_range_index(start, stop, len(array))
    selection = np.take(array, range(start_idx, stop_idx+1), mode='wrap')
    return selection, start_idx


def _window_peak(fft_mag, window, peak_filter):
    mags, start_idx = _get_window(fft_mag, window)

    if peak_filter is not None:
        denom = np.sum(peak_filter)
        mags = scipy.signal.lfilter(peak_filter[::-1], denom, mags)
        filter_delay = np.max(peak_filter)
    else:
        filter_delay = 0
    print(mags)

    max_idx = np.argmax(mags)
    peak_energy = mags[max_idx]

    peak_idx = max_idx - filter_delay
    peak_idx += start_idx
    if peak_idx > len(fft_mag):
        peak_idx -= len(fft_mag)

    return peak_idx, peak_energy
