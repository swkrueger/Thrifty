"""
Unit tests for carrier_detect module.
"""

import pytest
import numpy as np

from thrifty import carrier_detect


RANGE_INDEX_TESTDATA = [
    (50, 100, 1024, (50, 100)),
    (0, -1, 1024, (0, 1023)),
    (-10, 10, 1024, (1014, 1034)),
    (-1, 0, 1024, (1023, 1024)),
]


@pytest.mark.parametrize("start,stop,length,expected", RANGE_INDEX_TESTDATA)
def test_fft_range_index(start, stop, length, expected):
    """Test frequency-bin-to-FFT-indices conversion."""
    assert carrier_detect.fft_range_index(start, stop, length) == expected


DETECT_WINDOW_TESTDATA = [
    # freq_min, freq_max, freq, detected
    (-81.0e3, -79.0e3, -80.0e3, True),
    (-81.0e3, -79.0e3, -79.1e3, True),
    (-81.0e3, -79.0e3, -80.9e3, True),
    (-81.0e3, -79.0e3, -82.0e3, False),
    (-81.0e3, -79.0e3, -78.0e3, False),
    (-81.0e3, -79.0e3, 0.0e3, False),
    (-81.0e3, -79.0e3, 0.0e3, False),

    (79.0e3, 81.0e3, 80.0e3, True),
    (79.0e3, 81.0e3, 79.1e3, True),
    (79.0e3, 81.0e3, 80.9e3, True),
    (79.0e3, 81.0e3, 82.0e3, False),
    (79.0e3, 81.0e3, 78.0e3, False),
    (79.0e3, 81.0e3, -80.0e3, False),
    (79.0e3, 81.0e3, 0.0e3, False),

    (-10.0e3, 5.0e3, 0.0e3, True),
    (-10.0e3, 5.0e3, -9.9e3, True),
    (-10.0e3, 5.0e3, 4.9e3, True),
    (-10.0e3, 5.0e3, 6.0e3, False),
    (-10.0e3, 5.0e3, -11.0e3, False),
]


@pytest.mark.parametrize("freq_min,freq_max,carrier_freq,expected_detected",
                         DETECT_WINDOW_TESTDATA)
def test_detect_window(freq_min, freq_max, carrier_freq, expected_detected):
    """Test detect() with different windows, but without a filter."""
    # pylint: disable=too-many-locals

    block_len = 8192
    carrier_len = 2085
    sample_rate = 2.2e6
    threshold = (500.0, 0.0, 0.0)

    bin_freq = sample_rate / block_len
    bin_min, bin_max = int(freq_min / bin_freq), int(freq_max / bin_freq)

    carrier_rads = 2j * np.pi * carrier_freq
    carrier = np.exp(carrier_rads * np.arange(carrier_len) / sample_rate)
    block = np.concatenate([carrier, np.zeros(block_len - carrier_len)])
    fft_mag = np.abs(np.fft.fft(block))

    detected, _, _, _ = carrier_detect.detect(
        fft_mag, threshold, (bin_min, bin_max))
    assert detected == expected_detected


FILTER_TESTDATA = [
    ([0, 1, 2, 8, 4, 1, 0], [0.8, 1, 0.7], (3, 4.96)),
    ([0, 1, 2, 8, 4, 1, 0], [1, 0.5, 0.5], (4, 5.25)),
]


@pytest.mark.parametrize("fft_mag, peak_filter, expected", FILTER_TESTDATA)
def test_peak_filter(fft_mag, peak_filter, expected):
    """Test detect() with a peak filter."""
    _, peak_idx, peak_energy, _ = carrier_detect.detect(
        fft_mag, (0, 0, 0), None, peak_filter)
    assert (peak_idx, peak_energy) == expected
