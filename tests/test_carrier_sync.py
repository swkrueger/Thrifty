"""
Unit tests for carrier_sync module.
"""

import pytest
import numpy as np

from thrifty import carrier_sync
from thrifty.signal_utils import Signal


FREQ_SHIFT_TESTDATA = [
    (128, 0, 0),
    (128, -32, 32),
    (128, 32, 16),
    (128, -10.5, 0.5),
    (128, 8.3, -8.3),
]


@pytest.mark.parametrize("size,freq,shift", FREQ_SHIFT_TESTDATA)
def test_freq_shift(size, freq, shift):
    """Test freq_shift() with sinusoidal signals."""
    signal = np.exp(2j*np.pi*np.arange(size)/size*freq)
    signal_fft = np.fft.fft(signal)

    expected = np.exp(2j*np.pi*np.arange(size)/size*(freq+shift))
    expected_fft = np.fft.fft(expected)

    got = carrier_sync.freq_shift(Signal(signal), shift)

    # import matplotlib.pyplot as plt
    # plt.plot(np.abs(signal_fft), label="Signal")
    # plt.plot(np.abs(expected_fft), label="Expected")
    # plt.plot(np.abs(got), label="Got")
    # plt.legend()
    # plt.show()

    np.testing.assert_allclose(np.abs(got), np.abs(expected_fft),
                               atol=1e-6, rtol=1e-6)


def test_dirichlet_kernel():
    """Test dirichlet_kernel with specific parameters."""
    expected = np.array([-0.1711, 0.0164, 0.3164, 0.6468, 0.9034, 1.,
                         0.9034, 0.6468, 0.3164, 0.0164, -0.1711])
    got = carrier_sync.dirichlet_kernel(np.arange(-5, 6), 8192, 2015)
    np.testing.assert_allclose(got, expected, rtol=2e-3)


INTERPOLATOR_OFFSETS = [-0.51, -0.5, -0.25, -0.1263, -0.1, 0.,
                        0.001, 0.2, 0.4995, 0.56]


@pytest.mark.parametrize("offset", INTERPOLATOR_OFFSETS)
def test_dirichlet_interpolator(offset):
    """Test Dirichlet interpolator with signal with different freq shifts."""
    peak_idx, width, block_len, carrier_len = 10, 6, 8192, 2024
    freq = (1.*offset+peak_idx)*carrier_len/block_len
    carrier = np.exp(2j*np.pi*np.arange(carrier_len)/carrier_len*freq)
    signal = np.concatenate([carrier, np.zeros(block_len-carrier_len)])
    signal_fft = np.abs(np.fft.fft(signal))
    interpolator = carrier_sync.make_dirichlet_interpolator(
        width, block_len, carrier_len)
    got = interpolator(signal_fft, peak_idx)
    np.testing.assert_allclose(got, offset, atol=1e-8, rtol=1e-8)
