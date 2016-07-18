"""
Unit tests for utility functions.
"""

import numpy as np
import pytest

from thrifty import util


@pytest.mark.parametrize("num", [15, 16])
def test_fft_bin(num):
    """Validate fft_bin against numpy's fftfreq function."""
    expected = np.fft.fftfreq(num, 1./num)
    got = np.array([util.fft_bin(i, num) for i in range(num)])
    np.testing.assert_array_equal(got, expected)
