"""
Unit tests for soa_estimator module.
"""

import pytest
import numpy as np
import scipy.signal

from thrifty import soa_estimator


# 5-bit (length 31) Gold code
TEMPLATE = np.array([1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1,
                     -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1])

BLOCK_LEN = 64
OOK_SIGNAL = (TEMPLATE + 1) / 2
PEAK_MAG = 16
MAX_SIDEBAND = 5  # max Gold xcorr = 2**(n+2)/2


def gen_block(pos):
    """Generate block with an OOK signal at the given position."""
    block = np.zeros(BLOCK_LEN)
    end = min(len(block), pos+len(OOK_SIGNAL))
    block[pos:end] += OOK_SIGNAL[:end-pos]
    return block


DESPREADER_TESTDATA = [0, 1, 10, BLOCK_LEN-len(TEMPLATE),
                       BLOCK_LEN-len(TEMPLATE)+1, BLOCK_LEN-1]


@pytest.mark.parametrize("pos", DESPREADER_TESTDATA)
def test_despreader_peaks(pos):
    """Test correlation peaks with signal at different positions."""
    despread = soa_estimator.make_despreader(TEMPLATE, BLOCK_LEN)
    block = gen_block(pos)
    fft = np.fft.fft(block)
    corr = despread(fft)
    corr_abs = np.abs(corr)

    assert len(corr) == BLOCK_LEN - len(TEMPLATE) + 1

    if pos <= BLOCK_LEN - len(TEMPLATE):
        peak_idx = np.argmax(corr_abs)
        non_peak = np.delete(corr_abs, peak_idx)
        assert peak_idx == pos
        assert corr_abs[peak_idx] >= PEAK_MAG - 0.1
        np.testing.assert_array_less(non_peak, MAX_SIDEBAND + 0.1)
    else:
        np.testing.assert_array_less(corr_abs, MAX_SIDEBAND + 0.1)


@pytest.mark.parametrize("pos", DESPREADER_TESTDATA)
def test_despreader_cross_validate(pos):
    """Cross-validate despreader against scipy's correlate function."""
    despread = soa_estimator.make_despreader(TEMPLATE, BLOCK_LEN)
    block = gen_block(pos)
    fft = np.fft.fft(block)
    corr = despread(fft)
    corr2 = scipy.signal.correlate(block, TEMPLATE, mode='valid')
    np.testing.assert_allclose(corr, corr2, atol=1e-12, rtol=1e-12)


WINDOW_TESTDATA = [
    # (block_len, history_len, template_len), (expected_start, expected_stop)
    ((64, 31, 32), (0, 33)),
    ((64, 32, 32), (0, 32)),
    ((64, 33, 32), (1, 32)),
    ((64, 63, 32), (16, 17)),
]


@pytest.mark.parametrize("params,expected", WINDOW_TESTDATA)
def test_calculate_window(params, expected):
    """Test calculate_window with different parameters."""
    got = soa_estimator.calculate_window(*params)
    assert got == expected


DETECT_TESTDATA = [
    # idx, block_len, window, expected
    (0, 33, (0, 33), True),  # No padding, left edge
    (32, 33, (0, 33), True),  # No padding, right edge
    (1, 33, (1, 32), True),  # With padding, right edge
    (31, 33, (1, 32), True),  # With padding, right edge
    (0, 33, (1, 32), False),  # With padding, outside right edge
    (32, 33, (1, 32), False),  # With padding, outside right edge
]


@pytest.mark.parametrize("idx,corr_len,window,expected", DETECT_TESTDATA)
def test_detect(idx, corr_len, window, expected):
    """Test peak detection with a constant threshold."""
    corr_mag = np.zeros(corr_len)
    corr_mag[idx] = 100
    thresh_coeffs = (99, 0, 0)

    detected, peak_idx, peak_ampl, _ = soa_estimator.peak_detect(
        corr_mag, thresh_coeffs, window, None)

    assert detected == expected
    if expected:
        assert peak_idx == idx
        assert peak_ampl == 100
