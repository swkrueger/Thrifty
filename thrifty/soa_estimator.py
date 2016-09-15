"""Detect presence and estimate sample-of-arrival of a DSSS signal."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np

from thrifty import signal
from thrifty import toads_data


def make_soa_estimator(template, thresh_coeffs, block_len, history_len):
    """Create a SoA estimator that uses the default algorithms.

    The default despreader, detector and interpolator algorithms will be used:
     - Despreader: correlate using FFT
     - Detector: Simple threshold detector.
     - Interpolator: Parabolic interpolator.

    Parameters
    ----------
    template : :class:`numpy.ndarray`
        Template signal.
    thresh_coeffs : (float, float, float) tuple
        Coefficients of threshold formula: (constant, snr, stddev).
    block_len : int
        Size of data blocks.
    history_len : int
        Number of samples at the end of each block that are repeated in the
        next block.

    Returns
    -------
    soa_estimator : callable
    """
    despread = make_despreader(template, block_len)
    template_len = len(template)
    window = calculate_window(block_len, history_len, template_len)

    def _soa_estimate(fft):
        assert len(fft) == block_len
        corr = despread(fft)
        corr_mag = np.abs(corr)
        detected, peak_idx, peak_ampl = peak_detect(
            corr_mag, thresh_coeffs, window)
        if detected:
            offset = parabolic_interpolation(corr_mag, peak_idx)
        else:
            offset = 0
        noise = np.mean(corr_mag)
        info = toads_data.CorrDetectionInfo(peak_idx, offset, peak_ampl, noise)
        return detected, info, corr

    return _soa_estimate


def make_despreader(template, block_len):
    """Correlate / despread using FFT."""
    corr_len = block_len - len(template) + 1
    template = np.concatenate([template, np.zeros(corr_len - 1)])
    template_fft = np.fft.fft(template)

    def _despread(fft):
        corr_fft = fft * template_fft.conjugate()
        corr_full = signal.compute_ifft(corr_fft)
        corr = corr_full[:corr_len]
        return corr

    return _despread


def calculate_window(block_len, history_len, template_len):
    """Calculate the interval of values that are unique in a correlation block.

    Returns a half-open interval [start, stop).

    The minimum `history_len` is `template_len - 1`, but extra values are
    required at both sides of the correlation peak in order to perform
    interpolation. It is thus necessary to increase the `history_size` of the
    data blocks to secure padding in case the correlation peak is at the edge
    of the correlation block. It is necessary to limit peak detection to the
    range of values within the correlation block that are unique to that block
    to prevent duplicate detections.
    """
    assert history_len >= template_len - 1
    corr_len = block_len - template_len + 1
    padding = history_len - template_len + 1
    left_pad = padding // 2
    right_pad = padding-left_pad
    start, stop = left_pad, corr_len-right_pad
    return start, stop


def peak_detect(corr_mag, thresh_coeffs, window):
    """Simple threshold detector to determine presence of template signal."""
    start, stop = window
    threshold = calculate_threshold(corr_mag, thresh_coeffs)
    peak_idx = np.argmax(corr_mag[start:stop]) + start
    peak_ampl = corr_mag[peak_idx]
    detected = peak_ampl > threshold

    return detected, peak_idx, peak_ampl


def calculate_threshold(corr_mag, thresh_coeffs):
    """Calculate detector threshold given the formula's coefficients."""
    thresh_const, thresh_snr, thresh_stddev = thresh_coeffs
    stddev = np.std(corr_mag) if thresh_stddev else 0
    noise = np.mean(corr_mag)
    thresh = thresh_const + thresh_snr * noise + thresh_stddev * stddev
    return thresh


def parabolic_interpolation(corr_mag, peak_idx):
    """Sub-sample SoA estimation using parabolic interpolation."""
    # pylint: disable=invalid-name
    if peak_idx == 0 or peak_idx == len(corr_mag) - 1:
        logging.warn("Parabolic interpolation failed: peak_idx out of bounds."
                     " Please ensure history_len >= template_len + 1.")
        return 0
    a, b, c = corr_mag[peak_idx-1], corr_mag[peak_idx], corr_mag[peak_idx+1]
    offset = 0.5 * (c - a) / (2 * b - a - c)
    if offset < -0.6 or offset > 0.6:
        logging.warn("Large offset using parabolic interpolation: %f.", offset)
    return offset
