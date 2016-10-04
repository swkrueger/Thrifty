"""Detect presence and estimate sample-of-arrival of a DSSS signal."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np

from thrifty import toads_data
from thrifty.signal_utils import Signal


def _clip_offset(offset):
    return -0.6 if offset < -0.6 else 0.6 if offset > 0.6 else offset


def calculate_window(block_len, history_len, template_len):
    """Calculate interval of values that are unique in a correlation block.

    Returns a half-open interval [start, stop).

    The minimum `history_len` is `template_len - 1`, but extra values are
    required at both sides of the correlation peak in order to perform
    interpolation. It is thus necessary to increase the `history_size` of
    the data blocks to secure padding in case the correlation peak is at
    the edge of the correlation block. It is necessary to limit peak
    detection to the range of values within the correlation block that are
    unique to that block to prevent duplicate detections.
    """
    assert history_len >= template_len - 1
    corr_len = block_len - template_len + 1
    padding = history_len - template_len + 1
    left_pad = padding // 2
    right_pad = padding-left_pad
    start, stop = left_pad, corr_len-right_pad
    return start, stop


class SoaEstimator(object):
    """A SoA estimator that uses the default algorithms.

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
    """

    def __init__(self, template, thresh_coeffs, block_len, history_len):
        self.template = Signal(template)
        self.template_energy = np.sum(self.template.power)

        template_len = len(template)
        self.corr_len = block_len - template_len + 1
        self.template_padded = np.concatenate([self.template,
                                               np.zeros(self.corr_len-1)])
        self.template_padded = Signal(self.template_padded)
        self.template_fft = self.template_padded.fft

        self.interpolate = gaussian_interpolation
        self.window = calculate_window(block_len, history_len, template_len)
        self.thresh_coeffs = thresh_coeffs

    def soa_estimate(self, fft):
        """Estimate the SoA of the given signal."""
        # assert len(fft) == block_len
        corr = self.despread(fft)
        peak_idx, peak_mag = self.get_peak(corr)
        noise_rms = self.estimate_noise(peak_mag, fft)
        threshold = self.calculate_threshold(corr, noise_rms)
        detected = peak_mag > threshold

        # detected, peak_idx, peak_ampl, noise_rms = self.peak_detect(corr_mag)
        offset = 0 if not detected else self.interpolate(corr, peak_idx)
        offset = _clip_offset(offset)
        info = toads_data.CorrDetectionInfo(peak_idx, offset,
                                            peak_mag, noise_rms)
        return detected, info, corr

    def __call__(self, fft):
        return self.soa_estimate(fft)

    def despread(self, fft):
        """Correlate / despread using FFT."""
        corr_fft = fft * self.template_fft.conj
        corr_full = corr_fft.ifft
        corr = corr_full[:self.corr_len]
        return corr

    def get_peak(self, corr):
        """Calculate peak index and estimate sqrt(power) of peak."""
        corr_mag = corr.mag
        start, stop = self.window
        peak_idx = np.argmax(corr_mag[start:stop]) + start
        peak_mag = corr_mag[peak_idx]
        return peak_idx, peak_mag

    def estimate_noise(self, peak_mag, fft):
        """Estimate noise from signal's rms / power."""
        # Can be sped up by using RMS value of signal before carrier recovery.
        signal_energy = fft.rms**2
        # alternative: signal_energy = np.sum(np.abs(np.fft.ifft(fft))**2)
        signal_corr_energy = signal_energy * self.template_energy

        # Subtract twice the peak power to compensate for both the correlation
        # peak's energy and the energy of the unmodulated carrier.
        peak_power = peak_mag**2
        noise_power = (signal_corr_energy - 2*peak_power) / len(fft)
        noise_rms = np.sqrt(noise_power)
        return noise_rms

    def calculate_threshold(self, corr, noise_rms):
        """Calculate detector threshold given the formula's coefficients."""
        thresh_const, thresh_snr, thresh_stddev = self.thresh_coeffs
        stddev = np.std(corr.mag) if thresh_stddev else 0
        thresh = (thresh_const +
                  thresh_snr * noise_rms**2 +
                  thresh_stddev * stddev**2)
        return np.sqrt(thresh)


def parabolic_interpolation(corr, peak_idx):
    """Sub-sample SoA estimation using parabolic interpolation."""
    # pylint: disable=invalid-name
    if peak_idx == 0 or peak_idx == len(corr) - 1:
        logging.warn("Parabolic interpolation failed: peak_idx out of bounds."
                     " Please ensure history_len >= template_len + 1.")
        return 0

    a, b, c = corr.mag[peak_idx-1], corr.mag[peak_idx], corr.mag[peak_idx+1]
    offset = 0.5 * (c - a) / (2 * b - a - c)
    return offset


def gaussian_interpolation(corr, peak_idx):
    """Sub-sample SoA estimation using Gaussian interpolation."""
    # pylint: disable=invalid-name
    if peak_idx == 0 or peak_idx == len(corr) - 1:
        logging.warn("Gaussian interpolation failed: peak_idx out of bounds."
                     " Please ensure history_len >= template_len + 1.")
        return 0

    a, b, c = corr.mag[peak_idx-1], corr.mag[peak_idx], corr.mag[peak_idx+1]
    a, b, c = np.log(a), np.log(b), np.log(c)
    offset = 0.5 * (c - a) / (2 * b - a - c)
    return offset
