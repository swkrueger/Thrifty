"""Detect presence of a carrier and synchronize to the carrier's frequency.

This module divides carrier synchronization into three steps:

 - Detector (coarse peak)
   Detect the presence of a carrier and the coarse frequency of the carrier.

 - Subsample interpolator (fine peak)
   Estimate the sub-bin frequency of the carrier.

 - Frequency shifter
   Remove the carrier by shifting the signal in the frequency domain.

The module is designed such that each of these steps can be replaced with a
different algorithm if necessary.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.optimize import curve_fit

from thrifty import toads_data
from thrifty import carrier_detect


class Synchronizer(object):
    """Connect carrier detector, interpolator, and frequency shifter together.

    Parameters
    ----------
    detector : callable -> (bool, int, float, float)
        Threshold detection algorithm that returns three values:
         - detected: detection verdict.
         - peak_idx: estimated position of carrier in FFT.
         - peak_mag: estimated peak magnitude.
         - noise_rms: estimated noise rms.
    interpolator : callable or None
        Fractional bin frequency estimation algorithm.
    shifter : callable
        Frequency-domain shift algorithm.
    """

    def __init__(self, detector, interpolator, shifter):
        self.detector = detector
        self.interpolator = interpolator
        self.shifter = shifter

    def sync(self, signal):
        """Detect presence of carrier, estimate frequency, and compensate.

        Parameters
        ----------
        signal : :class:`signal.Signal`
            Signal to be synchronized.

        Returns
        -------
        shifted_fft : :class:`numpy.ndarray` or None
        info : :class:`toads_data.CarrierSyncInfo`
        """
        fft_mag = signal.fft.mag
        detected, peak_idx, peak_mag, noise_rms = self.detector(fft_mag)
        offset = 0
        if detected:
            if self.interpolator is not None:
                offset = self.interpolator(fft_mag, peak_idx)
            shifted_fft = self.shifter(signal, -(peak_idx+offset))
        else:
            shifted_fft = None
        info = toads_data.CarrierSyncInfo(peak_idx, offset,
                                          peak_mag, noise_rms)
        return shifted_fft, info

    def __call__(self, signal):
        return self.sync(signal)


class DefaultSynchronizer(Synchronizer):
    """A carrier synchronizer that uses the default algorithms.

    The default detector, interpolator and shifter will be used:
     - Detector: Simple threshold detector, using the shape of the Dirichlet
                 kernel as matched filter.
     - Interpolator: Curve fitting of Dirichlet kernel to FFT.
     - Shifter: Use shift theorem to shift frequency in the time-domain.

    Parameters
    ----------
    thresh_coeffs : (float, float, float) tuple
        Coefficients of threshold formula: (constant, snr, stddev).
    window : (int, int) tuple
        Limit detection to the frequency bins [start, stop].
    block_len : int
        Size of data blocks.
    carrier_len : int
        The length of the carrier transmission, in number of samples.
    """

    def __init__(self, thresh_coeffs, window, block_len, carrier_len):
        self.thresh_coeffs = thresh_coeffs
        self.window = window
        filter_width = int(block_len / carrier_len) * 2
        self.weights = dirichlet_weights(filter_width, block_len, carrier_len)

        interpolator = make_dirichlet_interpolator(block_len, carrier_len)
        Synchronizer.__init__(self, self.detect, interpolator, freq_shift)

    def detect(self, fft_mag):
        """Detect the presence of a carrier in a FFT."""
        return carrier_detect.detect(fft_mag,
                                     self.thresh_coeffs,
                                     self.window,
                                     self.weights)


def dirichlet_kernel(xdata, block_len, carrier_len):
    """Calculate Dirichlet weights.

    The Dirichlet kernel is the discrete time version of the sinc function.
    """
    # pylint: disable=invalid-name
    N, W = block_len, carrier_len
    xdata = np.array(xdata, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = np.sin(np.pi*W*xdata/N) / np.sin(np.pi*xdata/N) / W
        weights[np.isnan(weights)] = 1
    return weights


def dirichlet_weights(filter_len, block_len, carrier_len):
    """Create weights for matching peak to a Dirichlet kernel.

    Parameters
    ----------
    filter_len : int
        Size of matched filter, in samples, to apply to obtain a better
        estimate of the peak's mag.
    """
    rel = np.arange(-(filter_len//2), filter_len//2+1)
    return dirichlet_kernel(rel, block_len, carrier_len)


def make_dirichlet_interpolator(block_len, carrier_len, width=6):
    """Estimate sub-bin carrier frequency by fitting a Dirichlet kernel.

    The actual carrier frequency may not fall on the center frequency of a
    FFT bin. For a better estimate of the carrier frequency, the FFT can
    be interpolated. This method fits a Dirichlet kernel function.

    Since the code is transmitted for a limited time only, the carrier can
    be decomposed into two components: a sinusoidal and a rect window function.
    The DFT of a rect window function is a sinc-line function (more
    specifically, a Dirichlet kernel) [1]_.

    Parameters
    ----------
    width : int
        Number of samples to use for curve fitting.
    block_len : int
        Size of data blocks.
    carrier_len : int
        The length of the carrier transmission, in number of samples.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Discrete_Fourier_transform
    """

    def _fit_model(xdata, amplitude, time_offset):
        xdata = np.array(xdata, dtype=np.float64)
        dirichlet = dirichlet_kernel(xdata-time_offset, block_len, carrier_len)
        return amplitude * np.abs(dirichlet)

    def _interpolator(fft_mag, peak_idx):
        """Curve fitting of Dirichlet kernel to FFT."""
        xdata = np.array(np.arange(-(width//2), width//2+1))
        ydata = fft_mag[peak_idx + xdata]
        initial_guess = (fft_mag[peak_idx], 0)
        popt, _ = curve_fit(_fit_model, xdata, ydata, p0=initial_guess)
        _, fit_offset = popt
        return fit_offset

    return _interpolator


def parabolic_interpolator(fft_mag, peak_idx):
    """Estimate sub-bin carrier frequency by fitting a parabola."""
    # pylint: disable=invalid-name
    a, b, c = fft_mag[peak_idx-1], fft_mag[peak_idx], fft_mag[peak_idx+1]
    offset = (c - a) / (4*b - 2*a - 2*c)
    return offset


def make_polyfit_interpolator(width):
    """Estimate sub-bin carrier frequency by fitting a quadratic function with
    polyfit."""

    def _interpolator(fft_mag, peak_idx):
        xdata = np.array(np.arange(-(width//2), width//2+1))
        ydata = fft_mag[peak_idx + xdata]
        coeffs = np.polyfit(xdata, ydata, 2)
        offset = -coeffs[1] / coeffs[0] / 2

        return offset

    return _interpolator


def freq_shift(signal, shift):
    """Shift a signal in the frequency domain by a fractional number of bins.

    The shift theorem is used to shift frequency in the time-domain.
    Yes, this is slow and should be replaced with a faster algorithm.

    Parameters
    ----------
    signal : signal.Signal
    shift : float
        Number of (potentially fractional) shift to shift the signal by.
    """
    freqs = np.arange(len(signal)) * 1. / len(signal) - 0.5
    shift_signal = np.exp(2j * np.pi * shift * freqs)
    shifted_time = signal * shift_signal
    shifted_fft = shifted_time.fft
    return shifted_fft


def freq_shift_integer(signal, shift):
    """Shift a signal by an integer number of samples."""
    int_shift = int(np.round(shift))
    shifted_fft = np.roll(signal.fft, int_shift)
    return shifted_fft
