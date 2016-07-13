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

import contextlib
import numpy as np
from scipy.optimize import curve_fit

from thrifty import toads_data
from thrifty import carrier_detect


# Adapted from https://github.com/borg-project/borg/blob/master/borg/util.py
@contextlib.contextmanager
def numpy_set_errors(**kwargs):
    """Temporarily modify numpy error options."""
    old = np.seterr(**kwargs)
    try:
        yield
    except:
        raise
    finally:
        np.seterr(**old)


def sync(fft, detector, interpolator, shifter):
    """Connect carrier detector, interpolator, and frequency shifter.

    Parameters
    ----------
    fft : :class:`numpy.ndarray`
        FFT of data to be synchronized.
    detector : callable -> (bool, int, float)
        Threshold detection algorithm that returns three values:
         - detected: detection verdict.
         - peak_idx: estimated position of carrier in FFT.
         - peak_energy: estimated peak energy.
    interpolator : callable or None
        Fractional bin frequency estimation algorithm.
    shifter : callable
        Frequency-domain shift algorithm.

    Returns
    -------
    shifted_fft : :class:`numpy.ndarray` or None
    info : :class:`toads_data.CarrierSyncInfo`
    """
    fft_mag = np.abs(fft)
    detected, peak_idx, peak_energy = detector(fft_mag)
    offset = 0
    if detected:
        if interpolator is not None:
            offset = interpolator(fft_mag, peak_idx)
        shifted_fft = shifter(fft, -(peak_idx+offset))
    else:
        shifted_fft = None
    noise = np.mean(fft_mag)
    info = toads_data.CarrierSyncInfo(peak_idx, offset, peak_energy, noise)
    return shifted_fft, info


def make_syncer(thresh_coeffs, window, block_len, carrier_len,
                filter_len=0, interpol_width=6):
    """Create a carrier synchronizer method that uses the default algorithms.

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
    filter_len : int
        Size of matched filter, in samples, to apply to obtain a better
        estimate of the peak's energy.
    carrier_len : int
        The length of the carrier transmission, in number of samples.
    """
    # pylint: disable=too-many-arguments
    if filter_len > 0:
        rel = np.arange(-(filter_len//2), filter_len//2+1)
        weights = dirichlet_kernel(rel, block_len, carrier_len)
    else:
        weights = None

    detector = lambda fft_mag: carrier_detect.detect(fft_mag, thresh_coeffs,
                                                     window, weights)
    interpolator = make_dirichlet_interpolator(interpol_width,
                                               block_len, carrier_len)
    shifter = freq_shift
    return lambda fft: sync(fft, detector, interpolator, shifter)


def dirichlet_kernel(xdata, block_len, carrier_len):
    """Calculate Dirichlet weights.

    The Dirichlet kernel is the discrete time version of the sinc function.
    """
    # pylint: disable=invalid-name
    N, W = block_len, carrier_len
    xdata = np.array(xdata, dtype=np.float64)
    with numpy_set_errors(divide='ignore'):
        weights = np.sin(np.pi*W*xdata/N) / np.sin(np.pi*xdata/N) / W
        weights[np.isnan(weights)] = 1
    return weights


def make_dirichlet_interpolator(width, block_len, carrier_len):
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
        xdata = np.array(np.arange(-(width//2), width//2+1))
        ydata = fft_mag[peak_idx + xdata]
        initial_guess = (fft_mag[peak_idx], 0)
        popt, _ = curve_fit(_fit_model, xdata, ydata, p0=initial_guess)
        _, fit_offset = popt
        return fit_offset

    return _interpolator


def freq_shift(fft, shift):
    """Shift a signal in the frequency domain by a fractional number of bins.

    The shift theorem is used to shift frequency in the time-domain.
    Yes, this is slow and should be replaced with a faster algorithm.

    Parameters
    ----------
    fft : array_like
    shift : float
        Number of (potentially fractional) shift to shift the signal by.
    """
    if isinstance(shift, int) or shift.is_integer():
        shifted_fft = np.roll(fft, shift)
    else:
        freqs = np.arange(len(fft)) * 1. / len(fft) - 0.5
        shift = np.exp(2j * np.pi * shift * freqs)
        shifted_time = np.fft.ifft(fft) * shift
        shifted_fft = np.fft.fft(shifted_time)
    return shifted_fft
