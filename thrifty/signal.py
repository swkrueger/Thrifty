"""Common util function operating on digital signals."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

try:
    import pyfftw
    USE_PYFFTW = True
except ImportError:
    USE_PYFFTW = False


if USE_PYFFTW:
    pyfftw.interfaces.cache.enable()


def compute_fft(samples):
    """Compute FFT."""
    if USE_PYFFTW:
        return pyfftw.interfaces.numpy_fft.fft(samples)
    else:
        return np.fft.fft(samples)


def compute_ifft(fft):
    """Compute inverse FFT."""
    if USE_PYFFTW:
        return pyfftw.interfaces.numpy_fft.ifft(fft)
    else:
        return np.fft.ifft(fft)


def power(samples):
    """Calculate a complex signal's power."""
    return np.mean(samples * np.conj(samples)).real


class Signal(object):
    """Representation of signal in time-domain and frequency-domain, with
    on-demand conversion between the two domains."""

    def __init__(self, samples=None, fft=None):
        self._samples = samples
        self._fft = fft
        self._mag = None

        if self._samples is not None and self._fft is not None:
            assert len(self._samples) == len(self._fft)

        if samples is not None:
            self._length = len(samples)
        elif fft is not None:
            self._length = len(self._fft)
        else:
            self._length = 0

    @property
    def samples(self):
        """Get the time-domain samples of the signal."""
        if self._samples is None and self._fft is not None:
            self._samples = compute_ifft(self._fft)
        return self._samples

    @property
    def fft(self):
        """Get the frequency-domain samples of the signal."""
        if self._fft is None and self._samples is not None:
            self._fft = compute_fft(self._samples)
        return self._fft

    @property
    def mag(self):
        """Calculate the magnitude of the time-domain samples."""
        if self._mag is None:
            self._mag = np.abs(self.samples)
        return self._mag

    def __len__(self):
        return self._length

    @property
    def rms(self):
        """Calculate signal's root mean square value."""
        if self._samples is not None:
            return np.sqrt(power(self._samples))
        elif self._fft is not None:
            return np.sqrt(power(self._fft)) / len(self._fft)
