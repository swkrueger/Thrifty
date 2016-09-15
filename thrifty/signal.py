"""Common util function operating on digital signals."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pyfftw

pyfftw.interfaces.cache.enable()


def compute_fft(samples):
    """Computer FFT."""
    return pyfftw.interfaces.numpy_fft.fft(samples)
    # return np.fft.fft(samples)


def compute_ifft(fft):
    """Compute inverse FFT."""
    return pyfftw.interfaces.numpy_fft.ifft(fft)
    # return np.fft.ifft(fft)


class Signal(object):
    """Representation of signal in time-domain and frequency-domain, with
    on-demand conversion between the two domains."""

    def __init__(self, samples=None, fft=None):
        self._samples = samples
        self._fft = fft

        if self._samples is not None and self._fft is not None:
            assert len(self._samples) == len(self._fft)

        if samples is not None:
            self._length = len(samples)
        elif fft is not None:
            self._length = len(samples)
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

    def __len__(self):
        return self._length
