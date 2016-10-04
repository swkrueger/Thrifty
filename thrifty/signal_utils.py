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
    if USE_PYFFTW:
        return pyfftw.interfaces.numpy_fft.fft(samples)
    else:
        return np.fft.fft(samples)


def compute_ifft(fft):
    if USE_PYFFTW:
        return pyfftw.interfaces.numpy_fft.ifft(fft)
    else:
        return np.fft.ifft(fft)


class Signal(np.ndarray):
    """Representation of signal with caching and on-demand calculations."""
    # Based on https://docs.scipy.org/doc/numpy/user/basics.subclassing.html

    def __new__(cls, input_array, fft=None, ifft=None):
        obj = np.asarray(input_array).view(cls)

        if fft is not None:
            obj._fft = fft
            fft._ifft = obj

        if ifft is not None:
            obj._ifft = ifft
            ifft._fft = obj

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._reset()

    def __array_wrap__(self, obj, context=None):
        # http://stackoverflow.com/questions/19223926/

        if obj.shape == ():
            return obj[()]  # if ufunc output is scalar, return it
        else:
            ret = np.ndarray.__array_wrap__(self, obj, context)
            if isinstance(ret, Signal):
                ret._reset()
            return ret

    def _reset(self):
        self._fft = None
        self._ifft = None
        self._rms = None
        self._mag = None
        self._power = None
        self._conj = None

    @property
    def fft(self):
        """FFT of signal."""
        if self._fft is None:
            self._fft = Signal(compute_fft(self), ifft=self)
        return self._fft

    @property
    def ifft(self):
        """IFFT of signal."""
        if self._ifft is None:
            self._ifft = Signal(compute_ifft(self), fft=self)
        return self._ifft

    @property
    def mag(self):
        """Calculate the magnitude of the samples."""
        if self._mag is None:
            if self._power is not None:
                self._mag = np.sqrt(self._power)
            else:
                self._mag = np.abs(self)  # TODO: Convert to Signal?
        return self._mag

    @property
    def power(self):
        """Calculate the power (magnitude^2) of the samples."""
        if self._power is None:
            # Alternative: self._power = (samples * np.conj(samples)).real
            # or: self._power = np.abs(self)**2
            self._power = self.mag**2
        return self._power

    @property
    def rms(self):
        """Calculate signal's root mean square value."""
        if self._rms is None:
            if self._fft is not None and self._fft._rms is not None:
                self._rms = self._fft._rms / len(self)
            elif self._ifft is not None and self._ifft._rms is not None:
                self._rms = self._ifft._rms * len(self)
            else:
                self._rms = np.sqrt(np.mean(self.power))
        return self._rms

    @property
    def conj(self):
        """Calculate and cache the complex conjugate."""
        if self._conj is None:
            self._conj = np.ndarray.conj(self)
            self._conj._conj = self
        return self._conj
