#!/usr/bin/env python

"""
Correlate with template and detect peak.
"""

import numpy as np
from collections import deque, namedtuple

def despreader(settings):
    code = settings.code_samples
    N = settings.data_len - len(code)
    template = np.concatenate([code, np.zeros(N)])
    template_fft = np.fft.fft(template)

    assert(settings.block_len + settings.peak_width == N)

    def despread(fft):
        F = fft * template_fft.conjugate()
        f_full = np.fft.ifft(F)
        f = f_full[:N]
        return f

    return despread


PeakDetectorResults = namedtuple(
    'PeakDetectorResults',
    'detected, peak_idx, peak_mag, threshold, noise')


def peak_detector(settings):
    means = deque([], 10)

    def detect(d):
        mag = np.abs(d)
        means.append(np.mean(mag))
        noise = np.mean(means)
        stddev = np.std(mag)

        t = settings.detector_threshold
        tc, tn, ts = t['constant'], t['snr'], t['stddev']
        threshold = tc + tn * noise + ts * stddev
        peak_idx = np.argmax(mag)
        peak_mag = mag[peak_idx]

        detected = peak_mag > threshold

        return PeakDetectorResults(
               detected, peak_idx, peak_mag, threshold, noise)

    return detect

