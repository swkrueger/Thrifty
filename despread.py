#!/usr/bin/env python

"""
Correlate with template and detect peak.
"""

import numpy as np
from collections import deque, namedtuple

def despreader(settings):
    code = settings.code_samples
    assert(settings.history_len >= len(code))
    N = settings.data_len - len(code)
    template = np.concatenate([code, np.zeros(N)])
    template_fft = np.fft.fft(template)

    # assert(settings.block_len + settings.peak_width == N)

    def despread(fft):
        F = fft * template_fft.conjugate()
        f_full = np.fft.ifft(F)
        f = f_full[:N]
        return f

    return despread


PeakDetectorResults = namedtuple(
    'PeakDetectorResults',
    'detected, peak_idx, offset, peak_mag, threshold, noise')


# def fit(peak_mag, peak_idx):
#     """Simple center-of-mass interpolation."""
#     n = 1
#     rel = np.array(np.arange(-n, n+1))
#     noms = (rel + n + 1) * peak_mag[peak_idx + rel]
#     nom = np.sum(noms)
#     den = np.sum(peak_mag[peak_idx + rel])
#     offset = nom / den - n - 1
#     return offset


def peak_detector(settings):
    means = deque([], 10)
    padding_len = settings.history_len - settings.code_len
    half_pad = int(padding_len / 2)

    def detect(d):
        mag = np.abs(d)
        means.append(np.mean(mag))
        noise = np.mean(means)
        stddev = np.std(mag)

        t = settings.detector_threshold
        tc, tn, ts = t['constant'], t['snr'], t['stddev']
        threshold = tc + tn * noise + ts * stddev
        peak_idx = np.argmax(mag[padding_len-half_pad:len(mag)-half_pad])
        peak_mag = mag[peak_idx]
        # peak_mag = np.sum(mag[peak_idx-1:peak_idx+2]) / (1+0.9+0.9)

        detected = peak_mag > threshold
        # offset = fit(mag, peak_idx)
        offset = 0

        return PeakDetectorResults(
               detected, peak_idx, offset, peak_mag, threshold, noise)

    return detect

