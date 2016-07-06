#!/usr/bin/env python

"""
Correlate with template and detect peak.
"""

import numpy as np
from collections import deque, namedtuple
import sys


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


def fit(corr, peak_idx, n):
    """Simple center-of-mass interpolation."""

    # TODO: use complex samples!

    if peak_idx < n or peak_idx >= len(corr) - n:
        print >>sys.stderr, 'Warning: padding too small for interpolation.',
        print >>sys.stderr, 'history_len should be >=', len(corr) + 2*n
        return 0

    rel = np.array(np.arange(-n, n+1))
    noms = (rel + n + 1) * corr[peak_idx + rel]
    nom = np.sum(noms)
    den = np.sum(corr[peak_idx + rel])
    offset = nom / den - n - 1
    return float(np.real(offset))


def fit_parabole(corr, peak_idx, n):
    """Fit 2nd order polynomial"""

    if peak_idx < n or peak_idx >= len(corr) - n:
        print >>sys.stderr, 'Warning: padding too small for interpolation.',
        print >>sys.stderr, 'history_len should be >=', len(corr) + 2*n
        return 0

    rel = np.arange(-n, n+1)

    coef = np.polyfit(rel, np.abs(corr[peak_idx + rel]), 2)
    der = np.polyder(coef)
    roots = np.roots(der)

    return roots[0]


def peak_detector(settings):
    means = deque([], 10)
    padding_len = settings.history_len - settings.code_len
    half_pad = int(padding_len / 2)
    left_pad = padding_len-half_pad

    def detect(d):
        mag = np.abs(d)
        means.append(np.mean(mag))
        noise = np.mean(means)
        stddev = np.std(mag)

        t = settings.detector_threshold
        tc, tn, ts = t['constant'], t['snr'], t['stddev']
        threshold = tc + tn * noise + ts * stddev

        peak_idx = np.argmax(mag[left_pad:len(mag)-half_pad]) + left_pad
        peak_mag = mag[peak_idx]
        # peak_mag = np.sum(mag[peak_idx-1:peak_idx+2]) / (1+0.9+0.9)

        detected = peak_mag > threshold

        if detected:
            offset = fit_parabole(d, peak_idx, 1)
            # offset = fit(d, peak_idx, 2)
            # offset = 0
        else:
            offset = 0

        return PeakDetectorResults(
               detected, peak_idx, offset, peak_mag, threshold, noise)

    return detect
