#!/usr/bin/env python

"""
Search for carrier and perform frequency shift if found.
"""

import numpy as np
from collections import deque
import carrier_sync_results


def find_peak(fft, settings):
    """Return index of peak within window [min_idx,max_idx)."""
    normalise = len(fft) / settings.sample_rate
    min_idx = int(settings.carrier_freq_min * normalise)
    max_idx = int(settings.carrier_freq_max * normalise)

    if min_idx > max_idx:
        min_idx, max_idx = max_idx, min_idx

    if min_idx == max_idx:
        return min_idx

    if min_idx < 0 and max_idx >= 0:
        m1 = np.argmax(fft[min_idx:]) + min_idx
        m2 = np.argmax(fft[:max_idx + 1])
        return m1 if fft[m1] > fft[m2] else m2
    else:
        return np.argmax(fft[min_idx:max_idx + 1]) + min_idx


def freq_shift(fft, peak):
    shifted_fft = np.roll(fft, -peak)
    return shifted_fft


def carrier_sync(b, means, settings):
    r = carrier_sync_results.CarrierSyncResults()
    r.fft = np.fft.fft(b)
    fft_mag = np.abs(r.fft)

    means.append(np.mean(fft_mag))
    r.noise = np.mean(means)

    r.threshold = settings.carrier_threshold_constant + \
                  settings.carrier_threshold_snr * r.noise

    r.peak = find_peak(fft_mag, settings)

    peak_mag = np.abs(r.fft[r.peak])
    if peak_mag > r.threshold:
        r.detected = True
        r.shifted_fft = freq_shift(r.fft, r.peak)
        assert(r.shifted_fft[0] >= r.fft[r.peak])

    return r


def carrier_sync_iter(blocks, settings):
    means = deque([], settings.carrier_noise_window_size)

    for bi, b in enumerate(blocks):
        r = carrier_sync(b, means, settings)
        r.idx = bi
        yield r


if __name__ == '__main__':
    pass

