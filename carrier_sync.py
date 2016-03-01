#!/usr/bin/env python

"""
Search for carrier and perform frequency shift if found.
"""

import numpy as np
from collections import deque
import carrier_sync_results
import block_reader
import argparse
import sys
import itertools

import matplotlib.pyplot as plt # tmp

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


def carrier_syncer(settings):
    means = deque([], settings.carrier_noise_window_size)

    def carrier_sync(b):
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

    return carrier_sync


if __name__ == '__main__':
    import settings

    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('input', type=argparse.FileType('rb'),
                        default='-',
                        help='input data (\'-\' streams from stdin)')
    parser.add_argument('-p', dest='plot', action='store_true',
                        default=False,
                        help='plot results')
    parser.add_argument('-a', dest='all', action='store_true',
                        default=False,
                        help='output both detections and non-detections')
    parser.add_argument('-s', dest='sample_rate', type=float,
                        default=settings.sample_rate,
                        help='overwrite sample rate')
    parser.add_argument('-c', dest='chip_rate', type=float,
                        default=settings.chip_rate,
                        help='overwrite chip rate')

    args = parser.parse_args()
    settings.sample_rate = args.sample_rate
    settings.chip_rate = args.chip_rate
    # overwrite freq_min, max, threshold

    blocks = block_reader.data_reader(args.input, settings)
    syncer = carrier_syncer(settings)

    for i, r in enumerate(itertools.imap(syncer, blocks)):
        r.idx = i
        if not r.detected and not args.all:
            continue
        if args.plot:
            r.plot(settings)
            plt.show()

        sys.stderr.write(r.summary(settings) + '\n')

