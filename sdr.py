#!/usr/bin/env python

"""
The main CLI interface for the SDR.
TODO: better description.
"""

import sys
import time
import matplotlib.pyplot as plt
import numpy as np

import settings
import carrier_sync
import despread
import argparse
import block_reader

def peak_summarizer(settings):
    prev_corr_t = [0]

    def summarize(r, bi):
        abs_idx = settings.block_len * bi + r.peak_idx
        t = abs_idx / settings.sample_rate
        corr_dt = t - prev_corr_t[0]
        prev_corr_t[0] = t
        
        SNR = 20 * np.log10(r.peak_mag / r.noise)
        return "corr; idx: {}:{} ({:.6f} s); dt={:.6f}; peak={:.0f} " \
                "thresh={:.0f}; noise={:.0f} SNR={:.2f}".format(
                bi, r.peak_idx, t, corr_dt, r.peak_mag,
                r.threshold, r.noise, SNR)

    return summarize

def main(args, settings):
    blocks = block_reader.data_reader(args.input, settings)
    csync = carrier_sync.carrier_syncer(settings)
    despreader = despread.despreader(settings)
    peak_detect = despread.peak_detector(settings)
    peak_summarize = peak_summarizer(settings)
    epoch = time.time()

    for bi, b in enumerate(blocks):
        c = csync(b)
        c.idx = bi

        if c.detected:
            sys.stderr.write(c.summary(settings) + '\n')
            # c.plot(settings)
            # plt.show()

            corr = despreader(c.shifted_fft)
            p = peak_detect(corr)

            if p.detected:
                sys.stderr.write(peak_summarize(p, bi) + '\n')
                abs_idx = settings.block_len * bi + p.peak_idx
                # print epoch, abs_idx, p.peak_mag, c.peak, np.abs(c.shifted_fft[0])

                # plt.plot(np.abs(corr))
                # plt.show()

                # Output: carrier freq, carrier phase, carrier energy, carrier SNR, abs peak index, peak energy, peak SNR


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('input', type=argparse.FileType('rb'),
                        default='-',
                        help='input data (\'-\' streams from stdin)')
    parser.add_argument('-s', dest='sample_rate', type=float,
                        default=settings.sample_rate,
                        help='overwrite sample rate')
    parser.add_argument('-c', dest='chip_rate', type=float,
                        default=settings.chip_rate,
                        help='overwrite chip rate')
    parser.add_argument('-p', dest='plot',
                        choices=['always', 'carrier_detect', 'corr_peak', 'never'],
                        default='never',
                        help='when a plot should be triggered')
    # todo: overwrite freq window

    args = parser.parse_args()
    settings.sample_rate = args.sample_rate
    settings.chip_rate = args.chip_rate

    main(args, settings)

