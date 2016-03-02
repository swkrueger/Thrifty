#!/usr/bin/env python

"""
The main CLI interface for the SDR.
TODO: better description.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np

import settings
import carrier_sync
import despread
import argparse
import block_reader

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

    args = parser.parse_args()
    settings.sample_rate = args.sample_rate
    settings.chip_rate = args.chip_rate

    blocks = block_reader.data_reader(args.input, settings)
    csync = carrier_sync.carrier_syncer(settings)
    despreader = despread.despreader(settings)
    peak_detect = despread.peak_detector(settings)
    prev_corr_t = 0

    for bi, b in enumerate(blocks):
        r = csync(b)
        r.idx = bi

        if r.detected:
            sys.stderr.write(r.summary(settings) + '\n')
            # r.plot(settings)
            # plt.show()

            corr = despreader(r.shifted_fft)
            detected, peak_idx, peak_mag, threshold, noise = peak_detect(corr)
            if detected:
                abs_idx = settings.block_len * bi + peak_idx
                t = abs_idx / settings.sample_rate
                corr_dt = t - prev_corr_t
                
                SNR = 20 * np.log10(peak_mag / noise)
                print "Corr peak @ {:.6f} s ({}:{}={}) dt={:.6f} thresh={:.0f} corr_max={:.0f} SNR={:.2f}".format(
                        t, bi, peak_idx, abs_idx, corr_dt, threshold, peak_mag, SNR)

                plt.plot(np.abs(corr))
                plt.show()

