#!/usr/bin/env python

"""
Estimates ToA from time-domain samples.
"""

import sys
import time
import matplotlib.pyplot as plt
import numpy as np

import settings
import carrier_sync
import data
import despread
import argparse
import block_reader
import carrier_fit

from utility import freq_range_parser

def peak_summarizer(settings):
    prev_corr_t = [0]

    def summarize(r, bi):
        abs_idx = settings.block_len * bi + r.peak_idx
        t = abs_idx / settings.sample_rate
        corr_dt = t - prev_corr_t[0]
        prev_corr_t[0] = t
        
        SNR = 20 * np.log10(r.peak_mag / r.noise)
        return "corr; idx: {}:{}{:+.3f} ({:.6f} s); dt={:.6f}; peak={:.0f} " \
                "thresh={:.0f}; noise={:.0f} SNR={:.2f}".format(
                bi, r.peak_idx, r.offset, t, corr_dt, r.peak_mag,
                r.threshold, r.noise, SNR)

    return summarize

def main(args, settings):
    if args.raw == True:
        blocks = block_reader.data_reader(args.input, settings)
    else:
        blocks = block_reader.serialized_block_reader(args.input, settings)

    csync = carrier_sync.carrier_syncer(settings)
    despreader = despread.despreader(settings)
    peak_detect = despread.peak_detector(settings)
    peak_summarize = peak_summarizer(settings)

    for timestamp, bi, b in blocks:
        # plt.plot(b.real)
        # plt.plot(b.imag)
        # plt.show()

        c = csync(b)
        c.data = b
        c.idx = bi

        if c.detected:
            sys.stderr.write(c.summary(settings) + '\n')

            carrier_fit.fit(c, settings)

            cr = data.CarrierSyncResult(c.peak, c.offset, np.abs(c.shifted_fft[0]), c.noise)

            if args.plot in ['always', 'carrier_detect']:
                c.plot(settings)
                plt.show()

            corr = despreader(c.shifted_fft)
            p = peak_detect(corr)

            pr = data.ToaDetectionResult(p.peak_idx, p.offset, p.peak_mag, p.noise)

            if p.detected:
                sys.stderr.write(peak_summarize(p, bi) + '\n')
                soa = settings.block_len * bi + p.peak_idx + p.offset

                det = data.DetectionResult(timestamp, bi, soa, cr, pr)
                print det.serialize()

                # print "%.06f" % timestamp, abs_idx, p.peak_mag, c.peak, np.abs(c.shifted_fft[0]), p.offset, p.noise, c.noise, c.offset

                if args.plot in ['always', 'corr_peak']:
                    plt.plot(np.abs(corr))
                    plt.show()

                # Output: carrier freq, carrier phase, carrier energy, carrier SNR, abs peak index, peak energy, peak SNR


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('input',
                        type=argparse.FileType('rb'), default='-',
                        help='input data (\'-\' streams from stdin)')
    parser.add_argument('--raw', dest='raw', action='store_true',
                        help='input data is raw binary data')

    parser.add_argument('-s', dest='sample_rate', type=float,
                        default=settings.sample_rate,
                        help='overwrite sample rate')
    parser.add_argument('-c', dest='chip_rate', type=float,
                        default=settings.chip_rate,
                        help='overwrite chip rate')

    parser.add_argument('-w', '--carrier-window', dest='carrier_window',
                        action=freq_range_parser.FreqRangeAction,
                        help='carrier detection window (e.g. "10-20 kHz")')

    parser.add_argument('-p', dest='plot',
                        choices=['always', 'carrier_detect', 'corr_peak', 'never'],
                        default='never',
                        help='when a plot should be triggered')

    args = parser.parse_args()

    # default carrier window
    if args.carrier_window == None:
        args.carrier_window = freq_range_parser.parse(settings.carrier_window)

    settings.sample_rate = args.sample_rate
    settings.chip_rate = args.chip_rate

    # normalize carrier window
    bin_freq = settings.sample_rate / settings.data_len
    settings.carrier_window_norm = freq_range_parser.normalize(
            args.carrier_window, bin_freq)

    main(args, settings)

