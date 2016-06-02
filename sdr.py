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
import carrier_fit

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
    if args.input_format == 'raw':
        blocks = block_reader.data_reader(args.input, settings)
    elif args.input_format == 'serialized_blocks':
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

            if args.plot in ['always', 'carrier_detect']:
                c.plot(settings)
                plt.show()

            if args.carrier_detect_output != None:
                s = block_reader.serialize_block(b)
                print >>args.carrier_detect_output, bi, s

            corr = despreader(c.shifted_fft)
            p = peak_detect(corr)

            if p.detected:
                sys.stderr.write(peak_summarize(p, bi) + '\n')
                abs_idx = settings.block_len * bi + p.peak_idx

                # TODO: print time.time() of carrier detection
                print "%.06f" % timestamp, abs_idx, p.peak_mag, c.peak, np.abs(c.shifted_fft[0]), p.offset, p.noise, c.noise, c.offset

                if args.plot in ['always', 'corr_peak']:
                    plt.plot(np.abs(corr))
                    plt.show()

                # Output: carrier freq, carrier phase, carrier energy, carrier SNR, abs peak index, peak energy, peak SNR


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('input', type=argparse.FileType('rb'),
                        default='-',
                        help='input data (\'-\' streams from stdin)')
    parser.add_argument('--input_format', dest='input_format',
                        choices=['raw', 'serialized_blocks'],
                        default='raw',
                        help='format of the input data')
    parser.add_argument('-s', dest='sample_rate', type=float,
                        default=settings.sample_rate,
                        help='overwrite sample rate')
    parser.add_argument('-c', dest='chip_rate', type=float,
                        default=settings.chip_rate,
                        help='overwrite chip rate')
    parser.add_argument('--carrier_freq_min', dest='carrier_freq_min',
                        type=float, default=settings.carrier_freq_min,
                        help='overwrite minimum carrier frequency')
    parser.add_argument('--carrier_freq_max', dest='carrier_freq_max',
                        type=float, default=settings.carrier_freq_max,
                        help='overwrite maximum carrier frequency')
    parser.add_argument('-p', dest='plot',
                        choices=['always', 'carrier_detect', 'corr_peak', 'never'],
                        default='never',
                        help='when a plot should be triggered')
    parser.add_argument('--export_on_carrier_detect', dest='carrier_detect_output',
                        type=argparse.FileType('w'),
                        default=None,
                        help='(temporary command) write serialized blocks on carrier detect')

    args = parser.parse_args()
    settings.sample_rate = args.sample_rate
    settings.chip_rate = args.chip_rate
    settings.carrier_freq_min = args.carrier_freq_min
    settings.carrier_freq_max = args.carrier_freq_max

    main(args, settings)

