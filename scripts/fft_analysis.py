#!/usr/bin/env python

"""
Calculate a mean FFT over multiple FFTs.

Example:
    # rtl_sdr -f 433.83M -s 2.4M -g 55 data.bin
    # python fft_analysis.py data.bin
"""

from __future__ import print_function

import argparse

import numpy as np
import matplotlib.pyplot as plt

from thrifty import settings
from thrifty.block_data import block_reader, complex_to_raw


def _main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', nargs='?',
                        type=argparse.FileType('rb'), default='-',
                        help="input data ('-' streams from stdin)")
    parser.add_argument('-i', '--integrate', type=int, default=100,
                        help="Number of blocks to integrate over")
    setting_keys = ['block_size', 'block_history']
    config, args = settings.load_args(parser, setting_keys)

    blocks = block_reader(args.input, config.block_size, config.block_history)

    fft_sum = np.zeros(config.block_size, dtype=np.float)
    hist_sum = np.zeros(256)

    fft_freqs = np.fft.fftfreq(config.block_size, 1./config.block_size)
    fft_freqs = np.fft.fftshift(fft_freqs)
    cnt = 0

    for _, _, block in blocks:
        samples = complex_to_raw(block)
        for s in samples:
            hist_sum[s] += 1

        fft = np.fft.fft(block)
        fft_mag = np.abs(fft)
        fft_sum += fft_mag
        cnt += 1

        if cnt == args.integrate:
            plt.subplot(1, 2, 1)
            plt.plot(fft_freqs, np.fft.fftshift(fft_sum / args.integrate))
            plt.subplot(1, 2, 2)
            plt.plot(hist_sum*1. / args.integrate, '.-')
            plt.xlim([0, 255])
            plt.tight_layout()
            plt.show()

            fft_sum = np.zeros(config.block_size, dtype=np.float)
            hist_sum = np.zeros(256)
            cnt = 0


if __name__ == '__main__':
    _main()
