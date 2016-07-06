#!/usr/bin/env python

'''
Generate Gold code template.

Generate Gold code and sample as PSK signal. An integer sampler is used.
No filter (e.g. antialiasing filter) is applied.
'''

import argparse
import numpy as np
from sp.gold import gold


def generate_code(bit_length, code_index):
    return gold(bit_length)[code_index]


def resample(code, sample_rate, chip_rate):
    # Resample code
    sps = sample_rate / chip_rate
    N_code = int(sps * len(code))
    code_indices = np.arange(N_code) * len(code) / N_code
    c = np.array(code)[code_indices] * 2 - 1

    print "Code: {} symbols @ {:.6f} MHz = {:.3f} ms ; {} samples @ {:.6f} Msps".format(
            len(code), chip_rate/1e6, len(code)/chip_rate*1e3, N_code, sample_rate/1e6)

    return c


def generate_template(bit_length, code_index, sample_rate, chip_rate):
    code = generate_code(bit_length, code_index)
    return resample(code, sample_rate, chip_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-n', '--length', dest='length', type=int, default=10,
                        help='Register length -- code length will be 2^n-1')
    parser.add_argument('-i', '--index', dest='index', type=int, default=0,
                        help='Code index -- which Gold code to generate')
    parser.add_argument('-s', '--sample-rate', dest='sample_rate', type=float,
                        default=2.2e6, help='Sample rate')
    parser.add_argument('-r', '--chip-rate', dest='chip_rate', type=float,
                        default=1.08e6, help='Chip rate')
    parser.add_argument('-o', '--output', type=argparse.FileType('wb'),
                        default='template.npy', help='output file')
    args = parser.parse_args()

    samples = generate_template(args.length, args.index,
                                args.sample_rate, args.chip_rate)
    np.save(args.output, samples)

