"""
Generate Gold code template by sampling the ideal code signal.

Gold codes is the only type of code that is currently supported. An integer
sampler is used. No filter (e.g. antialiasing filter) is applied.
"""

from __future__ import division
from __future__ import print_function

import argparse

import numpy as np

from thrifty import gold
from thrifty import settings


def generate(bit_length, code_index, sps):
    """Generate a Gold code template.

    Parameters
    ----------
    bit_length : int
        Gold code register length.
    code_index : int
        Index of code within the set of Gold codes of equal length.
    sps : float
        Samples per code symbol (bit).

    Returns
    -------
    template : nparray
    """
    code = gold.gold(bit_length, code_index)
    return resample(code, sps)


def resample(code, sps):
    """Sample `code` at `sps` samples per symbol using an integer sampler."""
    length = int(sps * len(code))
    indices = np.arange(length) * len(code) // length
    symbols = np.where(code, 1, -1)
    samples = np.array(symbols)[indices]
    return samples


def _main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('length', type=int, help="Gold code register length. "
                                                 "Code length will be 2^n-1.")
    parser.add_argument('index', nargs='?', type=int, default=0,
                        help="Index of code within the set of Gold codes of "
                             "equal length.")
    parser.add_argument('-o', '--output', type=argparse.FileType('wb'),
                        default='template.npy', help="Output file (.npy).")

    setting_keys = ['sample_rate', 'chip_rate']
    config, args = settings.load_args(parser, setting_keys)

    sps = config.sample_rate / config.chip_rate
    samples = generate(args.length, args.index, sps)
    np.save(args.output, samples)

    code_len = 2**args.length - 1
    print("Generated new template: {} symbols @ {:.6f} MHz "
          "= {:.3f} ms --> {} samples @ {:.6f} Msps"
          .format(code_len,
                  config.chip_rate / 1e6,
                  code_len / config.chip_rate * 1e3,
                  len(samples),
                  config.sample_rate / 1e6))


if __name__ == '__main__':
    _main()
