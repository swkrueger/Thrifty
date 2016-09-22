"""
Calculate the root mean square value for blocks of data.

Example:
    # rtl_sdr -f 433.81M -s 2.4M -g 55 - | noise_rms.py
"""

from __future__ import print_function

import argparse

import numpy as np

from thrifty import settings
from thrifty.block_data import block_reader


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
    rmss = []
    for _, _, block in blocks:
        rms = np.sqrt(np.sum(block * block.conj())).real
        rmss.append(rms)
        if len(rmss) == args.integrate:
            print(np.mean(rmss))
            rmss = []


if __name__ == '__main__':
    _main()
