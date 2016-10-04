"""Fine-tune the estimated chip rate of a positioning signal."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

import numpy as np
import scipy

from thrifty.carrier_sync import make_syncer
from thrifty.soa_estimator import SoaEstimator
from thrifty.block_data import card_reader
from thrifty.setting_parsers import metric_float
from thrifty import template_generate


def search(fft, initial_chip_rate, bit_length, code_index, sample_rate):
    """Find chip rate that yields the maximum correlation peak when being
    correlated with the ideal Gold code template."""

    def _objective(params):
        chip_rate = params[0]

        sps = sample_rate / chip_rate
        detected, corr_info = _match(fft, sps, bit_length, code_index)

        if not detected:
            ampl = 0
        else:
            ampl = -corr_info.energy

        print(".. try chip rate {} -> {}".format(chip_rate, -ampl))
        return ampl

    res = scipy.optimize.minimize(_objective, initial_chip_rate,
                                  method='Nelder-Mead',
                                  options={'xtol': 100, 'disp': True})

    return res.x[0]


def _match(fft, sps, bit_length, code_index):
    template = template_generate.generate(
        bit_length, code_index, sps)

    block_history = len(template) - 1
    soa_estimate = SoaEstimator(template=template,
                                thresh_coeffs=(0, 0, 0),
                                block_len=len(fft),
                                history_len=block_history)

    detected, corr_info, _ = soa_estimate(fft)
    return detected, corr_info


def _find_block(blocks, block_id):
    for _, block_idx, block in blocks:
        if block_idx == block_id:
            return block
    raise Exception("Could not find block with index {}".format(block_id))


def _plot(fft, chip_rate, bit_length, code_index, sample_rate):
    import matplotlib.pyplot as plt

    sps = sample_rate / chip_rate
    template = template_generate.generate(bit_length, code_index, sps)
    detected, corr_info = _match(fft, sps, bit_length, code_index)
    assert detected

    signal = np.fft.ifft(fft)
    start = corr_info.sample
    plt.plot(np.abs(signal[start:start+len(template)]))
    scaled_template = (template + 1) / 2 * np.max(np.abs(signal)) * 0.9
    plt.plot(np.arange(len(template))-corr_info.offset, scaled_template)
    plt.savefig('chip_rate_search.pdf', format='pdf')
    plt.show()


def _main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('card_file',
                        type=argparse.FileType('rb'), default='-',
                        help="card file with positioning signal to match "
                             "against.")
    parser.add_argument('block_id', type=int,
                        help="Block within the card file that contains the"
                             "positioning signal to match against.")
    parser.add_argument('sample_rate', type=metric_float,
                        help="Sample rate that was used to capture the data.")
    parser.add_argument('chip_rate', type=metric_float,
                        help="Estimated chip rate.")
    parser.add_argument('bit_length', type=int,
                        help="Register length of gold code to generate and "
                             "match.")
    parser.add_argument('code_index', nargs='?', type=int, default=0,
                        help="Index of code within the set of Gold codes of "
                             "equal length.")
    parser.add_argument('-p', '--plot', action="store_true",
                        help="Plot best fit.")
    args = parser.parse_args()

    blocks = card_reader(args.card_file)
    block = _find_block(blocks, args.block_id)

    # Synchronize to carrier. It is assumed that a strong carrier is present.
    sps = args.sample_rate / args.chip_rate
    carrier_len = int((2**args.bit_length - 1) * sps)
    sync = make_syncer(thresh_coeffs=(100, 0, 0),
                       window=None,
                       block_len=len(block),
                       carrier_len=carrier_len)
    fft = np.fft.fft(block)
    shifted_fft, _ = sync(fft)
    assert shifted_fft is not None

    best = search(fft=shifted_fft,
                  initial_chip_rate=args.chip_rate,
                  bit_length=args.bit_length,
                  code_index=args.code_index,
                  sample_rate=args.sample_rate)

    print("Best chip rate: {}".format(best))

    if args.plot:
        _plot(shifted_fft, best, args.bit_length,
              args.code_index, args.sample_rate)


if __name__ == '__main__':
    _main()
