#!/usr/bin/env python

"""
Estimate position from TDOA values.
"""

from __future__ import division
from __future__ import print_function

import numpy as np

from thrifty import tdoa_est

SPEED_OF_LIGHT = tdoa_est.SPEED_OF_LIGHT


def solve_1d(tdoa_array, rx_pos):
    """Simple 1D position estimator for 2xRX"""
    assert len(rx_pos) == 2
    assert len(rx_pos[0]) == 1

    tdoa_pos = tdoa_array['tdoa'] * SPEED_OF_LIGHT
    rx_dist = rx_pos[0] + rx_pos[1]
    if rx_pos[0] > rx_pos[1]:
        pos = (rx_dist - tdoa_pos) / 2
    else:
        pos = (rx_dist + tdoa_pos) / 2
    return pos


def solve_analytically(tdoa_array, rx_pos):
    # TODO
    pass


def solve_numerically(tdoa_array, rx_pos):
    # TODO
    # use analytic solution as initial value
    pass


def solve(tdoa_array, rx_pos):
    # TODO: estimate estimate confidence with SNR and DOP and return with pos
    num_rx = len(rx_pos)
    dimensions = len(rx_pos[0])

    if num_rx == 2 and dimensions == 1:
        return solve_1d(tdoa_array, rx_pos)
    else:
        raise "Not implemented yet"


def load_pos_array(fname):
    # TODO: variable number of dimensions
    # TODO: additional info like SNR and DOP
    return np.loadtxt(fname, dtype={'names': ('timestamp', 'tx', 'x'),
                                    'formats': ('f8', 'i4', 'f8')})


def _main():
    import argparse

    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)

    parser.add_argument('tdoa', nargs='?',
                        type=argparse.FileType('r'), default='data.tdoa',
                        help="tdoa data (\"-\" streams from stdin)")
    parser.add_argument('-o', '--output', dest='output',
                        type=argparse.FileType('w'), default='data.pos',
                        help="output file (\'-\' for stdout)")
    parser.add_argument('-r', '--rx-coordinates', dest='rx_pos',
                        type=argparse.FileType('r'), default='pos-rx.cfg',
                        help="path to config file that contains the "
                             "coordinates of the receivers")
    args = parser.parse_args()

    tdoa_array = tdoa_est.load_tdoa_array(args.tdoa)
    rx_pos = tdoa_est.load_pos_config(args.rx_pos)
    positions = solve(tdoa_array, rx_pos)

    for idx, position in enumerate(positions):
        timestamp = tdoa_array['timestamp'][idx]
        txid = tdoa_array['tx'][idx]
        print("{:.6f} {} {}".format(timestamp, txid, position),
              file=args.output)


if __name__ == '__main__':
    _main()
