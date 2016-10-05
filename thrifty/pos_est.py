#!/usr/bin/env python

"""
Estimate position from TDOA values.
"""

from __future__ import division
from __future__ import print_function

import numpy as np

from thrifty import tdoa_est

SPEED_OF_LIGHT = tdoa_est.SPEED_OF_LIGHT

POSITION_INFO_DTYPE = {
    'names': ('group_id', 'timestamp', 'tx', 'dop', 'snr', 'x', 'y', 'z'),
    'formats': ('i4', 'f8', 'i4', 'f8', 'f8', 'f8', 'f8', 'f8')
}


def solve_1d(tdoa_array, rx_pos):
    """Simple 1D position estimator for 2xRX."""
    assert len(rx_pos) == 2
    assert len(rx_pos[0]) == 1
    assert len(tdoa_array) == 1

    tdoa_pos = tdoa_array['tdoa'][0] * SPEED_OF_LIGHT
    rx_dist = rx_pos[0] + rx_pos[1]
    if rx_pos[0] > rx_pos[1]:
        position = (rx_dist - tdoa_pos) / 2
    else:
        position = (rx_dist + tdoa_pos) / 2

    return (position,), tdoa_array['snr'][0]


def solve_analytically(tdoa_array, rx_pos):
    # TODO
    pass


def solve_numerically(tdoa_array, rx_pos):
    # TODO
    # use analytic solution as initial value
    pass


def solve(tdoa_groups, rx_pos):
    # TODO: estimate estimate confidence with SNR and DOP and return with pos
    num_rx = len(rx_pos)
    dimensions = len(rx_pos[0])

    results = []
    for group_id, timestamp, tx, tdoas in tdoa_groups:
        if num_rx == 2 and dimensions == 1:
            coords, snr = solve_1d(tdoas, rx_pos)
        else:
            raise "Not implemented yet"
        dop = 0
        results.append((group_id, timestamp, tx, dop, snr) + coords)

    dtype = {
        'names': POSITION_INFO_DTYPE['names'][:5 + dimensions],
        'formats': POSITION_INFO_DTYPE['formats'][:5 + dimensions]
    }
    results = np.array(results, dtype=dtype)
    return results


def save_positions(output, results):
    for position in results:
        fields = list(position)
        fields[1] = "{:.6f}".format(fields[1])  # format timestamp
        print(*fields, file=output)


def load_positions(fname):
    num_fields = len(np.genfromtxt(fname, max_rows=1))  # FIXME
    dtype = {
        'names': POSITION_INFO_DTYPE['names'][:num_fields],
        'formats': POSITION_INFO_DTYPE['formats'][:num_fields]
    }
    data = np.genfromtxt(fname, dtype=dtype)
    return data


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

    tdoa_groups = tdoa_est.load_tdoa_groups(args.tdoa)
    rx_pos = tdoa_est.load_pos_config(args.rx_pos)
    results = solve(tdoa_groups, rx_pos)
    save_positions(args.output, results)


if __name__ == '__main__':
    _main()
