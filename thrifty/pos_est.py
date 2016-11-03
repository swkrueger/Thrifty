#!/usr/bin/env python

"""
Estimate position from TDOA values.
"""

from __future__ import division
from __future__ import print_function

import scipy.optimize
import numpy as np
import itertools

from thrifty import tdoa_est

SPEED_OF_LIGHT = tdoa_est.SPEED_OF_LIGHT

POSITION_INFO_DTYPE = {
    'names': ('group_id', 'timestamp', 'tx', 'dop', 'snr', 'x', 'y', 'z'),
    'formats': ('i4', 'f8', 'i4', 'f8', 'f8', 'f8', 'f8', 'f8')
}
# TODO: split DOP into multiple values (one per dimension)

MAX_DIST = 10e3


class EstimationError(Exception):
    pass


def solve_1d(tdoa_array, rx_pos):
    """Simple 1D position estimator for 2xRX."""
    rx0 = rx_pos.keys()[0]
    rx1 = rx_pos.keys()[1]

    assert len(rx_pos) == 2
    assert len(rx_pos[rx0]) == 1
    assert len(tdoa_array) == 1

    tdoa_pos = tdoa_array['tdoa'][0] * SPEED_OF_LIGHT
    rx_dist = rx_pos[rx0] + rx_pos[rx1]
    if rx_pos[rx0] > rx_pos[rx1]:
        position = (rx_dist - tdoa_pos) / 2
    else:
        position = (rx_dist + tdoa_pos) / 2

    return (position,), tdoa_array['snr'][0]


def solve_analytically(tdoa_array, rx_pos):
    # TODO
    pass


def solve_numerically(tdoa_array, rx_pos):
    """Solve position using the Levenberg-Marquardt minimization algorithm."""
    # TODO: use analytic solution or previous position as initial value
    # TODO: experiment with different algorithms
    # TODO: use SNR or some confidence value as weight

    dims = len(rx_pos[0])
    uniq_rx = np.unique(np.concatenate([tdoa_array['rx0'], tdoa_array['rx1']]))
    if len(uniq_rx) < dims + 1:
        raise EstimationError("Underdetermined")

    rx_coords = np.array(rx_pos.values())
    min_bounds = np.amin(rx_coords, axis=0) - MAX_DIST
    max_bounds = np.amax(rx_coords, axis=0) + MAX_DIST

    rx0 = np.array([rx_pos[rxid] for rxid in tdoa_array['rx0']])
    rx1 = np.array([rx_pos[rxid] for rxid in tdoa_array['rx1']])

    x0 = [0.1, 0.1]

    def model(pos):
        # position relative to {rx0, rx1}
        pos_rx0, pos_rx1 = rx0 - pos, rx1 - pos
        # distance to {rx0, rx1}
        dist0 = np.linalg.norm(pos_rx0, axis=1)
        dist1 = np.linalg.norm(pos_rx1, axis=1)
        # predicted TDOA (in m)
        predicted_tdoa = dist0 - dist1

        residuals = tdoa_array['tdoa'] * SPEED_OF_LIGHT - predicted_tdoa
        return residuals

    def jac(pos):
        pos_rx0, pos_rx1 = rx0 - pos, rx1 - pos
        dist0 = np.linalg.norm(pos_rx0, axis=1)
        dist1 = np.linalg.norm(pos_rx1, axis=1)
        return pos_rx0 / dist0[:, None] - pos_rx1 / dist1[:, None]

    res = scipy.optimize.least_squares(model, x0,
                                       jac=jac,
                                       bounds=(min_bounds, max_bounds))

    # TODO: also return residual or a measure of the quality or confidence of
    #       the estimate

    snr_mean = np.mean(tdoa_array['snr'])

    return res.x, snr_mean


def dop_matrix(pos, rx_pos, rx_pairs):
    pos = np.array(pos)
    rx0 = np.array([np.array(rx_pos[rxid]) for rxid, _ in rx_pairs])
    rx1 = np.array([np.array(rx_pos[rxid]) for _, rxid in rx_pairs])
    pos_rx0, pos_rx1 = rx0 - pos, rx1 - pos
    dist0 = np.linalg.norm(pos_rx0, axis=1)
    dist1 = np.linalg.norm(pos_rx1, axis=1)
    G = pos_rx0 / dist0[:, None] - pos_rx1 / dist1[:, None]
    H_inv = G.T.dot(G)
    try:
        H = np.linalg.inv(H_inv)
    except np.linalg.LinAlgError:
        H = None
    return H


def dop(pos, rx_pos, rx_pairs):
    matrix = dop_matrix(pos, rx_pos, rx_pairs)
    if matrix is None:
        return -1
    return np.sqrt(np.trace(matrix))


def solve(tdoa_groups, rx_pos):
    # TODO: estimate confidence with SNR and DOP and return with pos
    num_rx = len(rx_pos)
    dimensions = len(rx_pos[rx_pos.keys()[0]])

    results = []
    for group_id, timestamp, tx, tdoas in tdoa_groups:
        try:
            if num_rx == 2 and dimensions == 1:
                coords, snr = solve_1d(tdoas, rx_pos)
            else:
                coords, snr = solve_numerically(tdoas, rx_pos)
            rx_pairs = zip(tdoas['rx0'], tdoas['rx1'])
            dop_est = dop(coords, rx_pos, rx_pairs)
            results.append((group_id, timestamp, tx, dop_est, snr) +
                           tuple(coords))
            # print(tx, dop_est, snr, coords)
        except EstimationError as e:
            print("Failed to estimate group #{}: {}".format(group_id, e))

    # TODO: apply Kalmin filter or something to average out the position
    #       estimates (move to separate module)

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
