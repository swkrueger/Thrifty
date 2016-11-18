from __future__ import print_function

import argparse
import itertools
import sys
import time

from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

from thrifty import toads_data
from thrifty import matchmaker
from thrifty import tdoa_est
from thrifty import stat_tools

MATCH_WINDOW = 0.2
WINDOW_SIZE = 4
SPEED_OF_LIGHT = tdoa_est.SPEED_OF_LIGHT

TXIDS = [0, 1, 2, 3, 4]


def print_detection_counts(toads, matched_toads):
    print('# Detection counts:')
    print(' - Total number of detections:    ', len(toads))
    print(' - Number of matched detections:  ', len(matched_toads))
    print(' - Number of unmatched detections:', len(toads)-len(matched_toads))
    print()


def print_count_table(matched_toads):
    data = toads_data.toads_array(matched_toads)
    rxids = np.sort(np.unique(data['rxid']))
    txids = np.sort(np.unique(data['txid']))

    rxnum = {rxids[i]: i for i in range(len(rxids))}
    txnum = {txids[i]: i for i in range(len(txids))}

    counts = [[0]*len(rxids) for _ in range(len(txids))]
    for row in data:
        rxid, txid = row['rxid'], row['txid']
        rxidx, txidx = rxnum[rxid], txnum[txid]
        counts[txidx][rxidx] += 1

    headers = ['v TX / RX >'] + list(rxids)
    table = [[txids[i]] + counts[i] for i in range(len(txids))]
    print('# Detection count table:')
    print(tabulate(table, headers=headers))
    # TODO: total row
    print()


def print_snr_table(matched_toads):
    # TODO: fix copy-paste from print_count_table
    data = toads_data.toads_array(matched_toads)
    rxids = np.sort(np.unique(data['rxid']))
    txids = np.sort(np.unique(data['txid']))

    rxnum = {rxids[i]: i for i in range(len(rxids))}
    txnum = {txids[i]: i for i in range(len(txids))}

    corr_peaks = [[[] for _ in range(len(rxids))] for _ in range(len(txids))]
    for row in data:
        rxid, txid = row['rxid'], row['txid']
        rxidx, txidx = rxnum[rxid], txnum[txid]
        corr_peaks[txidx][rxidx].append(row['energy'])
    corr_means = [[int(np.mean(corr_peaks[tx][rx]))
                   if len(corr_peaks[tx][rx]) > 0 else 0
                   for rx in range(len(rxids))]
                  for tx in range(len(txids))]


    headers = ['v TX / RX >'] + list(rxids)
    table = [[txids[i]] + corr_means[i] for i in range(len(txids))]
    print('# Mean corr peak amplitude table:')
    print(tabulate(table, headers=headers))
    print()


def print_match_stats(matches, misses, collisions):
    print("# Match stats:")
    print(" - Number of matches:", len(matches))
    print(" - Number of misses:", len(misses))
    print(" - Number of collisions:", len(collisions))
    print()


def print_match_length_histogram(matches):
    counts = {}
    for match in matches:
        counts[len(match)] = counts.get(len(match), 0) + 1

    print("# Match length histogram:")
    for count in sorted(counts.keys()):
        print(" - {}: {}".format(count, counts[count]))
    print()


def calculate_tdoas(toads, matches, rx1, rx2, beacon, tx):
    if beacon == tx:
        return None

    extract = matchmaker.extract_match_matrix(toads, matches,
                                              [rx1, rx2], [beacon, tx])
    data = toads_data.toads_array(toads)
    # print(data['txid'][extract])
    # extract = [x for x in extract if x[0] is not None and x[1] is not None]
    num_beacon = np.sum([toads[m[0]].txid == beacon for m in extract])
    num_tx = np.sum([toads[m[0]].txid == tx for m in extract])
    # print(len(extract), num_beacon, num_tx)

    if len(extract) == 0 or num_beacon < 3 or num_tx < 3:
        return None
    # if beacon == 2:
    #     print(extract)

    # print("rx1, rx2, beacon, tx:", rx1, rx2, beacon, tx)
    groups, failures = tdoa_est.estimate_tdoas(detections=toads,
                                        matches=extract,
                                        window_size=WINDOW_SIZE,
                                        beacon_pos={beacon: 0.0},
                                        rx_pos={rx1: 0.0, rx2: 0.0},
                                        sample_rate=2.4e6)
    matrix = tdoa_est.groups_to_matrix(groups)

    # tdoas = np.array([g.tdoas['tdoa'] for g in groups])
    # print("Number of TDOAs:", len(groups))
    # print("Number of failures:", len(failures))
    if len(groups) > 1:
        outliers = stat_tools.is_outlier(matrix['tdoa'])
        matrix = matrix[~outliers]
        # print("Number of outliers:", np.sum(outliers))
        # print num outliers / return num outliers

    # import matplotlib.pyplot as plt
    # plt.plot(np.array(tdoas) * SPEED_OF_LIGHT, '.-')
    # plt.show()

    # TODO: return timestamp
    return matrix


def print_tdoa_matrix(toads, matches, rx1, rx2):
    tdoa_matrices = [[calculate_tdoas(toads, matches, rx1, rx2, beacon, tx)
                      for tx in range(5)] for beacon in range(5)]

    print("# TDOA matrices for RX {} and RX {}".format(rx1, rx2))

    plt.figure()
    nrows = len(tdoa_matrices)
    ncols = 0 if nrows == 0 else len(tdoa_matrices[0])
    for row_idx, row in enumerate(tdoa_matrices):
        ncols = len(row)
        for col_idx, column in enumerate(row):
            if column is None:
                continue
            plt.subplot(nrows, ncols, row_idx*ncols + col_idx + 1)
            # tdoas = np.array([g.tdoas['tdoa'] for g in groups])
            # timestamp, tdoas
            plt.plot(column['timestamp'],
                     column['tdoa'] * SPEED_OF_LIGHT, '.-')
            plt.title('Beacon {}; Mobile {}'.format(row_idx, col_idx))
    # plt.tight_layout()
    plt.suptitle('TDOAs: RX {}, {}'.format(rx1, rx2))
    plt.show(block=False)

    tdoas = [[column['tdoa'] if column is not None else []
              for column in row] for row in tdoa_matrices]

    stds = [[np.around(np.std(column) * SPEED_OF_LIGHT, 1)
             if len(column) > 0 else 0.0
             for column in row]
            for row in tdoas]
    means = [[np.around(np.mean(column) * SPEED_OF_LIGHT, 1)
              if len(column) > 0 else 0.0
              for column in row]
             for row in tdoas]
    counts = [[len(column) for column in row] for row in tdoas]

    # TODO: print proper header (e.g. TX, and then "subheader")
    headers = ['v beacon/tx >'] + range(5)
    print("# TDOA variance matrix for {} and {}:".format(rx1, rx2))
    print(tabulate([[i] + stds[i] for i in TXIDS], headers=headers))
    print()
    print("# TDOA mean matrix for {} and {}:".format(rx1, rx2))
    print(tabulate([[i] + means[i] for i in TXIDS], headers=headers))
    print()
    print("# TDOA count matrix for {} and {}:".format(rx1, rx2))
    print(tabulate([[i] + counts[i] for i in TXIDS], headers=headers))
    print('\n')
    # print match SNR
    # print num failures / num outliers matrix
    # incorporate count in variance matrix?


def print_tdoa_variance_matrices(toads, matches):
    rxids = range(4)
    for rx1, rx2 in itertools.combinations(rxids, 2):
        print_tdoa_matrix(toads, matches, rx1, rx2)
    plt.show()


def _main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('input',
                        type=argparse.FileType('rb'), default='data.toads',
                        help=".toads data (\'-\' streams from stdin)")
    parser.add_argument('timeval', type=float, nargs='?', default=0,
                        help="time interval to calculate stats (minutes)."
                             "Use 0 to include everything earlier than 'now'.")
    parser.add_argument('--now', type=float,
                        help="fake the current time (timestamp)."
                             "Use a value less than zero to set the timestamp "
                             "relative to the last sample's timestamp.")
    parser.add_argument('--all', action='store_true')

    args = parser.parse_args()

    toads = toads_data.load_toads(args.input)

    if args.all:
        toads_cut = toads
    else:
        if args.now is None:
            stop_timestamp = int(time.time()) - 10
        elif args.now <= 0:
            stop_timestamp = toads[-1].timestamp + args.now
        else:
            stop_timestamp = args.now

        if args.timeval == 0:
            start_timestamp = toads[0].timestamp
        else:
            start_timestamp = stop_timestamp - args.timeval * 60

        toads_cut = [t for t in toads if (t.timestamp >= start_timestamp and
                                          t.timestamp <= stop_timestamp)]

    if len(toads_cut) == 0:
        print("Cut is empty")
        sys.exit(0)

    # TODO: timestamp-sync.py

    matches, misses, collisions = matchmaker.match_toads(toads_cut,
                                                         MATCH_WINDOW)

    matched_ids = np.sort(np.concatenate(matches))
    matched_toads = [toads[i] for i in matched_ids]
    matched_toads.sort(cmp=lambda x, y: x.timestamp < y.timestamp)

    time0 =min([t.timestamp for t in matched_toads])
    print('Timestamps relative to {:.6f}'.format(time0))
    for t in toads:
        t.timestamp -= time0

    print_detection_counts(toads_cut, matched_toads)
    print_count_table(matched_toads)
    print_snr_table(matched_toads)
    print_match_stats(matches, misses, collisions)
    print_match_length_histogram(matches)
    # TODO: match matrix: how many matches for each (RX, RX) pair
    # (and for each TX?)
    print_tdoa_variance_matrices(toads_cut, matches)


if __name__ == '__main__':
    _main()
