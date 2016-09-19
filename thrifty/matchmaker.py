#!/usr/bin/env python

"""
Match detections from the same transmitter detected by multiple receivers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging

from thrifty import toads_data


def match_toads(toads, window, min_match=2):
    """Match detections from multiple receivers.

    The coarse timestamp and transmitter ID is used to determine which
    detections originate from the same transmissions.

    Parameters
    ----------
    toads : list of DetectionResult
        List should be sorted by timestamp.
    window : float
        Size of timestamp window in seconds.
    min_match : int
        Minimum number of receivers that should receive a transmission for it
        to be considered a valid match.

    Returns
    -------
    matches : list
        List of matches, each match being a list of indices.
    misses : list
        List of detection indices that could not be matched.
    """
    # pylint: disable=too-many-locals
    num_det = len(toads)

    killed = [False] * len(toads)
    matches = []
    misses = []

    for i in range(num_det):
        if killed[i]:
            continue

        rx_match = {}
        rx_match[toads[i].rxid] = i

        for j in range(i + 1, num_det):
            if toads[j].txid != toads[i].txid:
                continue
            if toads[j].timestamp > toads[i].timestamp + window:
                break
            killed[j] = True

            if toads[j].rxid in rx_match != -1:
                prev = rx_match[toads[j].rxid]
                logging.warning("Multiple detections for RX %d and TX %d: "
                                "detection #%d and #%d collides.",
                                toads[j].rxid, toads[j].txid, prev, j)
                prev_ampl = toads[prev].corr_info.energy
                this_ampl = toads[j].corr_info.energy
                k = prev if prev_ampl > this_ampl else j
            else:
                k = j

            rx_match[toads[j].rxid] = k

        match = rx_match.values()
        if len(match) >= min_match:
            matches.append(match)
        else:
            misses.append(i)

    return matches, misses


def load_matches(file_):
    """Load match data from .match file."""
    matches = []
    for line in file_:
        if len(line) == 0 or line[0] == '#':
            continue
        match = map(int, line.split())
        matches.append(match)
    return matches


def save_matches(matches, file_):
    """Save match data to .match file."""
    for match in matches:
        file_.write(' '.join(map(str, match)) + '\n')


def _main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('input', nargs='?',
                        type=argparse.FileType('rb'), default='data.toads',
                        help=".toads data (\'-\' streams from stdin)")
    parser.add_argument('-o', '--output', dest='output',
                        type=argparse.FileType('wb'), default='data.match',
                        help="output file (\'-\' for stdout)")
    parser.add_argument('-w', '--window', dest='window', type=float,
                        default=0.2,
                        help="size of timestamp window in seconds")
    parser.add_argument('-n', '--num-matches', dest='num_matches',
                        type=int, default=2,
                        help="minimum number of receivers that should detect "
                             "a transmission for a match to be valid")
    args = parser.parse_args()

    toads = toads_data.load_toads(args.input)
    toads.sort(cmp=lambda x, y: x.timestamp < y.timestamp)
    matches, misses = match_toads(toads, args.window, args.num_matches)

    print("Number of matches:", len(matches))
    print("Number of misses:", len(misses))

    save_matches(matches, args.output)


if __name__ == "__main__":
    _main()
