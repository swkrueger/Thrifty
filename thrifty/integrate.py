#!/usr/bin/env python

"""
Merge RX detections, identify transmitter IDs, filter detections.

Merge multiple .toad files, identify transmitter IDs based on carrier
frequency, remove duplicate detections, and output .toads file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import itertools

import numpy as np

from thrifty import toads_data


def quantize_freqs(freqs):
    """
    Parameters
    ----------
    freqs : :class:`numpy.ndarray`
        Carrier bins of detections.

    Returns
    -------
    edges : :class:`numpy.ndarray`
    """

    first_bin = np.min(freqs)
    cnts = np.bincount(freqs - first_bin)
    last_bin = first_bin + len(cnts)
    thresh = np.mean(cnts) / 2

    transitions = []
    below_thresh = True
    for i, cnt in enumerate(cnts):
        if not below_thresh and cnt < thresh:
            transitions.append(i)
            below_thresh = True
        if below_thresh and cnt > thresh:
            if len(transitions) > 0:
                transitions.append(i)
            below_thresh = False
    if below_thresh:
        transitions.pop()

    transitions = np.array(transitions)
    # print transitions
    edges = (transitions[1:] + transitions[:-1]) // 2 + first_bin
    edges = np.concatenate([[first_bin], edges, [last_bin]])

    print("Freq bin counts: {} ++ {}".format(first_bin, cnts))
    print("Detected {} transmitter(s):".format(len(edges) - 1))

    for i in range(len(edges) - 1):
        print(" {}: bins {} - {}".format(i, edges[i], edges[i+1] - 1))

    return edges


def identify(freqs):
    """Identify transmitter IDs based on carrier frequency."""
    edges = quantize_freqs(freqs)
    return np.digitize(freqs, edges[:-1]) - 1


def filter_duplicates(detections):
    """
    Filter duplicate detections for the same transmission.

    The block prior to or after the full detection may contain a portion of
    positioning signal and also trigger a detection. It is thus necessary
    to remove those "duplicate" detections.
    It is assumed that all detections were captured by the same receiver.
    """
    # Sort by transmitter ID, then block ID, then timestamp
    idx = np.argsort(detections[['txid', 'block', 'timestamp']])

    cur = detections[idx]
    prev = np.roll(cur, 1)
    next_ = np.roll(cur, -1)

    mask_prev = ((cur['block'] == prev['block'] + 1) &
                 (cur['energy'] < prev['energy']))
    mask_next = ((cur['block'] == next_['block'] - 1) &
                 (cur['energy'] < next_['energy']))
    mask = ~(mask_prev | mask_next)

    reverse_idx = np.argsort(idx)

    # print np.column_stack([cur['txid'], cur['block'],
    #                        cur['energy'].astype('int'), mask])
    # print np.diff(cur[mask]['block'] * 6107 + cur[mask]['sample'])

    return mask[reverse_idx]


def integrate(*toad_list):
    """
    For each RX: identify TXs, add RX and TX info, merge.

    Warning: DetectionResult object is changed: RX and TX IDs are added.

    Returns: list of DetectionResult
        Detections from all RXs with RX and TX ID, sorted by timestamp.
    """

    toads = []

    for rxid, toad in enumerate(toad_list):
        print("Receiver #{}:".format(rxid))
        detections = toads_data.toads_array(toad, with_ids=False)
        txids = identify(detections['carrier_bin'])

        # assign txids and rxids
        for i, detection in enumerate(toad):
            detections['rxid'][i] = rxid
            detections['txid'][i] = txids[i]
            detection.rxid = rxid
            detection.txid = txids[i]

        mask = filter_duplicates(detections)
        filtered = list(itertools.compress(toad, mask))
        toads.extend(filtered)
        # print len(toad), len(filtered), np.sum(mask)

        print('')

    toads.sort(key=lambda x: x.timestamp)

    # for t in toads:
    #     print t.rxid, t.txid, t.block, t.timestamp

    return toads


def generate_toads(output, *toad_streams):
    output.write("# source_files: [%s]\n" % (' '.join(toad_streams)))
    # input_data = [open(fn, 'rb').readlines() for fn in toad_streams]

    # load toad files
    toad_list = []
    for stream in toad_streams:
        data = toads_data.load_toad(open(stream, 'rb'))
        toad_list.append(data)

    toads = integrate(*toad_list)
    for detection in toads:
        output.write(detection.serialize() + '\n')


def _main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('toad_file', type=str, nargs='+', help="toad file")
    parser.add_argument('-o', '--output', type=argparse.FileType('wb'),
                        default='-', help="output file (default is stdout)")

    args = parser.parse_args()
    generate_toads(args.output, *args.toad_file)


if __name__ == "__main__":
    _main()
