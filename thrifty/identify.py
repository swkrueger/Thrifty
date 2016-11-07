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
from collections import defaultdict
import glob
import itertools

import numpy as np

from thrifty import toads_data
from thrifty.settings import parse_kvconfig


def detect_transmitter_windows(freqs, verbose=False):
    """Detect transmitter frequency windows automatically.

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
    low_thresh = np.std(cnts) * 0.4
    high_thresh = np.std(cnts) * 1.25

    peaks = []
    below_thresh = True
    above_thresh_start = None
    for i, cnt in enumerate(cnts):
        if not below_thresh and cnt < low_thresh:
            peaks.append((above_thresh_start, i))
            above_thresh_start = None
            below_thresh = True
        if below_thresh and cnt > high_thresh:
            above_thresh_start = i
            below_thresh = False
    if not below_thresh:
        peaks.append((above_thresh_start, len(cnts) - 1))

    edges = [(peaks[i][1] + peaks[i+1][0]) // 2 for i in range(len(peaks)-1)]
    edges = np.concatenate([[first_bin],
                            np.array(edges) + first_bin,
                            [last_bin]])

    if verbose:
        print("Window threshold: low = {}; high = {}:"
              .format(low_thresh, high_thresh))
        print("Freq bin counts: {} ++ {}".format(first_bin, cnts))
        print("Detected {} transmitter(s):".format(len(edges) - 1))

        for i in range(len(edges) - 1):
            start, stop = edges[i], edges[i+1] - 1
            bins = cnts[start-first_bin:stop-first_bin+1]
            print(" {}: bins {} - {}".format(i, start, stop))
            print("     {}".format(bins))

    return edges


def auto_classify_transmitters(detections):
    """Identify transmitter IDs based on carrier frequency."""
    # Split by receiver
    detections_by_rx = defaultdict(list)
    for detection in detections:
        detections_by_rx[detection.rxid].append(detection)

    edges = {}
    for rxid, rx_detections in detections_by_rx.iteritems():
        freqs = np.array([d.carrier_info.bin for d in rx_detections])
        rx_edges = detect_transmitter_windows(freqs)

        summary = ("Detected {} transmitter(s) at RX {}:"
                   .format(len(rx_edges) - 1, rxid))
        for i in range(len(rx_edges) - 1):
            summary += " {}-{}".format(rx_edges[i], rx_edges[i+1] - 1)
        print(summary)

        edges[rxid] = rx_edges[:-1]

    txids = [np.digitize(d.carrier_info.bin, edges[d.rxid]) - 1
             for d in detections]

    return txids


def classify_transmitters(detections, freqmap):
    """Identify transmitter IDs based on the closest nominal frequency."""
    txids = []
    for detection in detections:
        freq = detection.carrier_info.bin + detection.carrier_info.offset
        this_txid = -1  # unidentifier (FIXME: don't use magic number)
        for txid, range_ in freqmap[detection.rxid].iteritems():
            start, stop = range_
            if freq >= start and freq <= stop:
                this_txid = txid
        txids.append(this_txid)
    return txids


def identify_transmitters(detections, freqmap):
    """
    Identify transmitters and add TX info to detections.
    The DetectionResult object (detections) is changed in-place.
    """

    if freqmap is None:
        txids = auto_classify_transmitters(detections)
    else:
        txids = classify_transmitters(detections, freqmap)

    for i, detection in enumerate(detections):
        detection.txid = txids[i]


def identify_duplicates(detections):
    """
    Returns a mask for filtering duplicate detections for the same transmitter.

    The block prior to or after the full detection may contain a portion of
    positioning signal and also trigger a detection. It is thus necessary
    to remove those "duplicate" detections.
    It is assumed that all detections were captured by the same receiver.

    The mask will exclude unidentified detections.
    """
    array = toads_data.toads_array(detections, with_ids=True)

    # Sort by receiver ID, then transmitter ID, then block ID, then timestamp
    idx = np.argsort(array[['rxid', 'txid', 'block', 'timestamp']])

    cur = array[idx]
    prev = np.roll(cur, 1)
    next_ = np.roll(cur, -1)

    # TODO: only filter if SOA is within code_len
    mask_unidentified = (cur['txid'] == -1)  # FIXME: magic number
    mask_prev = ((cur['block'] == prev['block'] + 1) &
                 (cur['energy'] < prev['energy']))
    mask_next = ((cur['block'] == next_['block'] - 1) &
                 (cur['energy'] < next_['energy']))
    mask = ~(mask_prev | mask_next | mask_unidentified)

    reverse_idx = np.argsort(idx)

    return mask[reverse_idx]


def filter_duplicates(detections):
    """Return detections with duplicates and unidentified detections removed,
    sorted by timestamp."""
    mask = identify_duplicates(detections)
    filtered = list(itertools.compress(detections, mask))
    filtered.sort(key=lambda x: x.timestamp)
    return filtered


def load_toad_files(toad_globs):
    filenames = []
    for toad_glob in toad_globs:
        filenames.extend(glob.glob(toad_glob))

    detections = []
    for filename in filenames:
        with open(filename, 'r') as file_:
            detections.extend(toads_data.load_toad(file_))

    return detections, filenames


def load_freqmap(file_):
    if file_ is None:
        return None
    strings = parse_kvconfig(file_)

    tx_ranges = {}
    rx_offset = {}

    for key, value in strings.iteritems():
        if key[0] == '@':
            rx_offset[int(key[1:])] = float(value)
        else:
            # TODO: use regex
            start, stop = [float(x.strip()) for x in value.split('-')]
            tx_ranges[int(key)] = (start, stop)
            # TODO: ensure that ranges do not overlap

    freq_map = {}
    for rxid, offset in rx_offset.iteritems():
        freq_map[rxid] = {}
        for txid, range_ in tx_ranges.iteritems():
            start, stop = range_
            freq_map[rxid][txid] = (start+offset, stop+offset)
            print(rxid, txid, freq_map[rxid][txid])

    return freq_map


def integrate(detections, freqmap=None):
    """Identify and filter."""
    identify_transmitters(detections, freqmap)
    filtered = filter_duplicates(detections)
    return filtered


def generate_toads(output, toad_globs, freqmap):
    detections, filenames = load_toad_files(toad_globs)
    output.write("# source_files: [%s]\n" % (' '.join(filenames)))
    filtered = integrate(detections, freqmap)

    print("Removed {} duplicates / unidentified transmisisons "
          "from {} detections.".format(len(detections)-len(filtered),
                                       len(detections)))

    for detection in filtered:
        output.write(detection.serialize() + '\n')


def _main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('toad_file', type=str, nargs='*', default="*.toad",
                        help="toad file(s) from receivers [default: *.toad]")
    parser.add_argument('-o', '--output', type=argparse.FileType('wb'),
                        default='data.toads',
                        help="output file [default: *.taods]")
    parser.add_argument('-m', '--map', type=argparse.FileType('r'),
                        help="schema for mapping DFT index to transmitter ID "
                             "[default: auto-detect]")
    args = parser.parse_args()

    freqmap = load_freqmap(args.map)
    generate_toads(args.output, args.toad_file, freqmap)


if __name__ == "__main__":
    _main()
