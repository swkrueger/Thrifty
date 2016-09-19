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


def detect_transmitter_windows(freqs):
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
    thresh = np.std(cnts) * 2
    print("New bin threshold:", thresh)

    peaks = []
    below_thresh = True
    above_thresh_start = None
    for i, cnt in enumerate(cnts):
        if not below_thresh and cnt < thresh:
            peaks.append((above_thresh_start, i))
            above_thresh_start = None
            below_thresh = True
        if below_thresh and cnt > thresh:
            above_thresh_start = i
            below_thresh = False
    if not below_thresh:
        peaks.append((above_thresh_start, len(cnts) - 1))

    edges = [(peaks[i][1] + peaks[i+1][0]) // 2 for i in range(len(peaks)-1)]
    edges = np.concatenate([[first_bin],
                            np.array(edges) + first_bin,
                            [last_bin]])

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

    print(len(detections))
    edges = {}
    for rxid, rx_detections in detections_by_rx.iteritems():
        print(rxid, len(rx_detections))
        freqs = np.array([d.carrier_info.bin for d in rx_detections])
        edges[rxid] = detect_transmitter_windows(freqs)[:-1]

    txids = [np.digitize(d.carrier_info.bin, edges[d.rxid]) - 1
             for d in detections]

    return txids


def classify_transmitters(detections, nominal_freqs):
    """Identify transmitter IDs based on the closest nominal frequency."""
    # nominal_freqs = [(freq, txid) for txid, freq in txfreqs.iteritems()]
    # nominal_freqs.sort()
    # for i in range(len(nominal_freqs)):
    #     if nominal_freqs[i][0] > nominal_freqs[i+1][0]:
    #         raise Exception("Frequency spacing between transmitter {} and {}"
    #                         " too small: it should be at least {}".format(
    #                             nominal_freqs[i][1], nominal_freqs[i+1][1],
    #                             window_size))
    txids = []
    for detection in detections:
        freq = detection.carrier_info.bin + detection.carrier_info.offset
        best_dfreq, best_txid = None, None
        for txid, nominal_freq in nominal_freqs.iteritems():
            dfreq = abs(freq - nominal_freq)
            if best_dfreq is None or dfreq < best_dfreq:
                best_dfreq, best_txid = dfreq, txid
        txids.append(best_txid)
    return txids


def identify_transmitters(detections, nominal_freqs):
    """
    Identify transmitters and add TX info to detections.
    The DetectionResult object (detections) is changed in-place.
    """

    if nominal_freqs is None:
        txids = auto_classify_transmitters(detections)
    else:
        txids = classify_transmitters(detections, nominal_freqs)

    for i, detection in enumerate(detections):
        detection.txid = txids[i]


def identify_duplicates(detections):
    """
    Returns a mask for filtering duplicate detections for the same transmitter.

    The block prior to or after the full detection may contain a portion of
    positioning signal and also trigger a detection. It is thus necessary
    to remove those "duplicate" detections.
    It is assumed that all detections were captured by the same receiver.
    """
    array = toads_data.toads_array(detections, with_ids=True)

    # Sort by receiver ID, then transmitter ID, then block ID, then timestamp
    idx = np.argsort(array[['rxid', 'txid', 'block', 'timestamp']])

    cur = array[idx]
    prev = np.roll(cur, 1)
    next_ = np.roll(cur, -1)

    # TODO: only filter if SOA is within code_len
    mask_prev = ((cur['block'] == prev['block'] + 1) &
                 (cur['energy'] < prev['energy']))
    mask_next = ((cur['block'] == next_['block'] - 1) &
                 (cur['energy'] < next_['energy']))
    mask = ~(mask_prev | mask_next)

    reverse_idx = np.argsort(idx)

    return mask[reverse_idx]


def filter_duplicates(detections):
    """Return detections with duplicates removed, sorted by timestamp."""
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


def load_txfreqs(file_):
    if file_ is None:
        return None
    strings = parse_kvconfig(file_)
    txfreqs = {int(txid): float(nominal_freq)
               for txid, nominal_freq in strings.iteritems()}
    return txfreqs


def generate_toads(output, toad_globs, nominal_freqs):
    detections, filenames = load_toad_files(toad_globs)
    output.write("# source_files: [%s]\n" % (' '.join(filenames)))
    identify_transmitters(detections, nominal_freqs)
    filtered = filter_duplicates(detections)

    print("Removed {} duplicates from {} detections.".format(
        len(detections)-len(filtered), len(detections)))

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
    parser.add_argument('-t', '--tx-frequencies', type=argparse.FileType('r'),
                        help="nominal frequencies of transmitters in terms of"
                             "the DFT index [default: auto-detect]")
    # TODO: eliminate detections not within window:
    # parser.add_argument('-w', '--window', dest='window',
    #                     type=float, default=10,
    #                     help="size of frequency window with center at the "
    #                     "nominal frequency in which a transmitter's carrier "
    #                     "frequency is to be expected, in terms of DFT "
    #                     "indices [default: 10]")
    # TODO: --rx-freq-correction = apply frequency corrections to RX detections
    #                              e.g. @rx0: -4 -- can include in tx-freq.conf

    args = parser.parse_args()

    txfreqs = load_txfreqs(args.tx_frequencies)
    generate_toads(args.output, args.toad_file, txfreqs)


if __name__ == "__main__":
    _main()
