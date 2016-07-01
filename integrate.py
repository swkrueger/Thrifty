#!/usr/bin/env python

'''
Merge RX detections, identify transmitter IDs, filter detections.

Merge multiple .toad files, identify transmitter IDs based on carrier
frequency, remove duplicate detections, and output .toads file.
'''

import argparse
import data
import numpy as np
import itertools

# (TODO: function should be called separately for different receivers)
# (TODO: output debug info such as the bins)
def identify(freqs):
    '''
    Identify transmitter IDs of detections based on carrier frequency.

    Args:
        freqs: nparray of carrier bins of detections
    Return: nparray of transmitter IDs
    '''

    bin0 = np.min(freqs)
    cnts = np.bincount(freqs - bin0)
    thresh = np.mean(cnts) / 2

    transitions = []
    below_thresh = False
    for i, c in enumerate(cnts):
        if below_thresh == False and c < thresh:
            transitions.append(i)
            below_thresh = True
        if below_thresh == True and c > thresh:
            transitions.append(i)
            below_thresh = False
    if below_thresh:
        transitions.pop()

    transitions = np.array(transitions)
    # print transitions
    bins = (transitions[1:] + transitions[:-1]) / 2 + bin0
    bins = np.insert(bins, 0, bin0)
    # print bins

    print 'Freq bin counts: {} ++ {}'.format(bin0, cnts)
    # print 'Transitions:', transitions
    print 'Detected {} transmitter(s):'.format(len(bins))
    binN = bin0 + len(cnts)
    b = np.concatenate([bins, [binN]])
    for i in range(len(b) - 1):
        print ' {}: bins {} - {}'.format(i, b[i], b[i + 1] - 1)
    
    return np.digitize(freqs, bins) - 1


def filter_duplicates(detections):
    '''
    Filter 'duplicate' detections.

    It is assumed that all detections were captured on the same receiver.
    '''
    # Sort by transmitter ID, then block ID, then timestamp
    idx = np.argsort(detections[['txid', 'block', 'timestamp']])

    cur = detections[idx]
    prev = np.roll(cur, 1)
    next = np.roll(cur, -1)

    mask1 = (cur['block'] == prev['block'] + 1) & (cur['energy'] < prev['energy'])
    mask2 = (cur['block'] == next['block'] - 1) & (cur['energy'] < next['energy'])
    mask = ~(mask1 | mask2)

    reverse_idx = np.argsort(idx)

    # print np.column_stack([cur['txid'], cur['block'], cur['energy'].astype('int'), mask])
    # print np.diff(cur[mask]['block'] * 6107 + cur[mask]['sample'])

    return mask[reverse_idx]


def integrate(*toad_list):
    '''
    For each RX: identify TXs, add RX and TX info, merge.

    Warning: DetectionResult object is changed: RX and TX IDs are added.

    Returns: list of DetectionResult
        Detections from all RXs with RX and TX ID, sorted by timestamp.
    '''

    toads = []

    for rxid, toad in enumerate(toad_list):
        print 'Receiver #{}:'.format(rxid)
        detections = data.toads_array(toad, with_ids=False)
        txids = identify(detections['carrier_bin'])

        # assign txids and rxids
        for i in range(len(toad)):
            detections['rxid'][i] = rxid
            detections['txid'][i] = txids[i]
            toad[i].rxid = rxid
            toad[i].txid = txids[i]

        mask = filter_duplicates(detections)
        filtered = list(itertools.compress(toad, mask))
        toads.extend(filtered)
        # print len(toad), len(filtered), np.sum(mask)

        print ''

    toads.sort(key=lambda x: x.timestamp)

    # for t in toads:
    #     print t.rxid, t.txid, t.block, t.timestamp
    
    return toads


def generate_peaks(output, *toad_streams):
    output.write('# source_files: [%s]\n' % (' '.join(toad_streams)))
    # input_data = [open(fn, 'rb').readlines() for fn in toad_streams]

    # load toad files
    toad_list = []
    for stream in toad_streams:
        d = data.load_toad(open(stream, 'rb'))
        toad_list.append(d)

    toads = integrate(*toad_list)
    for t in toads:
        output.write(t.serialize() + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('peakd_file', type=str,
            nargs='+', help='peakd file')
    parser.add_argument('-o', '--output', type=argparse.FileType('wb'),
            default='-', help='output file (default is stdout)')
    # TODO: --verboise

    args = parser.parse_args()

    generate_peaks(args.output, *args.peakd_file)

