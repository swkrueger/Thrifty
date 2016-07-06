#!/usr/bin/env python

'''
Identify transmitter IDs and remove duplicate detections.

Transmitter IDs are classified based on carrier frequency.
'''

import argparse
import data
import numpy as np

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
    avg = np.mean(cnts)
    print cnts

    transitions = []
    below_avg = False
    for i, c in enumerate(cnts):
        if below_avg == False and c < avg:
            transitions.append(i)
            below_avg = True
        if below_avg == True and c > avg:
            transitions.append(i)
            below_avg = False
    if below_avg:
        transitions.pop()

    transitions = np.array(transitions)
    print transitions
    bins = (transitions[1:] + transitions[:-1]) / 2 + bin0
    bins = np.insert(bins, 0, bin0)
    print bins
    
    return np.digitize(freqs, bins) - 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('input',
                        type=argparse.FileType('rb'), default='-',
                        help='input data (\'-\' streams from stdin)')

    args = parser.parse_args()

    toads = data.load_toads(args.input)
    detections = data.toads_array(toads)

    txids = identify(detections['carrier_bin'])
    np.set_printoptions(threshold=np.nan)
    print txids

