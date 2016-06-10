#!/usr/bin/env python

'''
Build clock sync model from beacon transmissions.
'''

import argparse
import data
import matchmaker
import sys
from collections import deque

import numpy as np
import matplotlib.pyplot as plt


# TODO: support multiple beacons
#  (we'll have to take RX and beacon positions as parameter)
# TODO: support >2 receivers
def clock_sync(detections, matches, beacon):
    '''
    Args:
        detections: detection array
        matches: match array
        beacon: beacon ID
    '''
    num_rx = max([x.rxid for x in toads]) + 1

    # Extract beacon's matches
    # We currently do not support incomplete matches
    matches = np.array([x for x in matches
            if sum([y != -1 for y in x]) == num_rx and
               detections['txid'][x[0]] == beacon])
    print 'Number of detection groups:', len(matches)

    soa = detections['soa'][matches]
    sdoa = soa[:,1] - soa[:,0]
    dsdoa = np.abs(np.diff(sdoa))
    is_rapid_change = dsdoa > np.mean(dsdoa) * 10
    discontinuities = np.where(is_rapid_change)[0]

    # print np.column_stack([np.diff(soa[:,0]), np.diff(soa[:,1]), sdoa[1:],
    #                        dsdoa, rapid_change]).astype(int)
    print 'Number of discontinuities:', np.size(discontinuities)
    print 'Discontinuities:', ' '.join(map(str, discontinuities))

    edges = np.concatenate([[0], discontinuities, [len(matches)]])
    for i in range(len(edges) - 1):
        a, b = edges[i] + 1, edges[i + 1]
        if a + 3 >= b:
            continue
        # print 'Range', a, b
        sync(soa[a:b])


    # # print detections['rxid'][matches]
    # # print detections['txid']
    # print sdoa

    # detect discontinuities


def sync(soa, deg=2, plot=False):
    # sdoa = soa[:,1] - soa[:,0]
    # dsdoa = np.abs(np.diff(sdoa))
    # print np.column_stack([np.diff(soa[:,0]), np.diff(soa[:,1]), sdoa[1:],
    #                        dsdoa]).astype(int)

    soa1, soa2 = soa[:,0], soa[:,1]
    coef = np.polyfit(soa1, soa2, deg)
    fit = np.poly1d(coef)
    residuals = soa2 - fit(soa1)

    s2m = 3e8 / 2.2e6
    print 'residuals: std dev = {:.01f} m; max = {:.01f} m'.format(
            np.std(residuals) * s2m,
            np.max(np.abs(residuals)) * s2m)

    # sdoa = soa2 - soa1
    # dsdoa = np.concatenate([[0], np.diff(sdoa)])
    # dsoa1 = np.concatenate([[0, 0], np.diff(np.diff(soa1))])
    # dsoa2 = np.concatenate([[0, 0], np.diff(np.diff(soa2))])
    # print np.column_stack([soa1, soa2, dsoa1, dsoa2, dsdoa, residuals*1000]).astype('int')

    if plot:
        plt.figure(figsize=(11,5))
        plt.subplot(1,2,1)
        plt.plot(soa1, residuals, '.-')
        plt.title('residuals')
        plt.xlabel('RX1 sample')
        plt.ylabel('Residual (samples)')
        plt.grid()

        plt.subplot(1,2,2)
        plt.hist(residuals, 20)
        plt.title('Histogram: residuals')
        plt.grid()
        # plt.savefig('clock_sync_{}_captured.pdf'.format(basename, int(soa1[0])), format='pdf')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('toads',
                        type=argparse.FileType('rb'), default='rx.toads',
                        help='toads data (\'-\' streams from stdin)')
    parser.add_argument('matches',
                        type=argparse.FileType('rb'), default='rx.match',
                        help='toads data (\'-\' streams from stdin)')
    parser.add_argument('--id', dest='beacon_id', type=int, default=0,
                        help='transmitter ID of beacon')

    args = parser.parse_args()

    toads = data.load_toads(args.toads)
    detections = data.toads_array(toads, with_ids=True)
    matches = matchmaker.load_matches(args.matches)

    np.set_printoptions(threshold=np.nan) # FIXME
    clock_sync(detections, matches, args.beacon_id)

