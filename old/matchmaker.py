#!/usr/bin/env python

'''
Match detections from the same transmitter detected by multiple receivers.
'''

import argparse
import data
import sys
from collections import deque

def match(toads, window, min_match=2):
    '''
    Arguments:
        toads: list of DetectionResult sorted by timestamp
        window: size of timestamp window in seconds
        min_match: minimum number of receivers that should receive a
                   transmission for it to be considered a match
    '''
    num_rx = max([x.rxid for x in toads]) + 1
    N = len(toads)

    killed = [False] * len(toads)
    matches = []
    misses = []

    for i in range(N):
        if killed[i]:
            continue

        match = [-1] * num_rx
        match[toads[i].rxid] = i

        for j in range(i + 1, N):
            if toads[j].txid != toads[i].txid:
                continue
            if toads[j].timestamp > toads[i].timestamp + window:
                break
            killed[j] = True

            if match[toads[j].rxid] != -1:
                m = match[toads[j].rxid]
                sys.stdout.write('Warning: multiple detections for RX {} '
                        'and TX {}: detecion #{} and #{} collides\n'.format(
                        toads[j].txid, toads[j].rxid, m, j))
                k = m if toads[m].toa.energy > toads[j].toa.energy else j
            else:
                k = j

            # print j, toads[j].rxid
            match[toads[j].rxid] = k

        if sum([x != -1 for x in match]) > 1:
            # print 'match', match
            matches.append(match)
        else:
            # print 'no match', i
            misses.append(i)

    return matches, misses


def load_matches(f):
    matches = []
    for line in f:
        if len(line) == 0 or line[0] == '#':
            continue
        match = map(int, line.split())
        matches.append(match)
    return matches


def save_matches(matches, f):
    for m in matches:
        f.write(' '.join(map(str, m)) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('input',
                        type=argparse.FileType('rb'), default='rx.toads',
                        help='input data (\'-\' streams from stdin)')
    parser.add_argument('-o', '--output', dest='output',
            type=argparse.FileType('wb'), default='rx.match',
            help='output file (\'-\' for stdout)')
    parser.add_argument('-w', '--window', dest='window', type=float,
            default=0.2, help='size of timestamp window in seconds')
    parser.add_argument('-n', '--num-matches', dest='num_matches', type=int,
            default=2, help='minimum number of receivers that should detect a '
                            'transmission for it to be considered a match')
    # TODO: --verbose

    args = parser.parse_args()

    toads = data.load_toads(args.input)
    matches, misses = match(toads, args.window, args.num_matches)

    print 'Number of matches:', len(matches)
    print 'Number of misses:', len(misses)

    save_matches(matches, args.output)

