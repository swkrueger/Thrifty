#!/usr/bin/env python

"""
Merge multiple .peakd files into a single .peaks file.
"""

import argparse

def merge(output, *peakds):
    # timestamp, rxid, params = 
    output.write('# source_files: [%s]\n' % (' '.join(peakds)))
    input_data = [open(f, 'rb').readlines() for f in args.filename]

    a = []
    for rx_id, lines in enumerate(input_data):
        for l in lines:
            timestamp = float(l.split(' ', 1)[0])
            a.append((timestamp, rx_id, l))
    a.sort()

    for _, rx_id, data in a:
        output.write('%d %s' % (rx_id, data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--output', type=argparse.FileType('wb'),
            default='-', help='output file (default is stdout)')
    parser.add_argument('filename', type=str,
            nargs='+', help='peakd file')

    args = parser.parse_args()

    merge(args.output, *args.filename)

