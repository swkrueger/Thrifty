"""Calculate stats like the std. dev. from slices of the TDOA data."""

import numpy as np
import matplotlib.pyplot as plt

from thrifty import tdoa_est


def _parse_range(string):
    if string is None:
        return None
    a, b = string.split('-')
    a, b = int(a), int(b)
    return a, b


def _main():
    import argparse

    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.add_argument('tdoa', nargs='?',
                        type=argparse.FileType('r'), default='data.tdoa',
                        help="tdoa data (\"-\" streams from stdin)")
    parser.add_argument('--rx0', type=int, default=0,
                        help="receiver ID of first receiver")
    parser.add_argument('--rx1', type=int, default=1,
                        help="receiver ID of second receiver")
    parser.add_argument('--tx', type=int, default=0,
                        help="ID of mobile unit to analyze")
    parser.add_argument('--timestamp', type=_parse_range, default=None,
                        help="limit the analysis to TDOAs within a specific "
                             "range of timestamps")
    parser.add_argument('--detidx', type=_parse_range, default=None,
                        help="limit the analysis to TDOAs within a specific "
                             "range of detection IDs")
    args = parser.parse_args()

    if args.rx0 > args.rx1:
        args.rx0, args.rx1 = args.rx1, args.rx0

    tdoa_matrix = tdoa_est.load_tdoa_matrix(args.tdoa)
    selection = tdoa_matrix[(tdoa_matrix['rx0'] == args.rx0) &
                            (tdoa_matrix['rx1'] == args.rx1) &
                            (tdoa_matrix['tx'] == args.tx)]
    if args.timestamp is not None:
        start, stop = args.timestamp
        selection = selection[(selection['timestamp'] >= start) &
                              (selection['timestamp'] <= stop)]
    if args.detidx is not None:
        start, stop = args.detidx
        selection = selection[(selection['det0_idx'] >= start) &
                              (selection['det0_idx'] <= stop) &
                              (selection['det1_idx'] >= start) &
                              (selection['det1_idx'] <= stop)]

    tdoa_meter = selection['tdoa'] * tdoa_est.SPEED_OF_LIGHT
    timestamps = selection['timestamp']

    mean = np.mean(tdoa_meter)
    std = np.std(tdoa_meter)
    rms = np.sqrt(np.mean(tdoa_meter**2))
    print("TDOA bias: %.3f m" % mean)
    print("TDOA std dev: %.3f m" % std)
    print("TDOA RMS: %.3f m" % rms)

    plt.plot(timestamps, tdoa_meter, marker='.')
    # TODO: plot histogram of residuals
    # TODO: --export to save to a .PDF file
    plt.show()

if __name__ == '__main__':
    _main()
