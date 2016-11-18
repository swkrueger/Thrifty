#!/usr/bin/env python

"""
Analyze the difference in SOA of a beacon between two receivers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt

from thrifty import matchmaker
from thrifty import toads_data


SPEED_OF_LIGHT = 2.997e8
SAMPLE_RATE = 2.4e6  # FIXME
S2M = SPEED_OF_LIGHT / SAMPLE_RATE


def plot(soa0, residuals, discontinuities, avg_snr=None):
    s2m = 3e8 / 2.4e6  # FIXME

    avg_snr_db = 10 * np.log10(avg_snr)

    print("residuals: std dev = {:.01f} m; max = {:.01f} m; "
          "avg corr snr = {:.01f}"
          .format(np.std(residuals) * s2m,
                  np.max(np.abs(residuals)) * s2m,
                  avg_snr_db))

    plt.figure(figsize=(11, 6))
    plt.subplot(1, 2, 1)
    plt.plot(soa0, residuals * S2M, '.-')
    plt.title("Residuals")
    plt.xlabel("RX sample")
    plt.ylabel("Residual (samples)")
    plt.grid()
    # plt.ylim([-0.5, 0.5])

    for discontinuity in discontinuities:
        plt.axvline(discontinuity, color='k')

    plt.subplot(1, 2, 2)
    plt.hist(residuals, 20)
    plt.title("Histogram: residuals")
    plt.grid()

    plt.suptitle("Clock sync (stddev = {:.01f} m; max = {:.01f} m; "
                 "avg corr SNR: {:.01f} dB)"
                 .format(np.std(residuals) * s2m,
                         np.max(np.abs(residuals)) * s2m,
                         avg_snr_db))

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)


def analyze(detections, matches, deg=2):
    """
    Parameters
    ---------
    detections: detection array
    matches:    matches of beacon transmissions of the two receivers
    """

    print("Number of detection groups:", len(matches))

    soa = detections['soa'][matches]
    sdoa = soa[:, 1] - soa[:, 0]
    dsdoa = np.diff(sdoa)
    # plt.plot(dsdoa)
    # plt.show()

    is_rapid_change = dsdoa > np.mean(dsdoa) * 10
    discontinuities = np.where(is_rapid_change)[0]

    # print(np.column_stack([np.diff(soa[:,0]), np.diff(soa[:,1]), sdoa[1:],
    #                        dsdoa, is_rapid_change]).astype(int))
    print("Number of discontinuities:", np.size(discontinuities))
    print("Discontinuities (index):", " ".join(map(str, discontinuities)))
    discont_ids = detections[matches]['idx'][discontinuities]
    print("Discontinuities (toad id):", " ".join(map(str, discont_ids)))

    all_soas = []
    all_residuals = []
    all_avg_snr = []
    all_coefs = []
    discontinuity_soas = []

    edges = np.concatenate([[0], discontinuities, [len(matches)]])
    for i in range(len(edges) - 1):
        left, right = edges[i] + 1, edges[i + 1]
        if left + 8 >= right:
            # range too small
            continue

        cut = detections[matches][left:right]
        coef, residuals = fit_poly_model(cut['soa'], deg)
        # outliers = stat_tools.is_outlier(residuals)
        outliers = np.zeros(len(residuals), dtype=np.bool)
        filtered_cut = cut[~outliers]

        energy = filtered_cut['energy']
        noise = filtered_cut['noise']
        avg_snr = np.mean(energy**2 / noise**2)

        all_soas.append(filtered_cut['soa'])
        all_residuals.append(residuals[~outliers])
        all_avg_snr.append(avg_snr)
        all_coefs.append(coef)

        if right != len(matches):
            discontinuity_soas.append(soa[right, 0])
            # print(soa[right, 0], '\t', soa[right, 1], '\t', dsdoa[right])

        print('Cut #{}: {} outliers'.format(i+1, np.sum(outliers)))

    soas = np.concatenate(all_soas)
    residuals = np.concatenate(all_residuals)
    avg_snr = np.mean(all_avg_snr)
    plot(soas[:, 0], residuals, discontinuity_soas, avg_snr)

    return all_coefs


def fit_poly_model(soa, deg=2):
    soa0, soa1 = soa[:, 0], soa[:, 1]
    coef = np.polyfit(soa0, soa1, deg)
    fit = np.poly1d(coef)
    residuals = soa1 - fit(soa0)

    return coef, residuals


def parse_range(string):
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

    parser.add_argument('toads', nargs='?',
                        type=argparse.FileType('rb'), default='data.toads',
                        help="toads data (\"-\" streams from stdin)")
    parser.add_argument('matches', nargs='?',
                        type=argparse.FileType('rb'), default='data.match',
                        help="toads data (\"-\" streams from stdin)")
    parser.add_argument('--beacon', type=int, default=0,
                        help="transmitter ID of beacon")
    parser.add_argument('--rx0', type=int, default=0,
                        help="receiver ID of first receiver")
    parser.add_argument('--rx1', type=int, default=1,
                        help="receiver ID of second receiver")
    parser.add_argument('--range', type=parse_range, default=None,
                        help="limit to a range of detection IDs")
    parser.add_argument('--deg', type=int, default=2,
                        help="degree of polynomial to fit through the SOAs")
    parser.add_argument('--export', type=str, nargs='?',
                        const=True,
                        help="export plot to a .PDF file")

    args = parser.parse_args()

    toads = toads_data.load_toads(args.toads)
    detections = toads_data.toads_array(toads, with_ids=True)

    all_matches = matchmaker.load_matches(args.matches)

    matches = matchmaker.extract_match_matrix(toads, all_matches,
                                              [args.rx0, args.rx1],
                                              [args.beacon])
    if args.range is not None:
        start, stop = args.range
        matches = [m for m in matches
                   if (m[0] >= start and m[0] <= stop and
                       m[1] >= start and m[1] <= stop)]
    matches = np.array(matches)

    analyze(detections, matches, deg=args.deg)

    if args.export:
        if args.export is True:
            filename = 'beacon_analysis_beacon{}_rx{}_rx{}_deg{}'.format(
                       args.beacon, args.rx0, args.rx1, args.deg)
            if args.range is not None:
                filename += '_{}-{}'.format(args.range[0], args.range[1])
            filename += '.pdf'
        else:
            filename = args.export + '.pdf'

        plt.savefig(filename, format='pdf')

    plt.show()

if __name__ == '__main__':
    _main()
