#!/usr/bin/env python

"""
Build clock sync model from beacon transmissions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt

from thrifty import matchmaker
from thrifty import toads_data


def plot(soa1, residuals, discontinuities, avg_snr=None):
    s2m = 3e8 / 2.4e6  # FIXME

    print("residuals: std dev = {:.01f} m; max = {:.01f} m; "
          "avg corr snr = {:.01f}"
          .format(np.std(residuals) * s2m,
                  np.max(np.abs(residuals)) * s2m,
                  avg_snr))

    plt.figure(figsize=(11, 6))
    plt.subplot(1, 2, 1)
    plt.plot(soa1, residuals, '.-')
    plt.title("Residuals")
    plt.xlabel("RX sample")
    plt.ylabel("Residual (samples)")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.hist(residuals, 20)
    plt.title("Histogram: residuals")
    plt.grid()

    # TODO: plot discontinuities

    plt.suptitle("Clock sync (stddev = {:.01f} m; max = {:.01f} m; "
                 "avg corr SNR: {:.01f} dB)"
                 .format(np.std(residuals) * s2m,
                         np.max(np.abs(residuals)) * s2m,
                         avg_snr))

    plt.tight_layout()

    plt.subplots_adjust(top=0.90)
    plt.savefig('clocksync.pdf', format='pdf')
    plt.savefig('clocksync.png', format='png')
    plt.show()


def clock_sync(detections, matches, beacon):
    """
    Parameters
    ---------
    detections: detection array
    matches:    match array
    beacon:     beacon ID
    """

    if len(matches) == 0:
        return
    num_rx = len(matches[0])

    # Extract beacon's matches
    # Incomplete matches aren't supported at the moment
    matches = np.array([x for x in matches
                        if sum([y != -1 for y in x]) == num_rx
                        and detections['txid'][x[0]] == beacon])
    print("Number of detection groups:", len(matches))

    soa = detections['soa'][matches]
    sdoa = soa[:, 1] - soa[:, 0]
    dsdoa = np.abs(np.diff(sdoa))
    is_rapid_change = dsdoa > np.mean(dsdoa) * 10
    discontinuities = np.where(is_rapid_change)[0]

    # print np.column_stack([np.diff(soa[:,0]), np.diff(soa[:,1]), sdoa[1:],
    #                        dsdoa, rapid_change]).astype(int)
    print("Number of discontinuities:", np.size(discontinuities))
    print("Discontinuities:", " ".join(map(str, discontinuities)))

    all_soa1 = []
    all_residuals = []
    all_avg_snr = []
    all_coefs = []

    edges = np.concatenate([[0], discontinuities, [len(matches)]])
    for i in range(len(edges) - 1):
        left, right = edges[i] + 1, edges[i + 1]
        if left + 3 >= right:
            continue

        soa_ab = soa[left:right]
        coef, residuals = sync(soa_ab)

        energy = detections['energy'][matches][left:right]
        noise = detections['noise'][matches][left:right]
        snr = 20 * np.log10(energy / noise)
        avg_snr = np.mean(snr)

        all_soa1.append(soa_ab[:, 0])
        all_residuals.append(residuals)
        all_avg_snr.append(avg_snr)
        all_coefs.append(coef)

    soa1 = np.concatenate(all_soa1)
    residuals = np.concatenate(all_residuals)
    avg_snr = np.mean(all_avg_snr)
    plot(soa1, residuals, discontinuities, avg_snr)

    return all_coefs


def sync(soa, deg=2):
    soa1, soa2 = soa[:, 0], soa[:, 1]
    coef = np.polyfit(soa1, soa2, deg)
    fit = np.poly1d(coef)
    residuals = soa2 - fit(soa1)

    return coef, residuals


# def reldist(coefs, detections, matches):
#     coef = coefs[0]
#     matches = np.array([x for x in matches
#                         if sum([y != -1 for y in x]) == 2
#                         and detections['txid'][x[0]] == 1])
#     soa = detections['soa'][matches]
#     soa1, soa2 = soa[:, 0], soa[:, 1]
#     sdoa = soa2 - soa1
#     sync_func = np.poly1d(coef)
#     ref = sync_func(soa1)
#     relsoa = soa2 - ref
#     np.set_printoptions(threshold=np.inf)
#     print(np.column_stack([sdoa, ref, relsoa]))
# 
#     plt.plot(relsoa)
#     plt.savefig('reldist.pdf', format='pdf')
#     plt.savefig('reldist.png', format='png')
#     plt.show()


def _main():
    import argparse

    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)

    # TODO: default values for toads and matches:
    #  - zero parameters: rx.toads; rx.match
    #  - if one parameter: append ".toads" and ".match"
    #  - if two parameters: use the files being specified
    parser.add_argument('toads', nargs='?',
                        type=argparse.FileType('rb'), default='rx.toads',
                        help="toads data (\"-\" streams from stdin)")
    parser.add_argument('matches', nargs='?',
                        type=argparse.FileType('rb'), default='rx.match',
                        help="toads data (\"-\" streams from stdin)")
    parser.add_argument('--id', dest='beacon_id', type=int, default=0,
                        help="transmitter ID of beacon")

    args = parser.parse_args()

    toads = toads_data.load_toads(args.toads)
    detections = toads_data.toads_array(toads, with_ids=True)
    matches = matchmaker.load_matches(args.matches)

    clock_sync(detections, matches, args.beacon_id)
    # reldist(coefs, detections, matches)

if __name__ == '__main__':
    _main()
