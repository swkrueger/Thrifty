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


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Copy pasta from https://stackoverflow.com/questions/11882393/

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


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
    plt.ylim([-0.5, 0.5])

    for discontinuity in discontinuities:
        # print(discontinuity)
        plt.axvline(discontinuity, color='k')
    # TODO: plot discontinuities

    plt.subplot(1, 2, 2)
    plt.hist(residuals, 20)
    plt.title("Histogram: residuals")
    plt.grid()

    plt.suptitle("Clock sync (stddev = {:.01f} m; max = {:.01f} m; "
                 "avg corr SNR: {:.01f} dB)"
                 .format(np.std(residuals) * s2m,
                         np.max(np.abs(residuals)) * s2m,
                         avg_snr))

    plt.tight_layout()

    plt.subplots_adjust(top=0.90)
    plt.savefig('clocksync.pdf', format='pdf')
    plt.savefig('clocksync.png', format='png')


def clock_sync(toads, all_matches, rx0, rx1, beacon):
    """
    Parameters
    ---------
    detections: detection array
    matches:    match array
    beacon:     beacon ID
    """

    # TODO: refactor function prototype

    matches = matchmaker.extract_match_matrix(toads, all_matches,
                                              [rx0, rx1], [beacon])
    matches = np.array(matches)

    # Extract beacon's matches
    # np.array([x for x in matches
    #                     if sum([y != -1 for y in x]) == num_rx
    #                     and detections['txid'][x[0]] == beacon])
    print("Number of detection groups:", len(matches))

    detections = toads_data.toads_array(toads, with_ids=True)

    # rxid = detections['rxid'][matches]
    # for a, b in rxid:
    #     print(a, b)
    # sys.exit(0)

    soa = detections['soa'][matches]
    sdoa = soa[:, 1] - soa[:, 0]
    dsdoa = np.abs(np.diff(sdoa))
    dsdoa2 = np.diff(sdoa)
    # plt.plot(dsdoa)
    # plt.show()
    is_rapid_change = dsdoa > np.mean(dsdoa) * 10
    discontinuities = np.where(is_rapid_change)[0]
    # for i, x in enumerate(dsdoa):
    #     if x > np.mean(dsdoa) * 10:
    #         a, b = matches[i + 1], matches[i]
    #         print(x, a, b)
    #         print('-', sdoa[i + 1], sdoa[i], detections['soa'][a], detections['soa'][b])
    #         print('+', detections['timestamp'][a] - detections['timestamp'][b])
    # timestamps = detections['timestamp'][matches]
    # asdf = np.diff(timestamps[:, 1] - timestamps[:, 0])
    # plt.plot(asdf)
    # plt.show()

    # print np.column_stack([np.diff(soa[:,0]), np.diff(soa[:,1]), sdoa[1:],
    #                        dsdoa, rapid_change]).astype(int)
    print("Number of discontinuities:", np.size(discontinuities))
    print("Discontinuities:", " ".join(map(str, discontinuities)))

    all_soa1 = []
    all_residuals = []
    all_avg_snr = []
    all_coefs = []
    discontinuity_soas = []

    edges = np.concatenate([[0], discontinuities, [len(matches)]])
    for i in range(len(edges) - 1):
        left, right = edges[i] + 1, edges[i + 1]
        if left + 8 >= right:
            continue

        soa_ab = soa[left:right]
        coef, residuals = sync(soa_ab)

        offset = detections['soa'][matches][left:right]
        # print(np.diff(offset[:, 1] - offset[:, 0]))

        outliers = is_outlier(residuals)

        energy = detections['energy'][matches][left:right][~outliers]
        noise = detections['noise'][matches][left:right][~outliers]
        snr = 20 * np.log10(energy / noise)
        avg_snr = np.mean(snr)

        all_soa1.append(soa_ab[:, 0][~outliers])
        all_residuals.append(residuals[~outliers])
        all_avg_snr.append(avg_snr)
        all_coefs.append(coef)

        if right != len(matches):
            discontinuity_soas.append(soa[right, 0])
            print(soa[right, 0], '\t', soa[right, 1], '\t', dsdoa2[right])

    soa1 = np.concatenate(all_soa1)
    residuals = np.concatenate(all_residuals)
    avg_snr = np.mean(all_avg_snr)
    plot(soa1, residuals, discontinuity_soas, avg_snr)

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
                        type=argparse.FileType('rb'), default='data.toads',
                        help="toads data (\"-\" streams from stdin)")
    parser.add_argument('matches', nargs='?',
                        type=argparse.FileType('rb'), default='data.match',
                        help="toads data (\"-\" streams from stdin)")
    parser.add_argument('--id', dest='beacon_id', type=int, default=0,
                        help="transmitter ID of beacon")

    args = parser.parse_args()

    toads = toads_data.load_toads(args.toads)
    matches = matchmaker.load_matches(args.matches)

    clock_sync(toads, matches, 0, 1, args.beacon_id)
    clock_sync(toads, matches, 0, 2, args.beacon_id)
    clock_sync(toads, matches, 1, 2, args.beacon_id)
    # reldist(coefs, detections, matches)
    plt.show()

if __name__ == '__main__':
    _main()
