#!/usr/bin/env python

"""
Calculate distance relative beacon using nearest beacon transmission.

python ~/Nagraads/Code/Thrifty/scripts/reldist_nearest.py rx.toads rx-cut.match
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from statsmodels.nonparametric.smoothers_lowess import lowess

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


def find_nearest(array, values):
    indices = np.searchsorted(array, values)
    for i, idx in enumerate(indices):
        value = values[i]
        if idx > 0 and (idx == len(array) or
                        np.abs(value - array[idx-1]) <
                        np.abs(value - array[idx])):
            indices[i] = idx - 1
    return indices


def test_find_nearest():
    nearest = find_nearest([5, 10, 15], [4, 5, 6, 9, 10, 11, 14, 16])
    np.testing.assert_equal(nearest, [0, 0, 0, 1, 1, 1, 2, 2])


def reldist_nearest(tx_soa, beacon_soa):
    nearest_idx = find_nearest(beacon_soa[:, 0], tx_soa[:, 0])
    nearest_soa = beacon_soa[nearest_idx]
    # SoA relative to nearest beacon SoA
    relsoa = tx_soa - nearest_soa
    # Reldist in samples
    reldist = (relsoa[:, 1] - relsoa[:, 0])  # / 2.0
    # print(relsoa[:, 1], relsoa[:, 0])
    return reldist


def reldist_linpol(tx_soa, beacon_soa):
    # Interpolate between two nearest beacon samples
    beacon_rx0, beacon_rx1 = beacon_soa[:, 0], beacon_soa[:, 1]
    tx_rx0, tx_rx1 = tx_soa[:, 0], tx_soa[:, 1]

    high_idx = np.searchsorted(beacon_rx0, tx_rx0)
    low_idx = high_idx - 1
    length = len(beacon_soa[:, 0])
    if high_idx[-1] >= length:
        high_idx[-1] = length - 1
    if low_idx[0] < 0:
        high_idx[0] = 0

    weight = ((tx_rx0 - beacon_rx0[low_idx]) /
              (beacon_rx0[high_idx] - beacon_rx0[low_idx]))
    weight[np.isinf(weight)] = 1  # remove nan
    # Reldist in samples
    reldist = (tx_rx1 - (beacon_rx1[low_idx] * (1-weight) +
                         beacon_rx1[high_idx] * weight))  # / 2.0
    return reldist


def reldist_fit(tx_soa, beacon_soa):
    # TODO: try to "smoothen" beacon curve, interpolate from it
    pass


def plot_freq(x, tx_carrier_freq, beacon_carrier_freq):
    # Freq
    plt.figure()
    hz_per_bin = 2.4e6 / 16384
    plt.plot(x, tx_carrier_freq[:, 0] * hz_per_bin)
    plt.plot(x, tx_carrier_freq[:, 1] * hz_per_bin)
    plt.plot(x, beacon_carrier_freq[:, 0] * hz_per_bin)
    plt.plot(x, beacon_carrier_freq[:, 1] * hz_per_bin)
    plt.ylabel('Freq offset (kHz)')
    plt.xlabel('TX timestamp at RX0 (s)')
    plt.grid()
    plt.tight_layout()
    # plt.show()


def plot_doppler(x, tx_carrier_freq, beacon_carrier_freq):
    # Doppler
    plt.figure()
    
    dop1 = (tx_carrier_freq[:, 0] - tx_carrier_freq[:, 1])
    dop2 = (beacon_carrier_freq[:, 0] -
            beacon_carrier_freq[:, 1])
    dop_bin = (dop1 - dop2) / 2.0
    # 50 km/h doppler = 433e6 * ((3e8 + 50/3.6) / 3e8 - 1)
    #                 = 50/3.6 / 3e8 * 433e6
    hz_per_bin = 2.4e6 / 16384
    dop = dop_bin * hz_per_bin * 3e8 / 433e6 * 3.6
    dop_outliers = is_outlier(dop)

    dopx = x[~dop_outliers]

    dop_smooth = lowess(dop[~dop_outliers], dopx,
                        is_sorted=True, frac=0.025, it=0)

    plt.plot(dopx, dop[~dop_outliers], 'r', linewidth=0.2, alpha=0.5)
    plt.plot(dop_smooth[:, 0], dop_smooth[:, 1], 'b')
    plt.ylabel('Doppler shift (km/h)')
    plt.xlabel('TX timestamp at RX0 (s)')
    plt.grid()
    plt.tight_layout()
    # plt.show()


def reldist(detections, matches, tx_id, beacon_id, method='linpol'):
    # We only support 2xRX at the moment
    # filter matches
    fig_prefix = 'reldist_' + method
    num_rx = 2
    matches = [x for x in matches if sum([y != -1 for y in x]) == num_rx]
    tx_matches = np.array([x for x in matches
                           if detections['txid'][x[0]] == tx_id])
    beacon_matches = np.array([x for x in matches
                               if detections['txid'][x[0]] == beacon_id])
    tx_soa = detections['soa'][tx_matches]
    beacon_soa = detections['soa'][beacon_matches]

    tx_timestamp_full = detections['timestamp'][tx_matches]
    tx_timestamp_full -= tx_timestamp_full[0, 0]  # relative to first detection
    # FIXME: we assume detections and matches are sorted by timestamp

    if method == 'nearest':
        reldist_full = reldist_nearest(tx_soa, beacon_soa)
    elif method == 'linpol':
        reldist_full = reldist_linpol(tx_soa, beacon_soa)

    # lithium mean sample rate rel to sys clock: 2399988.41
    # silicis mean sample rate rel to sys clock: 2400064.84
    # TODO: sync s2m with beacon...
    s2m = 2.997e8 / 2.4e6  # FIXME: use "real sample rate" based on clock sync
    # TODO: calibrate s2m using measurement

    d_beacon = -917.888911019 / s2m    # distance(Tx, Si) - distance(Tx, Li)
    dist_rx = 8964.1335967 / s2m       # distance between RX1 and RX0
    dist_beacon = 4024.48549105 / s2m  # distance from RX1 (ref) to beacon
    
    reldist_full = (reldist_full + d_beacon + dist_rx) / 2.0 - dist_beacon
    # reldist_full /= 2.0
    # print(reldist_full[0])

    outliers = is_outlier(reldist_full)
    print("Number of outliers: {} / {}"
          .format(np.sum(outliers), len(outliers)))

    reldist = reldist_full[~outliers]
    tx_timestamp = tx_timestamp_full[~outliers][:, 1]

    print('mean =', np.mean(reldist)*s2m)
    print('std =', np.std(reldist)*s2m)

    plt.plot(tx_timestamp, reldist * s2m)
    # plt.plot(reldist * s2m, '.-')
    plt.ylabel('TX position relative to beacon (m)')
    plt.xlabel('TX timestamp at RX0 (s)')
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_prefix + '.pdf', format='pdf')
    # plt.show()

    tx_soa_rx1 = tx_soa[~outliers][:, 1]
    diff_all = np.diff(reldist) / np.diff(tx_soa_rx1) * s2m * 2.4e6
    # Alternative: diff2 = np.diff(reldist * s2m) / np.diff(tx_timestamp)

    diff_outliers = is_outlier(diff_all)
    diff = diff_all[~diff_outliers]
    diffx = tx_timestamp[1:][~diff_outliers]

    # Smoothing using savgol:
    # d_reldist = scipy.signal.savgol_filter(reldist, 31, 3, 1)
    # diff3 = d_reldist[1:] / np.diff(tx_soa_rx1) * s2m * 2.4e6

    # Smoothing using lowess
    diff_lowess = lowess(diff, diffx, is_sorted=True, frac=0.025, it=0)
    diff_smooth_x, diff_smooth = diff_lowess[:, 0], diff_lowess[:, 1]

    plt.figure()
    plt.plot(diffx, diff * 3.6, 'r', linewidth=0.2, alpha=0.5)
    plt.plot(diff_smooth_x, diff_smooth * 3.6, 'b')
    plt.ylabel('d(reldist)/d(timestamp) (km/h)')
    plt.xlabel('TX timestamp at RX0 (s)')
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_prefix + '_diff.pdf', format='pdf')
    # plt.show()

    # Doppler
    nearest_idx = find_nearest(beacon_soa[:, 0], tx_soa[:, 0])
    tx_carrier_freq = (detections['carrier_bin'][tx_matches] +
                       detections['carrier_offset'][tx_matches])
    beacon_carrier_freq = (detections['carrier_bin'][beacon_matches] +
                           detections['carrier_offset'][beacon_matches])
    beacon_carrier_freq = beacon_carrier_freq[nearest_idx]

    plot_freq(tx_timestamp_full[:, 0], tx_carrier_freq, beacon_carrier_freq)
    plt.savefig(fig_prefix + '_freq.pdf', format='pdf')
    plot_doppler(tx_timestamp_full[:, 0], tx_carrier_freq, beacon_carrier_freq)
    plt.savefig(fig_prefix + '_doppler.pdf', format='pdf')

    if method == 'nearest':
        # # Without beacon SoA interpolation
        cuts = [(200, 349),
                (665, 835),
                (960, 1160),
                (1380, 1575),
                (1680, 1870),
                (2010, 2095),
                (2300, 2640)]
    elif method == 'linpol':
        # With linear interpolation between subsequent beacon SOA
        cuts = [(50, 230),
                (655, 810),
                (950, 1150),
                (1350, 1555),
                (1665, 1850),
                (2000, 2080),
                (2290, 2610)]

    for i, range_ in enumerate(cuts):
        start, stop = range_
        cut = reldist[start:stop+1]
        timestamps = tx_timestamp[start:stop+1]

        cut_outliers = is_outlier(cut)
        cut = cut[~cut_outliers]
        timestamps = timestamps[~cut_outliers]

        std = np.std(cut)
        mean = np.mean(cut)

        # number of sample within one std dev
        num1std = np.count_nonzero((cut >= mean - std) & (cut <= mean + std))
        num1std_per = num1std / len(cut) * 100

        title = ("Cut #{}: {}-{} ({} samples, {} out): "
                 "mean = {:.1f}; std = {:.2f} ({:.0f}%)"
                 .format(i + 1, start, stop, len(cut), np.sum(cut_outliers),
                         mean*s2m, std*s2m, num1std_per))
        print(title)

        plt.figure(figsize=(11, 6))
        plt.suptitle(title)
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

        plt.subplot(gs[0])
        plt.plot(timestamps, cut * s2m, '.-')
        plt.ylabel('TX position relative to beacon (m)')
        plt.xlabel('TX timestamp at RX0 (s)')
        plt.grid()

        plt.subplot(gs[1])
        plt.hist(cut * s2m, 20)
        plt.title("Histogram")

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig(fig_prefix + '-cut{}.pdf'.format(i + 1), format='pdf')
        # plt.show()

    plt.show()


def _main():
    import argparse

    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)

    parser.add_argument('toads', nargs='?',
                        type=argparse.FileType('rb'), default='rx.toads',
                        help="toads data (\"-\" streams from stdin)")
    parser.add_argument('matches', nargs='?',
                        type=argparse.FileType('rb'), default='rx.match',
                        help="toads data (\"-\" streams from stdin)")
    args = parser.parse_args()

    toads = toads_data.load_toads(args.toads)
    detections = toads_data.toads_array(toads, with_ids=True)
    matches = matchmaker.load_matches(args.matches)
    reldist(detections, matches, 1, 0)


if __name__ == '__main__':
    _main()
