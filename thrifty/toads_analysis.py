"""Calculate stats on data in .toads file.'"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from thrifty import toads_data
from thrifty import util
from thrifty import matchmaker


def split_by_column(detections, columns):
    if len(columns) == 0:
        return detections
    splits = OrderedDict()
    column = columns.pop(0)
    for field in np.unique(detections[column]):
        field_data = detections[detections[column] == field]
        splits[field] = split_by_column(field_data, columns[:])
    return splits


def split_rxtx(detections):
    """Split detections by (RX ID, TX ID)."""
    return split_by_column(detections, ['rxid', 'txid'])


def print_stats(data):
    """Print mean, standard deviation, min and max of data in toads file."""

    print("Number of detections: {}".format(len(data)))

    peak = data['carrier_energy']
    print("Carrier peak: mean={:.1f}, std={:.2f}, min={:.1f}, max={:.1f}"
          .format(np.mean(peak), np.std(peak), np.min(peak), np.max(peak)))

    noise = data['carrier_noise']
    print("Carrier noise: mean={:.1f}, std={:.2f}, min={:.1f}, max={:.1f}"
          .format(np.mean(noise), np.std(noise), np.min(noise), np.max(noise)))

    snr = util.snr(peak, noise)
    print("Carrier SNR (dB): mean={:.1f}, std={:.2f}, min={:.1f}, max={:.1f}"
          .format(np.mean(snr), np.std(snr), np.min(snr), np.max(snr)))

    freq = data['carrier_bin']
    print("Carrier bin: mean={:.0f}, std={:.3f}, min={:.0f}, max={:.0f}"
          .format(np.mean(freq), np.std(freq), np.min(freq), np.max(freq)))

    offset = data['carrier_offset']
    print("Carrier offset: mean={:.3f}, std={:.3f}, min={:.3f}, max={:.3f}"
          .format(np.mean(offset), np.std(offset),
                  np.min(offset), np.max(offset)))

    peak = data['energy']
    print("Corr peak: mean={:.1f}, std={:.2f}, min={:.1f}, max={:.1f}"
          .format(np.mean(peak), np.std(peak), np.min(peak), np.max(peak)))

    noise = data['noise']
    print("Corr noise: mean={:.1f}, std={:.2f}, min={:.1f}, max={:.1f}"
          .format(np.mean(noise), np.std(noise), np.min(noise), np.max(noise)))

    snr = util.snr(peak, noise)
    print("Corr SNR (dB): mean={:.1f}, std={:.2f}, min={:.1f}, max={:.1f}"
          .format(np.mean(snr), np.std(snr), np.min(snr), np.max(snr)))

    offset = data['offset']
    print("Corr offset: mean={:.3f}, std={:.3f}, min={:.3f}, max={:.3f}"
          .format(np.mean(offset), np.std(offset),
                  np.min(offset), np.max(offset)))


def print_rxtx_stats(splits):
    for rxid, rx_data in splits.iteritems():
        for txid, tx_detections in rx_data.iteritems():
            print("# Stats for RX #{}'s detections of "
                  "TX #{}'s transmissions:\n"
                  .format(rxid, txid))
            print_stats(tx_detections)
            print("\n")


def _plot_column(ax, detections, column, **kwargs):
    if isinstance(column, list):
        for c in column:
            _plot_column(ax, detections, c, **kwargs)
        ylabel = ', '.join(column)
        ax.set_ylabel(ylabel)
        return

    if column == 'freqs':
        y_data = detections['carrier_bin'] + detections['carrier_offset']
    elif column == 'snr':
        y_data = 20 * np.log10(detections['energy'] / detections['noise'])
    elif column == 'corr_snr':
        y_data = 20 * np.log10(detections['carrier_energy'] /
                               detections['carrier_noise'])
    else:
        y_data = detections[column]
    ax.plot(detections['timestamp'], y_data,
            marker='.', **kwargs)
    ax.set_xlabel('Timestamp')
    ax.set_ylabel(column)


def _plot_per_rx(splits, func):
    fig = plt.figure()
    sharex = None
    for idx, (rxid, rx_data) in enumerate(splits.iteritems()):
        if sharex is not None:
            ax = fig.add_subplot(len(splits), 1, idx+1, sharex=sharex)
        else:
            ax = fig.add_subplot(len(splits), 1, idx+1)
            sharex = ax
        for txid, tx_detections in rx_data.iteritems():
            func(ax, tx_detections, rxid, txid)
        ax.set_title('RX {}'.format(rxid))
        ax.legend()
        ax.grid()
    return fig


def _plot_column_per_rx(splits, column, **kwargs):
    def plot(ax, tx_detections, rxid, txid):
        label = 'TX {}'.format(txid)
        _plot_column(ax, tx_detections, column, label=label, **kwargs)
    fig = _plot_per_rx(splits, plot)
    fig.suptitle(column)


def plot_rxtx_matrix(fig, splits, func):
    rxids = list(splits.keys())
    txids = list(np.unique(np.concatenate([tx.keys() for tx in splits.values()])))
    print(txids)
    shareax = None
    for rxid, rx_data in splits.iteritems():
        for txid, tx_detections in rx_data.iteritems():
            if len(tx_detections) == 0:
                continue
            rxidx = rxids.index(rxid)
            txidx = txids.index(txid)

            args = {}
            if shareax is not None:
                args['sharex'] = shareax
                args['sharey'] = shareax
            ax = fig.add_subplot(len(rxids), len(txids),
                                 rxidx*len(txids) + txidx + 1, **args)
            if shareax is None:
                shareax = ax
            func(ax, tx_detections)
            ax.set_title("RX {} TX {}".format(rxid, txid))
            ax.grid()
    return fig


def plot_minute_histogram(splits):
    def plot(ax, detections):
        bins = np.floor_divide(detections['timestamp'], 60).astype('int64')
        counts = np.bincount(bins)
        ax.plot(counts)
        ax.set_xlabel('Minute')
        ax.set_ylabel('Count')
    fig = plt.figure()
    plot_rxtx_matrix(fig, splits, plot)
    fig.suptitle('Histogram of number of detections per minute')


def plot_column_matrix(splits, column, **kwargs):
    def plot(ax, detections):
        _plot_column(ax, detections, column, **kwargs)
    fig = plt.figure()
    plot_rxtx_matrix(fig, splits, plot)

    if isinstance(column, list):
        title = ', '.join(column)
    else:
        title = column

    fig.suptitle('Matrix: ' + title)


def plot_column_histogram_matrix(splits, column, bins=10):
    def plot(ax, detections):
        ax.hist(detections[column], bins)
        ax.set_xlabel('Count')
        ax.set_ylabel('Bin ({})'.format(column))

    fig = plt.figure()
    plot_rxtx_matrix(fig, splits, plot)
    fig.suptitle('Histograms: ' + column)


def plot_offset_hist2d(splits):
    def plot(ax, detections):
        # each timestamp bin = 2.5 minutes
        tbins = (np.max(detections['timestamp']) -
                 np.min(detections['timestamp'])) / 150
        tbins = max(tbins, 1)

        ax.hist2d(detections['offset'], detections['timestamp'], [10, tbins])
        ax.set_xlabel('Time')
        ax.set_ylabel('Offset')
        # plt.colorbar()

    fig = plt.figure()
    plot_rxtx_matrix(fig, splits, plot)
    fig.suptitle('Corr offset vs. time histograms: ')


def plot_carrier_histogram(splits):
    def plot(ax, detections, rxid, txid):
        first_bin = np.amin(detections['carrier_bin'])
        cnts = np.bincount(detections['carrier_bin'] - first_bin)
        label = 'TX {}'.format(txid)
        ax.plot(np.arange(len(cnts)) + first_bin, cnts, '.-', label=label)

    fig = _plot_per_rx(splits, plot)
    fig.suptitle('Carrier bin histogram')


def plot_timestamp_residuals(detections):
    groups = split_by_column(detections, ['rxid'])
    num_rxids = len(groups)
    fig = plt.figure()
    for i, (rxid, data) in enumerate(groups.iteritems()):
        ax = fig.add_subplot(num_rxids, 1, i+1)
        coeffs = np.polyfit(data['soa'], data['timestamp'], 1)
        poly = np.poly1d(coeffs)
        res = poly(data['soa']) - data['timestamp']
        ax.plot(data['soa'], res, '.-')
        ax.set_xlabel('SOA')
        ax.set_ylabel('Residual')
        ax.set_title('RX {}'.format(rxid))
    fig.suptitle('Timestamp residuals for linear fit')
    return fig


def plot_all(detections, splits):
    _plot_column_per_rx(splits, 'freqs')
    _plot_column_per_rx(splits, 'energy')
    _plot_column_per_rx(splits, 'carrier_energy')
    _plot_column_per_rx(splits, 'noise')
    # _plot_column_per_rx(splits, 'snr')
    plot_minute_histogram(splits)
    plot_column_matrix(splits, ['energy', 'carrier_energy'])
    plot_carrier_histogram(splits)
    plot_timestamp_residuals(detections)
    plot_column_histogram_matrix(splits, 'offset')
    plot_offset_hist2d(splits)
    plot_column_matrix(splits, 'offset', linewidth=0.1)


def _main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--toad', dest='toad', action='store_true',
                        help="input data is .toad data instead of .toads")
    parser.add_argument('-i', '--input',
                        type=argparse.FileType('rb'), default='data.toads',
                        help=".toads data (\'-\' streams from stdin)")
    parser.add_argument('-m', '--match',
                        type=argparse.FileType('rb'), default=None,
                        help="exclude unmatched detections")
    # TODO: --filter, e.g. txid == 2
    # TODO: take plot/print commands as arguments, e.g. cmd1 [cmd2,...]
    # TODO: tabbed Qt interface (like detect_analysis)
    args = parser.parse_args()

    if args.toad:
        toads = toads_data.load_toad(args.input)
        detections = toads_data.toads_array(toads, with_ids=False)
    else:
        toads = toads_data.load_toads(args.input)
        detections = toads_data.toads_array(toads, with_ids=True)

    if args.match:
        matches = matchmaker.load_matches(args.match)
        matched_ids = np.sort(np.concatenate(matches))
        detections = detections[matched_ids]

    time0 = np.min(detections['timestamp'])
    print('Timestamps relative to {:.6f}'.format(time0))
    detections['timestamp'] -= time0

    splits = split_rxtx(detections)

    print_rxtx_stats(splits)
    plot_all(detections, splits)
    plt.show()

    # TODO:
    # - plot per tx: freqs, corr_peaks, carrier_peaks, noise, snr
    # - print count / mean amplitude / mean snr matrix


if __name__ == '__main__':
    _main()
