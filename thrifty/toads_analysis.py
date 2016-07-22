"""Calculate stats on data in .toads file.'"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

import numpy as np

from thrifty import toads_data
from thrifty import util


def split_rxtx(detections):
    """Split detections by (RX ID, TX ID)."""
    for rxid in np.unique(detections['rxid']):
        rx_detections = detections[detections['rxid'] == rxid]
        for txid in np.unique(rx_detections['txid']):
            data = rx_detections[rx_detections['txid'] == txid]
            yield rxid, txid, data


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


def _main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('input', nargs='?',
                        type=argparse.FileType('rb'), default='rx.toads',
                        help=".toads data (\'-\' streams from stdin)")
    args = parser.parse_args()
    toads = toads_data.load_toads(args.input)
    detections = toads_data.toads_array(toads, with_ids=True)

    for rxid, txid, data in split_rxtx(detections):
        print("# Stats for RX #{}'s detections of TX #{}'s transmissions:\n"
              .format(txid, rxid))
        print_stats(data)
        print("\n")


if __name__ == '__main__':
    _main()
