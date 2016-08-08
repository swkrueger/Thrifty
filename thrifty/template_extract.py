"""
Extract a template from captured data.

Usage example:

    Capture data:
    $ thrifty capture rx1.card

    Generate a "base template" to be able to extract the code signal:
    $ thrifty template_generate 10 0 -o theoretical-template.npy

    Extract the new tamplte:
    $ thrifty template_extract rx1.card --template=theoretical-template.npy \\
                                        -o captured-template.npy

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

import numpy as np

from thrifty import detect
from thrifty import settings
from thrifty.block_data import card_reader
from thrifty.setting_parsers import normalize_freq_range


MAX_OFFSET = 0.1


def best_detection(detections, max_offset):
    """Get block with largest corr peak and offset less than `max_offset`."""
    best_result = None
    best_fft = None

    for detected, result, fft, _ in detections:
        if detected and abs(result.corr_info.offset) <= max_offset:
            if (best_result is None or
                    result.corr_info.energy > best_result.corr_info.energy):
                best_result = result
                best_fft = fft

    best_signal = np.fft.ifft(best_fft)
    return best_signal, best_result


def extract_template(signal, result, template_len):
    """Extract template from detection with OOK signal."""
    start = result.corr_info.sample
    cut = np.abs(signal[start:start+template_len])
    cut *= 2 / (np.mean(cut) + np.std(cut))
    cut = cut - np.mean(cut)  # OOK -> bipolar signal
    return cut


def plot(signal, template, offset):
    """Plot the newly captured template and the base template."""
    import matplotlib.pyplot as plt

    xdata = np.arange(len(template))
    plt.plot(xdata, signal, '.-')
    plt.plot(xdata - offset, template, '.-')
    plt.savefig('extract.pdf', format='pdf')
    plt.show()


def _main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('input',
                        type=argparse.FileType('rb'), default='-',
                        help="input data ('-' streams from stdin)")
    parser.add_argument('-o', '--output', type=argparse.FileType('wb'),
                        default='capture.npy', help="Output file (.npy)")
    parser.add_argument('-p', '--plot', action='store_true',
                        help="Plot base template and extracted template.")

    setting_keys = ['sample_rate', 'block_size', 'block_history',
                    'carrier_window', 'carrier_threshold',
                    'corr_threshold', 'template']
    config, args = settings.load_args(parser, setting_keys)

    bin_freq = config.sample_rate / config.block_size
    window = normalize_freq_range(config.carrier_window, bin_freq)

    blocks = card_reader(args.input)
    template = np.load(config.template)

    detections = detect.detect(blocks=blocks,
                               block_len=config.block_size,
                               history_len=config.block_history,
                               carrier_len=len(template),
                               carrier_thresh=config.carrier_threshold,
                               carrier_window=window,
                               template=template,
                               corr_thresh=config.corr_threshold,
                               yield_data=True)

    full_signal, result = best_detection(detections, MAX_OFFSET)
    signal = extract_template(full_signal, result, len(template))

    np.save(args.output, signal)
    print("Captured template from block #{} (timestamp: {:.6f}): "
          "offset={:+.3f}; corr_ampl={}".format(result.block,
                                                result.timestamp,
                                                result.corr_info.offset,
                                                result.corr_info.energy))
    if args.plot:
        plot(signal, template, result.corr_info.offset)


if __name__ == '__main__':
    _main()
