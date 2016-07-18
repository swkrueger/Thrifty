"""Detect presence of positioning signals and estimate sample-of-arrival."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import sys

import numpy as np

from thrifty import settings
from thrifty import toads_data
from thrifty import util
from thrifty.block_data import block_reader, card_reader
from thrifty.carrier_sync import make_syncer
from thrifty.setting_parsers import normalize_freq_range
from thrifty.soa_estimator import make_soa_estimator


def detect(blocks, block_len, history_len,
           carrier_len, carrier_thresh, carrier_window,
           template, corr_thresh):
    """Detect positioning signals and estimate sample-of-arrival.

    All-in-one signal detection and sample-of-arrival estimation. Find carrier,
    synchronise to carrier, correlate with template, and estimate SoA.
    """
    # pylint: disable=too-many-locals,too-many-arguments

    sync = make_syncer(thresh_coeffs=carrier_thresh,
                       window=carrier_window,
                       block_len=block_len,
                       carrier_len=carrier_len)
    soa_estimate = make_soa_estimator(template=template,
                                      thresh_coeffs=corr_thresh,
                                      block_len=block_len,
                                      history_len=history_len)
    new_len = block_len - history_len

    for timestamp, block_idx, block in blocks:
        assert len(block) == block_len
        fft = np.fft.fft(block)
        shifted_fft, carrier_info = sync(fft)

        if shifted_fft is not None:  # detected
            detected, corr_info, _ = soa_estimate(shifted_fft)
            soa = new_len * block_idx + corr_info.sample + corr_info.offset
        else:
            detected, corr_info, soa = False, None, None

        result = toads_data.DetectionResult(timestamp, block_idx, soa,
                                            carrier_info, corr_info)
        yield detected, result


def _carrier_freq(carrier_info, block_len, sample_rate):
    """Convert carrier bin and offset to frequency value in Hertz."""
    bin_freq = sample_rate / block_len
    idx = util.fft_bin(carrier_info.bin, block_len)
    pos = idx + carrier_info.offset
    freq = pos * bin_freq
    return freq


def _summary_line(detected, result, prev_soa, sample_rate, block_len):
    """Summarize detection results."""

    carrier_detect = (result.corr_info is not None)
    carrier_freq = _carrier_freq(result.carrier_info, block_len, sample_rate)
    snr = util.snr(result.carrier_info.energy, result.carrier_info.noise)
    info = ("blk={blk}; carrier: {det} @ {freq:.3f} kHz, "
            "SNR = {ampl:>4.0f} / {noise:>2.0f} = {snr:>5.2f} dB"
            .format(blk=result.block,
                    det="yes" if carrier_detect else "no ",
                    freq=carrier_freq / 1e3,
                    ampl=result.carrier_info.energy,
                    noise=result.carrier_info.noise,
                    snr=snr))

    if carrier_detect:
        time_diff = (result.soa - prev_soa) / sample_rate
        snr = util.snr(result.corr_info.energy, result.corr_info.noise)
        info += ("; corr: {det} @ {idx:>4}{offset:+.3f} (+{dt:.1f}s)"
                 ", SNR = {ampl:>4.0f}/{noise:>2.0f} = {snr:>5.2f} dB"
                 .format(det="yes" if detected else "no ",
                         idx=result.corr_info.sample,
                         offset=result.corr_info.offset,
                         dt=time_diff,
                         ampl=result.corr_info.energy,
                         noise=result.corr_info.noise,
                         snr=snr))

    return info


def _main():
    # pylint: disable=too-many-locals
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('input',
                        type=argparse.FileType('rb'), default='-',
                        help="input data ('-' streams from stdin)")
    parser.add_argument('--raw', dest='raw', action='store_true',
                        help="input data is raw binary data")
    parser.add_argument('--quiet', dest='quiet', action='store_true',
                        help="do not write anything to standard output")
    parser.add_argument('-o', '--output', dest='output',
                        type=argparse.FileType('wb'), default='-',
                        help="Output file (.toad) ('-' for stdout)")

    setting_keys = ['sample_rate', 'block_size', 'block_history',
                    'carrier_window', 'carrier_threshold',
                    'corr_threshold', 'template']
    config, args = settings.load_args(parser, setting_keys)

    quiet = args['quiet']
    info_out = sys.stderr if args['output'] == sys.stdout else sys.stdout

    bin_freq = config['sample_rate'] / config['block_size']
    window = normalize_freq_range(config['carrier_window'], bin_freq)
    block_len = config['block_size']
    history_len = config['block_history']

    if args['raw']:
        blocks = block_reader(args['input'], block_len, history_len)
    else:
        blocks = card_reader(args['input'])

    template = np.load(config['template'])

    detections = detect(blocks=blocks,
                        block_len=block_len,
                        history_len=history_len,
                        carrier_len=len(template),
                        carrier_thresh=config['carrier_threshold'],
                        carrier_window=window,
                        template=template,
                        corr_thresh=config['corr_threshold'])

    # Store previous SoAs for different frequency bins to output time interval
    # between subsequent transmissions from the same transmitter.
    prev_soas = {}

    for detected, result in detections:
        if detected:
            print(result.serialize(), file=args['output'])

        if not quiet:
            # Calculate time interval between subsequent transmissions
            dt_idx = result.carrier_info.bin // 2
            prev_soa = prev_soas.get(dt_idx, result.soa)
            if detected:
                prev_soas[dt_idx] = result.soa

            # Output summary line
            summary = _summary_line(detected, result, prev_soa,
                                    config['sample_rate'], block_len)
            print(summary, file=info_out)


if __name__ == '__main__':
    _main()
