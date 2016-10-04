"""Detect presence of positioning signals and estimate sample-of-arrival."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import sys
from collections import namedtuple

import numpy as np

from thrifty.settings import load_args
from thrifty import signal
from thrifty import toads_data
from thrifty import util
from thrifty.block_data import block_reader, card_reader
from thrifty.carrier_sync import DefaultSynchronizer
from thrifty.setting_parsers import normalize_freq_range
from thrifty.soa_estimator import SoaEstimator


DetectorSettings = namedtuple('DetectorSettings', [
    'block_len',
    'history_len',
    'carrier_len',
    'carrier_thresh',
    'carrier_window',
    'template',
    'corr_thresh'])


class Detector(object):
    """Detect positioning signals and estimate sample-of-arrival.

    All-in-one signal detection and sample-of-arrival estimation. Find carrier,
    synchronise to carrier, correlate with template, and estimate SoA.
    """
    def __init__(self, settings, blocks=None, rxid=-1, yield_data=False):
        self.settings = settings
        self.blocks = iter(blocks) if blocks is not None else None
        self.rxid = rxid
        self.yield_data = yield_data

        self.sync = DefaultSynchronizer(
            thresh_coeffs=settings.carrier_thresh,
            window=settings.carrier_window,
            block_len=settings.block_len,
            carrier_len=settings.carrier_len)

        self.soa_estimate = SoaEstimator(
            template=settings.template,
            thresh_coeffs=settings.corr_thresh,
            block_len=settings.block_len,
            history_len=settings.history_len)

        self.new_len = settings.block_len - settings.history_len

    def detect(self, timestamp, block_idx, block):
        """Process the given block of data."""
        assert len(block) == self.settings.block_len
        block_signal = signal.Signal(block)
        shifted_fft, carrier_info = self.sync(block_signal)

        if shifted_fft is not None:  # detected
            detected, corr_info, corr = self.soa_estimate(shifted_fft)
            soa = (self.new_len * block_idx
                   + corr_info.sample
                   + corr_info.offset)
        else:
            detected, corr_info, soa, corr = False, None, None, None

        result = toads_data.DetectionResult(timestamp, block_idx, soa,
                                            carrier_info, corr_info, self.rxid)
        if self.yield_data:
            return detected, result, shifted_fft, corr
        else:
            return detected, result

    def next(self):
        """Process the next block of data."""
        return self.detect(*next(self.blocks))

    def __call__(self, timestamp, block_idx, block):
        self.detect(timestamp, block_idx, block)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()


def _carrier_freq(carrier_info, block_len, sample_rate):
    """Convert carrier bin and offset to frequency value in Hertz."""
    bin_freq = sample_rate / block_len
    idx = util.fft_bin(carrier_info.bin, block_len)
    pos = idx + carrier_info.offset
    freq = pos * bin_freq
    return freq


class SummaryLineFormatter(object):
    """Generate a one-line summary of a detection."""
    def __init__(self, sample_rate, block_len, add_dt=False):
        self.sample_rate = sample_rate
        self.block_len = block_len
        self.add_dt = add_dt
        if add_dt:
            # Store previous SoAs for different frequency bins to output time
            # interval between subsequent transmissions from the same
            # transmitter.
            self.prev_soas = {}

    def __call__(self, detected, result):
        """Summarize detection results."""
        # if self.add_dt:
        #     # Calculate time interval between subsequent transmissions
        #     dt_idx = result.carrier_info.bin // 2
        #     prev_soa = self.prev_soas.get(dt_idx, result.soa)
        #     if detected:
        #         self.prev_soas[dt_idx] = result.soa
        #     time_diff = (result.soa - prev_soa) / self.sample_rate
        #     time_diff_str = " (+{:.1f}s)".format(time_diff)
        # else:
        #     time_diff_str = ""
        time_diff_str = ""

        carrier_detect = result.corr_info is not None
        carrier_freq = _carrier_freq(result.carrier_info,
                                     self.block_len,
                                     self.sample_rate)
        snr = util.snr(result.carrier_info.energy, result.carrier_info.noise)
        info = ("blk={blk}; carrier: {det} @ {freq:.3f} kHz"
                " / {idx:>3.0f}:{offset:.2f}, "
                "SNR = {ampl:>4.0f} / {noise:>2.0f} = {snr:>5.2f} dB"
                .format(blk=result.block,
                        det="yes" if carrier_detect else "no ",
                        freq=carrier_freq / 1e3,
                        idx=result.carrier_info.bin,
                        offset=result.carrier_info.offset,
                        ampl=result.carrier_info.energy,
                        noise=result.carrier_info.noise,
                        snr=snr))

        if carrier_detect:
            snr = util.snr(result.corr_info.energy, result.corr_info.noise)
            info += ("; corr: {det} @ {idx:>4}{offset:+.3f}{dt}"
                     ", SNR = {ampl:>4.0f}/{noise:>2.0f} = {snr:>5.2f} dB"
                     .format(det="yes" if detected else "no ",
                             idx=result.corr_info.sample,
                             offset=result.corr_info.offset,
                             dt=time_diff_str,
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-o', '--output', dest='output',
                       type=argparse.FileType('wb'),
                       help="Output file (.toad) ('-' for stdout)")
    group.add_argument('-a', '--append', dest='append',
                       type=argparse.FileType('ab'),
                       help="Output file to append to (.toad)")

    setting_keys = ['sample_rate', 'block_size', 'block_history',
                    'carrier_window', 'carrier_threshold',
                    'corr_threshold', 'template', 'rxid']
    config, args = load_args(parser, setting_keys)

    output_file = args.output if args.append is None else args.append
    info_out = sys.stderr if output_file == sys.stdout else sys.stdout
    bin_freq = config.sample_rate / config.block_size
    window = normalize_freq_range(config.carrier_window, bin_freq)

    if args.raw:
        blocks = block_reader(args.input, config.block_size,
                              config.block_history)
    else:
        blocks = card_reader(args.input)

    template = np.load(config.template)

    settings = DetectorSettings(block_len=config.block_size,
                                history_len=config.block_history,
                                carrier_len=len(template),
                                carrier_thresh=config.carrier_threshold,
                                carrier_window=window,
                                template=template,
                                corr_thresh=config.corr_threshold)
    detections = Detector(settings, blocks, rxid=config.rxid)
    summary_liner = SummaryLineFormatter(config.sample_rate,
                                         config.block_size,
                                         add_dt=True)

    for detected, result in detections:
        if detected and output_file is not None:
            print(result.serialize(), file=output_file)

        if not args.quiet:
            # Output summary line
            print(summary_liner(detected, result), file=info_out)


if __name__ == '__main__':
    _main()
