#!/usr/bin/env python

"""Detector that uses precomputed shifted template to compensate for the
carrier frequency offset.

Example usage:
    python -m "thrifty.experimental.detect_preshift" --num 101 rx.card
"""

from __future__ import print_function
import argparse

import numpy as np

from thrifty.detect import Detector, detector_cli
from thrifty.carrier_sync import freq_shift_integer
from thrifty.signal_utils import Signal
from thrifty.experimental.carrier_interpolators import parabolic


NUM_TEMPLATES = 21


class TemplateShifts(object):
    def __init__(self, template, block_len, num=NUM_TEMPLATES):
        self.corr_len = block_len - len(template) + 1
        template_padded = np.concatenate([template, np.zeros(self.corr_len-1)])
        template_padded = Signal(template_padded)

        self.shifts = np.linspace(-0.5, 0.5, num)
        freqs = np.arange(block_len) * 1. / block_len - 0.5

        self.ffts = []
        for shift in self.shifts:
            shift_signal = np.exp(-2j * np.pi * shift * freqs)
            shifted = template_padded * shift_signal
            self.ffts.append(shifted.fft.conj)

        self.num = num

    def get_nearest(self, shift):
        assert shift >= -0.5 and shift <= 0.5
        idx = int(np.round((shift + 0.5) * (self.num - 1)))
        return self.ffts[idx]


class PreshiftDetector(Detector):
    def __init__(self, settings, blocks, rxid=-1, yield_data=False,
                 num=NUM_TEMPLATES, interpolator=parabolic, corr_shift=False):
        super(PreshiftDetector, self).__init__(settings, blocks,
                                               rxid, yield_data)
        self.block_len = settings.block_len
        self.sync.shifter = self.freq_shift_postpone
        self.soa_estimate.despread = self.despread_shift
        self.templates = TemplateShifts(settings.template, settings.block_len,
                                        num=num)
        self.frac_shift = None
        if interpolator is not None:
            self.sync.interpolator = interpolator
        self.corr_shift = False

    def freq_shift_postpone(self, signal, shift):
        int_shift = int(np.round(shift))
        self.frac_shift = shift - int_shift
        return freq_shift_integer(signal, int_shift)

    def despread_shift(self, fft):
        template_fft = self.templates.get_nearest(self.frac_shift)
        corr_fft = fft * template_fft
        corr_full = corr_fft.ifft

        if self.corr_shift:
            freqs = np.arange(self.block_len) * 1. / self.block_len - 0.5
            shift_signal = np.exp(2j * np.pi * self.frac_shift * freqs)
            corr_shifted = corr_full * shift_signal
            corr = corr_shifted[:self.templates.corr_len]
        else:
            corr = corr_full[:self.templates.corr_len]

        return corr


class DefaultDetector(Detector):
    def __init__(self, settings, blocks, rxid=-1, yield_data=False,
                 integer_shift=False, interpolator=parabolic):
        super(DefaultDetector, self).__init__(settings, blocks,
                                              rxid, yield_data)
        if integer_shift:
            self.sync.shifter = freq_shift_integer
        if interpolator is not None:
            self.sync.interpolator = interpolator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--num',
                        type=int,
                        default=NUM_TEMPLATES,
                        help="Number of templates to precompute")

    detector_cli(PreshiftDetector, parser, ['num'])
