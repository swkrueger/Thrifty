#!/usr/bin/env python

"""Detector that can be used to experiment with different correlation peak
interpolation methods.

Example usage:
    python -m "thrifty.experimental.detect_xcorr_interpol" \
            --method autocorr rx.card -o rx.data
"""

from __future__ import print_function

import argparse

from thrifty.detect import Detector, detector_cli
from thrifty.soa_estimator import SoaEstimator
from thrifty.experimental import xcorr_interpolators


class IterativeSoaEstimator(SoaEstimator):
    def __init__(self, **args):
        super(IterativeSoaEstimator, self).__init__(**args)
        self._iterative = xcorr_interpolators.make_maximise(args['template'])
        self._last_fft = None
        self.interpolate = self.iterative_interpolate

    def soa_estimate(self, fft):
        self._last_fft = fft
        return super(IterativeSoaEstimator, self).soa_estimate(fft)

    def iterative_interpolate(self, corr_mag, peak_idx):
        signal = self._last_fft.ifft
        guess = xcorr_interpolators.gaussian(corr_mag, peak_idx)
        return self._iterative(signal, peak_idx, guess)


class InterpolationDetector(Detector):
    def __init__(self, settings, blocks, rxid=-1, method='gaussian'):
        super(InterpolationDetector, self).__init__(settings, blocks, rxid)

        if method == 'maximise':
            self.soa_estimate = IterativeSoaEstimator(
                template=settings.template,
                thresh_coeffs=settings.corr_thresh,
                block_len=settings.block_len,
                history_len=settings.history_len)
        else:
            if method == 'none':
                interpolator = xcorr_interpolators.none
            elif method == 'parabolic':
                interpolator = xcorr_interpolators.parabolic
            elif method == 'cosine':
                interpolator = xcorr_interpolators.cosine
            elif method == 'gaussian':
                interpolator = xcorr_interpolators.gaussian
            elif method == 'autocorr':
                interpolator = xcorr_interpolators.make_autocorr_fit(
                    settings.template)
            else:
                raise KeyError('Unknown interpolation method')

            self.soa_estimate.interpolate = interpolator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    method_names = xcorr_interpolators.INTERPOLATORS.keys()

    parser.add_argument('--method',
                        type=str,
                        default='gaussian',
                        help="Correlation interpolation method. "
                             "Valid methods are: " + ' '.join(method_names))

    detector_cli(InterpolationDetector, parser, ['method'])
