#!/usr/bin/env python

"""Detector that can be used to experiment with different carrier peak
interpolation methods.

Example usage:
    python -m "thrifty.experimental.detect_carrier_interpol" \
            --method cosine rx.card -o rx.data
"""

from __future__ import print_function
import argparse
from thrifty.detect import Detector, detector_cli
from thrifty.experimental import carrier_interpolators


class InterpolationDetector(Detector):
    def __init__(self, settings, blocks, rxid=-1, method=None, width=6):
        super(InterpolationDetector, self).__init__(settings, blocks, rxid)

        if method is not None:
            if isinstance(method, basestring):
                if method == 'none':
                    interpolator = carrier_interpolators.none
                elif method == 'parabolic':
                    interpolator = carrier_interpolators.parabolic
                elif method == 'cosine':
                    interpolator = carrier_interpolators.cosine
                elif method == 'gaussian':
                    interpolator = carrier_interpolators.gaussian
                elif method == 'dirichlet':
                    interpolator = carrier_interpolators.make_dirichlet(
                        settings.block_len,
                        settings.carrier_len)
                else:
                    raise KeyError('Unknown interpolation method')
                self.sync.interpolator = interpolator
            else:
                self.sync.interpolator = method


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    method_names = carrier_interpolators.INTERPOLATORS.keys()

    parser.add_argument('--method',
                        type=str,
                        default='dirichlet',
                        help="Carrier interpolation method. "
                             "Valid methods are: " + ' '.join(method_names))

    parser.add_argument('--width',
                        type=int,
                        default=6,
                        help="Number of samples to use for interpolation")

    detector_cli(InterpolationDetector, parser, ['method', 'width'])
