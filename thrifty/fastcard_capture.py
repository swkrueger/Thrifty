"""
Wrapper for using fastcard with a config file.

Example
-------
Assuming that all relevant settings are contained in the default config file,
blocks of data for which a carrier is detected can be captured from the RTL-SDR
using:

    $ fastcard_capture.py rx1.card


To check for detections without capturing it:

    $ fastcard_capture.py

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import signal
import subprocess
import sys
import logging

from thrifty import settings
from thrifty import setting_parsers


def _main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_file", nargs='?', help="Output file (.card)")
    parser.add_argument("--fastcard", dest="fastcard", default="fastcard",
                        help="Path to fastcard binary")
    parser.add_argument('-d', '--device-index', dest='device_index',
                        type=int, default=0, help="RTL-SDR device index")
    setting_keys = ['sample_rate', 'tuner_freq', 'tuner_gain',
                    'capture_skip', 'block_size', 'block_history',
                    'carrier_window', 'carrier_threshold']
    config, args = settings.load_args(parser, setting_keys)

    bin_freq = config.sample_rate / config.block_size
    window = setting_parsers.normalize_freq_range(
        config.carrier_window, bin_freq)
    constant, snr, stddev = config.carrier_threshold
    if stddev != 0:
        print("Warning: fastcard does not support 'stddev' in threshold "
              "formula", file=sys.stderr)

    call = [
        args.fastcard,
        '-i', 'rtlsdr',
        '-s', str(config.sample_rate),
        '-f', str(config.tuner_freq),
        '-g', str(config.tuner_gain),
        '-d', str(args.device_index),
        '-b', str(config.block_size),
        '-h', str(config.block_history),
        '-w', "{}-{}".format(window[0], window[1]),
        '-t', "{}c{}s".format(constant, snr),
        '-k', str(config.capture_skip)
    ]
    if args.output_file is not None:
        call.extend(['-o', args.output_file])
    logging.info("Calling %s", ' '.join(call))

    os.setpgrp()
    process = subprocess.Popen(call)

    def _signal_handler(signal_, _):
        try:
            if process.poll() is None:
                process.send_signal(signal_)
                returncode = process.wait()
                sys.exit(returncode)
        except OSError:
            pass

    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        returncode = process.wait()
        if returncode != 0:
            sys.exit(returncode)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    _main()
