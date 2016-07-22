"""Common Thrifty CLI interface.

A centralized interface for accessing Thrifty modules with CLI interfaces.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

from thrifty import detect
from thrifty import fastcard_capture
from thrifty import integrate
from thrifty import matchmaker
from thrifty import clock_sync


HELP = """usage: thrifty <command> [<args>]

Thrifty is proof-of-concept SDR software for TDOA positioning.

Thrifty is divided into several modules. Each module is accessible as a command
and has its own arguments.

Valid commands are:
    capture       Capture carrier detections from RTL-SDR using fastcard
    detect        Detect presence of positioning signals and estimate SoA
    integrate     Merge RX detections and identify transmitter IDs
    match         Match detections from multiple receivers
    clock_sync    Build clock sync model from beacon transmissions

Use 'thrifty help <command>' for information about the command's arguments."""


def _print_help():
    print(HELP)


def _main():
    # pylint: disable=protected-access

    if len(sys.argv) == 1:
        _print_help()
        sys.exit(1)

    command = sys.argv.pop(1)

    if command == 'help':
        if len(sys.argv) == 2:
            command = sys.argv.pop(1)
            sys.argv.append('--help')
        else:
            _print_help()
            sys.exit(0)

    if command == 'capture':
        method = fastcard_capture._main
    elif command == 'detect':
        method = detect._main
    elif command == 'integrate':
        method = integrate._main
    elif command == 'match':
        method = matchmaker._main
    elif command == 'clock_sync':
        method = clock_sync._main
    else:
        print("thrifty: {} is not a thrifty command. See 'thrifty --help'."
              .format(command), file=sys.stderr)
        sys.exit(1)

    sys.argv[0] += ' ' + command
    method()


if __name__ == "__main__":
    _main()
