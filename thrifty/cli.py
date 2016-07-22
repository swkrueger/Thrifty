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
from thrifty import toads_analysis
from thrifty import template_generate
from thrifty import template_extract


HELP = """usage: thrifty <command> [<args>]

Thrifty is proof-of-concept SDR software for TDOA positioning using inexpensive
SDR hardware such as the RTL-SDR.

Thrifty is divided into several modules. Each module is accessible as a command
and has its own arguments.

Valid commands are:

    ~ Core functionality ~
    capture           Capture carrier detections from RTL-SDR using fastcard
    detect            Detect presence of positioning signals and estimate SoA
    integrate         Merge RX detections and identify transmitter IDs
    match             Match detections from multiple receivers
    clock_sync        Build clock sync model from beacon transmissions

    ~ Analysis tools ~
    analyze_toads     Calculate statistics on data in a .toads file

    ~ Utilities ~
    template_generate Generate a new (ideal) template
    template_extract  Extract a new template from captured data

Use 'thrifty help <command>' for information about the command's arguments."""


MODULES = {
    # pylint: disable=protected-access
    'capture': fastcard_capture._main,
    'detect': detect._main,
    'integrate': integrate._main,
    'match': matchmaker._main,
    'clock_sync': clock_sync._main,
    'analyze_toads': toads_analysis._main,
    'template_generate': template_generate._main,
    'template_extract': template_extract._main,
}


def _print_help():
    print(HELP)


def _main():
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

    if command in MODULES:
        sys.argv[0] += ' ' + command
        method = MODULES[command]
        method()
    else:
        print("thrifty: {} is not a thrifty command. See 'thrifty --help'."
              .format(command), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _main()
