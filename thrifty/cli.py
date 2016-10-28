"""Common Thrifty CLI interface.

A centralized interface for accessing Thrifty modules with CLI interfaces.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import importlib


HELP = """usage: thrifty <command> [<args>]

Thrifty is proof-of-concept SDR software for TDOA positioning using inexpensive
SDR hardware such as the RTL-SDR.

Thrifty is divided into several modules. Each module is accessible as a command
and has its own arguments.

Valid commands are:

    ~ Core functionality ~
    capture           Capture carrier detections from RTL-SDR using fastcard
    detect            Detect presence of positioning signals and estimate SoA
    identify          Identify transmitter IDs and filter duplicate detections
    match             Match detections from multiple receivers
    tdoa              Estimate TDOA by synchronising with beacon transmissions
    pos               Estimate position from TDOA estimates

    ~ Analysis tools ~
    scope             Live time-domain and frequency-domain plots with triggers
    analyze_toads     Calculate statistics on data in a .toads file
    analyze_detect    Like 'detect', but plot signals for analysis
    analyze_beacon    Analyze the difference in SOA of a beacon between two RXs
    analyze_tdoa      Calculate stats from slices of the TDOA data

    ~ Utilities ~
    template_generate Generate a new (ideal) template
    template_extract  Extract a new template from captured data

Use 'thrifty help <command>' for information about the command's arguments."""


MODULES = {
    'capture': 'thrifty.fastcard_capture',
    'detect': 'thrifty.detect',
    'identify': 'thrifty.identify',
    'match': 'thrifty.matchmaker',
    'tdoa': 'thrifty.tdoa_est',
    'pos': 'thrifty.pos_est',
    'analyze_toads': 'thrifty.toads_analysis',
    'analyze_detect': 'thrifty.detect_analysis',
    'analyze_beacon': 'thrifty.beacon_analysis',
    'analyze_tdoa': 'thrifty.tdoa_analysis',
    'template_generate': 'thrifty.template_generate',
    'template_extract': 'thrifty.template_extract',
    'scope': 'thrifty.scope',
}


def _print_help():
    print(HELP)


def _main():
    if len(sys.argv) == 1:
        _print_help()
        sys.exit(1)

    command = sys.argv.pop(1)

    if command == 'help' or command == '--help':
        if len(sys.argv) == 2:
            command = sys.argv.pop(1)
            sys.argv.append('--help')
        else:
            _print_help()
            sys.exit(0)

    if command in MODULES:
        # pylint: disable=protected-access
        sys.argv[0] += ' ' + command
        module_name = MODULES[command]
        module = importlib.import_module(module_name)
        module._main()
    else:
        print("thrifty: {} is not a thrifty command. See 'thrifty --help'."
              .format(command), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _main()
