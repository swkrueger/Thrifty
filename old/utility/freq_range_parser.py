import argparse
import re

# from https://docs.python.org/2/library/re.html#simulating-scanf
FLOAT_REGEX = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
RANGE_REGEX = r'\s*-\s*'
PREFIX_REGEX = r'[kM]?'
HERTZ_REGEX = r'[hH][zZ]'

PATTERN = re.compile('^({0})(?:{1}({0}))?\s*({2})({3})?$'.format(
    FLOAT_REGEX, RANGE_REGEX, PREFIX_REGEX, HERTZ_REGEX), re.IGNORECASE)


class FreqRangeAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(FreqRangeAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        v = parse(values)
        setattr(namespace, self.dest, v)


def parse(s):
    """Parse frequency range given as a string.
    
    Examples:
        '10-20' -> (100, 200, False)
        '100-200 hz' -> (100, 200, True)
    """
    match = re.match(PATTERN, s)

    if not match:
        return None
    else:
        g = match.groups()

        mins, maxs, prefix, unit_hz = g[0], g[1], g[2], g[3]
        if maxs == None:
            maxs = mins

        minf, maxf = float(mins), float(maxs)

        if prefix.lower() == 'k':
            minf, maxf = minf * 1e3, maxf * 1e3
        if prefix.lower() == 'M':
            minf, maxf = minf * 1e6, maxf * 1e6

        return minf, maxf, unit_hz != None


def normalize(r, bin_freq):
    minf, maxf, hz_unit = r
    if not hz_unit:
        return int(minf), int(maxf)
    else:
        minf = int(minf / bin_freq)
        maxf = int(maxf / bin_freq)
        return minf, maxf
        
