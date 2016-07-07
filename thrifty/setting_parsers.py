"""
Helper functions for converting setting strings to values.
"""

import re


# from https://docs.python.org/2/library/re.html#simulating-scanf
FLOAT_REGEX = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
RANGE_REGEX = r'\s*-\s*'
FREQ_MAG_REGEX = r'[kKmM]?'
HERTZ_REGEX = r'[hH][zZ]'

FREQ_RANGE_PATTERN = re.compile(r'^({0})(?:{1}({0}))?\s*({2})({3})?$'.format(
    FLOAT_REGEX, RANGE_REGEX, FREQ_MAG_REGEX, HERTZ_REGEX), re.IGNORECASE)


def metric_float(string):
    """Parse a float with an optional metric prefix as suffix."""
    # TODO: parse metric prefix
    return float(string)


def freq_range(string):
    """Parse carrier frequency range given as a string.

    Parameters
    ----------
    string : str

    Returns
    -------
    start : float
    stop : float
    unit_hz : bool
        If true, the range is given in Hz.
        If false, it is given as the index of the frequency bin.

    Raises
    ------
    ValueError
        If the given string does not represent a frequency range.

    Examples
    --------
    >>> freq_range("10-20")
    (100.0, 200.0, False)
    >>> freq_range("100-200 hz")
    (100.0, 200.0, True)
    >>> freq_range("2 - 3 MHz")
    (2000000.0, 3000000.0, True)

    """

    match = re.match(FREQ_RANGE_PATTERN, string)

    if not match:
        raise ValueError('Invalid range')

    groups = match.groups()

    start_str, stop_str, prefix, unit = groups
    if stop_str is None:
        stop_str = start_str

    start, stop = float(start_str), float(stop_str)
    unit_hz = unit is not None

    if prefix.lower() == 'k':
        start, stop = start * 1e3, stop * 1e3
    elif prefix.lower() == 'm':
        start, stop = start * 1e6, stop * 1e6

    return start, stop, unit_hz


def normalize_freq_range(range_, bin_freq):
    """Normalize a frequency range to discrete frequency bin values.

    Parameters
    ----------
    range_: (float, float, bool)
        The result of `freq_range`, thus (start, stop, unit_hz).
    bin_freq : float
        Width of each frequency bin, in Hertz.

    Returns
    -------
    start_bin : int
    """

    start, stop, hz_unit = range_
    if not hz_unit:
        return int(start), int(stop)
    else:
        start = int(start / bin_freq)
        stop = int(stop / bin_freq)
        return start, stop
