"""
Helper functions for converting setting strings to values.
"""

import re


# from https://docs.python.org/2/library/re.html#simulating-scanf
_FLOAT_REGEX = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
_RANGE_REGEX = r'\s*-\s*'
_FREQ_MAG_REGEX = r'[kKmM]?'
_HERTZ_REGEX = r'[hH][zZ]'

_FREQ_RANGE_PATTERN = re.compile(r'^({0})(?:{1}({0}))?\s*({2})({3})?$'.format(
    _FLOAT_REGEX, _RANGE_REGEX, _FREQ_MAG_REGEX, _HERTZ_REGEX), re.IGNORECASE)

_THRESHOLD_SYMBOL = r'constant|c|snr|s|stddev|d|'
_THRESHOLD_TERM = r'^\s*(?=\S)(?:({0})\s*\*?\s*)?({1})\s*$'.format(
    _FLOAT_REGEX, _THRESHOLD_SYMBOL)

_SI_PREFIXES = {
    'y': 1e-24,  # yocto
    'z': 1e-21,  # zepto
    'a': 1e-18,  # atto
    'f': 1e-15,  # femto
    'p': 1e-12,  # pico
    'n': 1e-9,   # nano
    'u': 1e-6,   # micro
    'm': 1e-3,   # mili
    'c': 1e-2,   # centi
    'd': 1e-1,   # deci
    'k': 1e3,    # kilo
    'M': 1e6,    # mega
    'G': 1e9,    # giga
    'T': 1e12,   # tera
    'P': 1e15,   # peta
    'E': 1e18,   # exa
    'Z': 1e21,   # zetta
    'Y': 1e24,   # yotta
}


def metric_float(string):
    """Parse a float with an optional metric prefix as suffix.

    Examples
    --------
    >>> metric_float('123.4')
    123.4
    >>> metric_float('1.2M')
    1200000.0
    >>> metric_float('3.4m')
    0.0034
    """
    string = string.strip()
    if len(string) > 0 and string[-1] in _SI_PREFIXES:
        prefix = string[-1]
        quantity_str, multiplier = string[:-1], _SI_PREFIXES[prefix]
    else:
        quantity_str, multiplier = string, 1
    return float(quantity_str) * multiplier


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

    match = re.match(_FREQ_RANGE_PATTERN, string)

    if not match:
        raise ValueError('Invalid range: {}'.format(string))

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


def threshold(string):
    """Parse threshold formula given as a string.

    Parameters
    ----------
    string : str

    Returns
    -------
    (constant, snr, stddev) : float

    Raises
    ------
    ValueError
        If the given string does not represent a treshold formula.

    Examples
    --------
    >>> threshold("5 + 3*snr + stddev")
    (5.0, 3.0, 1.0)
    >>> threshold("10c+5s+2d")
    (10.0, 5.0, 2.0)

    """

    if not string:
        raise ValueError('Empty string')
    constant, snr, stddev = 0.0, 0.0, 0.0
    terms = string.split('+')
    for term in terms:
        match = re.match(_THRESHOLD_TERM, term)
        if not match:
            raise ValueError('Invalid threshold term: {}'.format(term))
        quantity_str, symbol = match.groups()
        if quantity_str is None:
            quantity = 1.0
        else:
            quantity = float(quantity_str)
        if symbol == 'constant' or symbol == 'c' or symbol == '':
            constant += quantity
        elif symbol == 'snr' or symbol == 's':
            snr += quantity
        elif symbol == 'stddev' or symbol == 'd':
            stddev += quantity
    return constant, snr, stddev
