"""
Unit tests for setting parsers module.
"""

import pytest

from thrifty import setting_parsers


def test_freq_range():
    """Test freq_range parser with valid values"""
    tests = [
        ('100', (100.0, 100.0, False)),
        ('-123.4', (-123.4, -123.4, False)),
        ('100-200', (100.0, 200.0, False)),
        ('10e1 - 20e1', (100.0, 200.0, False)),
        ('-100-100', (-100.0, 100.0, False)),
        ('-200--100', (-200.0, -100.0, False)),
        ('-200--100', (-200.0, -100.0, False)),
        ('100hz', (100.0, 100.0, True)),
        ('100-200 Hz', (100.0, 200.0, True)),
        ('10-20 khz', (10000.0, 20000.0, True)),
        ('1.2345-2.3456 KHZ', (1.2345e3, 2.3456e3, True)),
        ('433-435Mhz', (433e6, 435e6, True)),
        ('1.2345-2.3456 mhz', (1.2345e6, 2.3456e6, True)),
    ]
    for string, expected in tests:
        assert setting_parsers.freq_range(string) == expected


def test_freq_range_exceptions():
    """Test freq_range parser with invalid values"""
    tests = ['garbage']
    for test in tests:
        with pytest.raises(ValueError):
            setting_parsers.freq_range(test)


def test_metric_float():
    """Test metric_float parser with valid values"""
    tests = [
        ('1337.15', 1337.15),
        ('15.2M', 15200000.0),
        ('987k', 987000.0),
        ('55m', 0.055),
        ('164u ', 164e-6),
    ]
    for string, expected in tests:
        assert setting_parsers.metric_float(string) == expected


def test_metric_float_invalid():
    """Test metric_float parser with invalid values"""
    tests = ['garbage', 'x53m', '500A']
    for test in tests:
        with pytest.raises(ValueError):
            setting_parsers.metric_float(test)


def test_threshold():
    """Test threshold parser with valid values"""
    tests = [
        ('0', (0.0, 0.0, 0.0)),
        ('10.2', (10.2, 0.0, 0.0)),
        ('c', (1.0, 0.0, 0.0)),
        ('11c', (11.0, 0.0, 0.0)),
        ('100 * constant', (100.0, 0.0, 0.0)),
        ('snr', (0.0, 1.0, 0.0)),
        ('5.2*snr', (0.0, 5.2, 0.0)),
        (' 8s ', (0.0, 8.0, 0.0)),
        ('stddev', (0.0, 0.0, 1.0)),
        ('2.1stddev', (0.0, 0.0, 2.1)),
        ('8.7*d', (0.0, 0.0, 8.7)),
        ('10 + 4*snr', (10.0, 4.0, 0)),
        ('40 + 3.8*snr + 2 stddev', (40.0, 3.8, 2.0)),
        ('1+2s+3d+4+5s+6d', (5.0, 7.0, 9.0)),
        ('c + s + d', (1.0, 1.0, 1.0)),
    ]
    for string, expected in tests:
        assert setting_parsers.threshold(string) == expected


def test_threshold_exceptions():
    """Test threshold parser with invalid values"""
    tests = [
        '',
        ' ',
        '5+',
        'junk',
        '+5*stddev',
        '*snr',
        'stddev*snr',
        '5 * stdde',
        '2 * sn',
        'const',
    ]
    for test in tests:
        with pytest.raises(ValueError):
            setting_parsers.threshold(test)
