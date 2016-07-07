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
