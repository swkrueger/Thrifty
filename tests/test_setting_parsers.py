"""
Unit tests for setting parsers module.
"""

import pytest

from thrifty import setting_parsers


def test_freq_range():
    """Test freq_range function"""
    freq_range = setting_parsers.freq_range
    assert freq_range('100') == (100.0, 100.0, False)
    assert freq_range('-123.4') == (-123.4, -123.4, False)
    assert freq_range('100-200') == (100.0, 200.0, False)
    assert freq_range('10e1 - 20e1') == (100.0, 200.0, False)
    assert freq_range('-100-100') == (-100.0, 100.0, False)
    assert freq_range('-200--100') == (-200.0, -100.0, False)
    assert freq_range('-200--100') == (-200.0, -100.0, False)
    assert freq_range('100hz') == (100.0, 100.0, True)
    assert freq_range('100-200 Hz') == (100.0, 200.0, True)
    assert freq_range('10-20 khz') == (10000.0, 20000.0, True)
    assert freq_range('1.2345-2.3456 KHZ') == (1.2345e3, 2.3456e3, True)
    assert freq_range('433-435Mhz') == (433e6, 435e6, True)
    assert freq_range('1.2345-2.3456 mhz') == (1.2345e6, 2.3456e6, True)
    with pytest.raises(ValueError):
        freq_range('garbage')
