"""
Unit test for settings module.
"""

import io

import numpy as np
import numpy.testing as npt

from thrifty import block_data


def test_raw_to_complex():
    """Test raw-to-complex conversion."""
    raw_list = [0, 0, 127, 128, 255, 255]
    complex_list = [-0.9953-0.9953j, -0.0031+0.0047j, 0.9969+0.9969j]
    raw = np.array(raw_list, dtype=np.uint8)
    expected_result = np.array(complex_list, dtype=np.complex64)
    result = block_data.raw_to_complex(raw)
    npt.assert_allclose(result, expected_result, rtol=1e-2)


def test_complex_to_raw():
    """Test complex-to-raw conversion."""
    raw_list = [0, 0, 127, 128, 255, 255]
    complex_list = [-0.9953-0.9953j, -0.0031+0.0047j, 0.9969+0.9969j]
    raw = np.array(raw_list, dtype=np.uint8)
    complex_array = np.array(complex_list, dtype=np.complex64)
    actual_raw = block_data.complex_to_raw(complex_array)
    npt.assert_array_equal(actual_raw, raw)


def test_raw_to_complex_inverse():
    """Test that raw-to-complex is inverse of complex-to-raw."""
    expected = np.arange(256, dtype=np.uint8)
    actual = block_data.complex_to_raw(block_data.raw_to_complex(expected))
    npt.assert_array_equal(actual, expected)


def test_block_reader():
    """Test block_reader size and history."""
    stream = io.BytesIO("\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b"
                        "\x0c\x0d")
    blocks = list(block_data.block_reader(stream, 3, 1))

    indices = [b[1] for b in blocks]
    data = [b[2] for b in blocks]
    raw = [list(block_data.complex_to_raw(d)) for d in data]
    expected_raw = [[0x7f, 0x7f, 0x00, 0x01, 0x02, 0x03],
                    [0x02, 0x03, 0x04, 0x05, 0x06, 0x07],
                    [0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b]]
    # Note: The last few samples, [0x0a, 0x0b, 0x0c, 0x0d], will be skipped
    # since it does not fill an entire block_data.

    assert raw == expected_raw
    assert indices == range(len(data))


def test_card_reader():
    """Basic test for card_reader"""
    stream = io.BytesIO("# Some comments\n"
                        "# more comments\n"
                        "1000.5425 10 r0+Om5==\n"
                        "1000.5442 20 aaaaaa==")
    blocks = list(block_data.card_reader(stream))
    timestamps, indices, data = zip(*blocks)
    chars = [tuple(block_data.complex_to_raw(x)) for x in data]

    assert timestamps == (1000.5425, 1000.5442)
    assert indices == (10, 20)
    assert chars == [(175, 79, 142, 155), (105, 166, 154, 105)]
