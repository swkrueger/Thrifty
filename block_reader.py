#!/usr/bin/env python

import numpy as np

def raw_chunk_reader(stream, chunk_size):
    while True:
        buf = stream.read(chunk_size)
        if len(buf) == 0:
            break
        yield buf


def data_block_reader(stream, block_size):
    """Read and convert fixed-sized blocks of samples."""
    chunk_size = block_size * 2 # I and Q
    chunk = ""
    for raw in raw_chunk_reader(stream, chunk_size - len(chunk)):
        if len(raw) == 0:
            chunk = raw
        else:
            chunk += raw

        if len(chunk) < chunk_size:
            continue

        data = np.frombuffer(chunk, dtype=np.uint8)

        iq = np.empty(len(data)//2, 'complex') # gebruik complex64 eerder
        iq.real, iq.imag = data[::2], data[1::2]
        iq /= (255/2)
        iq -= (1 + 1j)

        assert(len(iq) == block_size)
        yield iq
        chunk = ""


import matplotlib.pyplot as plt
def data_reader(stream, settings):
    block_len, history_len = settings.block_len, settings.history_len
    assert(settings.data_len == block_len + history_len)

    data = np.zeros(settings.data_len)
    for b in data_block_reader(stream, block_len):
        data = np.concatenate([data[-history_len:], b])
        yield data

