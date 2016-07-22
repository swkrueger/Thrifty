"""
Generate Gold Codes / sequences.

Implementation is based on the gold code module in Matthew Baker's Signal
Processing Library (https://mubeta06.github.io/python/sp/) (LGPL license).
"""

from __future__ import print_function
from __future__ import division

import numpy as np


# Preferred pairs of LSFR taps
TAPS = {
    5: [[2], [1, 2, 3]],
    6: [[5], [1, 4, 5]],
    7: [[4], [4, 5, 6]],
    8: [[1, 2, 3, 6, 7], [1, 2, 7]],
    9: [[5], [3, 5, 6]],
    10: [[2, 5, 9], [3, 4, 6, 8, 9]],
    11: [[9], [3, 6, 9]],
}


def gold(bits, idx):
    """Generate the idx-th Gold code of length 2^bits - 1.

    Parameters
    ----------
    bits : int
        Length of LFSR. The length of the gold code will be
        :math:`2^{\\mathtt{bits}} - 1`.
    idx : int
        Index of the code to generate within the set of gold codes, where
        :math:`0 \\le \\mathtt{idx} < 2^{\\mathtt{bits}} + 1`.
    """
    bits = int(bits)
    if bits not in TAPS:
        raise ValueError('Preferred pairs for %d bits unknown.' % bits)
    seed = np.ones(bits, dtype=bool)

    seq1 = lfsr(TAPS[bits][0], seed)
    seq2 = lfsr(TAPS[bits][1], seed)

    if idx == 0:
        return seq1
    elif idx == 1:
        return seq2
    else:
        return np.logical_xor(seq1, np.roll(seq2, -idx + 2))


def lfsr(taps, init):
    """Generate a sequence using a linear feedback shift register (LSFR).

    Adapted from https://git.io/vKPF1.

    Parameters
    ----------
    taps: list or np.array
        List of polynomial exponents for non-zero terms other than 1 and n.
    init: list or np.array
        List of buffer initialisation values as 1's and 0's or booleans.

    Returns
    -------
    seq : np.array
    """
    nbits = len(init)
    init = np.array(init, dtype='bool')

    seq_len = (2**nbits) - 1
    seq = np.zeros(seq_len, dtype='bool')
    seq[:len(init)] = init

    for i in xrange(len(init), seq_len):
        seq[i] = seq[i - len(init)]
        for tap in taps:
            seq[i] ^= seq[i - len(init) + tap]

    return seq


def plot(seq):
    """Plot the autocorrelation of the given sequence."""
    import matplotlib.pyplot as plt

    bipolar = np.where(seq, 1.0, -1.0)
    autocorr = np.correlate(bipolar, bipolar, 'same')

    plt.figure()
    plt.title("Length {} Gold code autocorrelation".format(len(seq)))
    xdata = np.arange(len(seq)) - len(seq) // 2
    plt.plot(xdata, autocorr, '.-')
    plt.show()


def _print_stats(seq):
    bipolar = np.where(seq, 1.0, -1.0)
    autocorr = np.correlate(bipolar, bipolar, 'same')

    peaks = np.sort(np.abs(autocorr))
    peak = peaks[-1]
    noise = np.sqrt(np.mean(peaks[:-1]**2))

    peak_to_peak2 = peak / peaks[-2]
    peak_to_noise = peak / noise

    print("Peak amplitude: {:.0f}".format(peak))
    print("Largest non-peak amplitude: {:.0f}".format(peaks[-2]))
    print("Peak-to-max: {:.2f}".format(peak_to_peak2))
    print("Peak-to-noise: {:.2f}".format(peak_to_noise))


def _main():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('length', type=int,
                        help="Register length -- code length will be 2^n-1")
    parser.add_argument('index', nargs='?', type=int, default=0,
                        help="Code index -- which Gold code to generate")
    parser.add_argument('-p', '--plot', action='store_true',
                        help="Plot autocorrelation function")
    parser.add_argument('--stats', action='store_true',
                        help="Don't print the sequence, but print some stats "
                             "about the sequence.")
    args = parser.parse_args()

    seq = gold(args.length, args.index)

    if args.stats:
        _print_stats(seq)
    else:
        print(map(int, list(seq)))

    if args.plot:
        plot(seq)


if __name__ == '__main__':
    _main()
