from __future__ import print_function

import itertools
import numpy as np
from thrifty import pos_est
from thrifty import tdoa_est

SPEED_OF_LIGHT = pos_est.SPEED_OF_LIGHT

RX_POS = {
    0: [0, 10],
    1: [100, 0],
    2: [0, 100],
    3: [110, 90],
}

TX_POS = [30, 40]


def gen_tdoa_data(rx_pos, tx_pos):
    tdoas = []
    for rx0, rx1 in itertools.combinations(rx_pos.keys(), 2):
        rx0_pos = np.array(rx_pos[rx0])
        rx1_pos = np.array(rx_pos[rx1])
        t0 = np.linalg.norm(rx0_pos - tx_pos)
        t1 = np.linalg.norm(rx1_pos - tx_pos)
        tdoa = (t0 - t1) / SPEED_OF_LIGHT
        tdoas.append((rx0, rx1, tdoa, 0, 0, 0, 0))
    return np.array(tdoas, dtype=tdoa_est.TDOA_DTYPE)

tdoa_array = gen_tdoa_data(RX_POS, TX_POS)
print(pos_est.solve_numerically(tdoa_array, RX_POS))  # TODO: assert all close

rx_pairs = zip(tdoa_array['rx0'], tdoa_array['rx1'])
print(pos_est.dop(TX_POS, RX_POS, rx_pairs))


def test_1d_dop():
    tx_pos = [5]
    rx_pos = {0: [0], 1: [10]}
    rx_pairs = [(0, 1)]
    assert pos_est.dop(tx_pos, rx_pos, rx_pairs) == 0.5
