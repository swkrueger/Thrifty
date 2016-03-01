#!/usr/bin/env python

import carrier_sync
import numpy as np
import unittest

# import matplotlib.pyplot as plt

window_test_data = [
    #   freq_min  ,  freq_max  ,    freq    , detected
    [-81.0e3, -79.0e3, -80.0e3, True],
    [-81.0e3, -79.0e3, -79.1e3, True],
    [-81.0e3, -79.0e3, -80.9e3, True],
    [-81.0e3, -79.0e3, -82.0e3, False],
    [-81.0e3, -79.0e3, -78.0e3, False],
    [-81.0e3, -79.0e3,  80.0e3, False],
    [-81.0e3, -79.0e3,   0.0e3, False],

    [ 79.0e3,  81.0e3,  80.0e3, True],
    [ 79.0e3,  81.0e3,  79.1e3, True],
    [ 79.0e3,  81.0e3,  80.9e3, True],
    [ 79.0e3,  81.0e3,  82.0e3, False],
    [ 79.0e3,  81.0e3,  78.0e3, False],
    [ 79.0e3,  81.0e3, -80.0e3, False],
    [ 79.0e3,  81.0e3,   0.0e3, False],

    [-10.0e3,   5.0e3,   0.0e3, True],
    [-10.0e3,   5.0e3,  -9.9e3, True],
    [-10.0e3,   5.0e3,   4.9e3, True],
    [-10.0e3,   5.0e3,   6.0e3, False],
    [-10.0e3,   5.0e3, -11.0e3, False],
]

class Settings:
    sample_rate = 2.2e6
    carrier_noise_window_size = 1
    carrier_threshold_constant = 500
    carrier_threshold_snr = 0

class TestCarrierSync(unittest.TestCase):
    def testWindow(self):
        for freq_min, freq_max, carrier_freq, detected in window_test_data:
            s = Settings()
            s.carrier_freq_min = freq_min
            s.carrier_freq_max = freq_max
            
            N = 8192
            W = 2085
            c = np.exp(2j * np.pi * carrier_freq * np.arange(W) / s.sample_rate)
            y = np.concatenate([c, np.zeros(N - len(c))])

            r = carrier_sync.carrier_syncer(s)(y)

            # print freq_min, freq_max, carrier_freq, detected
            # r.plot(s)
            # plt.show()

            self.assertEqual(r.detected, detected)
            if detected:
                self.assertGreater(np.abs(r.shifted_fft[0]), W * 0.8)
            else:
                self.assertLess(np.abs(r.fft[r.peak]), s.carrier_threshold_constant)


if __name__ == '__main__':
    unittest.main()
