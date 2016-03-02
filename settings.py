#!/usr/bin/env python

"""
Default settings for SDR detector.
"""

import numpy as np

# RF frontend settings
frequency = 433.1e6
sample_rate = 2.2e6
gain = 20

# Code settings
code_samples = np.load(open('template.npy', 'r'))
code_len = len(code_samples)

chip_rate = 1.08e6
sps = sample_rate / chip_rate
# peak_width = int(2 * np.ceil(sps))
peak_width = 0 # temp

# Block reader settings
# - block_len >= code_len
# - history = block_len + peak_width
# - data_len = history + block_len = 2*block_len + peak_width
# - data_len should be a power of two
history_len = code_len + peak_width
min_data_len = code_len + history_len
data_len = 1<<int(np.ceil(np.log2(min_data_len)))
block_len = data_len - history_len

# Carrier sync settings
carrier_freq_min = -80e3
carrier_freq_max = -78e3
carrier_noise_window_size = 10 # blocks
carrier_threshold_constant = 40
carrier_threshold_snr = 2

# Peak detector settings
detector_noise_window_size = 10
detector_threshold = {
    'constant': 5,
    'snr': 4,
    'stddev': 1,
}

