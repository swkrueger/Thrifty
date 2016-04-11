#!/usr/bin/env python

"""
Insert some nice description here.
"""

import numpy as np
from scipy.optimize import curve_fit


def freq_shift_test(data, delta):
    freq_shift = np.exp(-2j * np.pi * delta * np.arange(len(data)))
    y3 = data * freq_shift
    shifted_fft = np.fft.fft(y3)
    return shifted_fft


def gen_fit_func(settings):
    N, W = settings.data_len, settings.history_len
    def dirichlet_fit(k, a, t):
        k = np.array(k, dtype=np.float64)
        f = lambda k: a * np.abs(np.sin(np.pi*W*(k-t)/N)/W/np.sin(np.pi*(k-t)/N))
        res = np.piecewise(k, [k-t == 0, k-t != 0], [lambda x: a, f])
        # print k, a, t, res
        # print np.abs(freq_shift_test(res, -t))
        return res
    return dirichlet_fit


def interpolate(peak_idx, fft_mag, settings):
    fit_func = gen_fit_func(settings)

    n = 3
    xdata = np.array(np.arange(-n, n+1))
    ydata = fft_mag[peak_idx + xdata]
    p0 = (fft_mag[peak_idx], 0)
    # print ydata
    popt, pcov = curve_fit(fit_func, xdata, ydata, p0=p0)

    bin_freq = settings.sample_rate / settings.data_len
    fit_ampl, fit_offset = popt
    # print fit_offset

    # print p0
    # print 'Curve fit:', fit_offset, popt, pcov
    # adjusted_freq = (peak_idx + fit_offset) * bin_freq
    # peak_freq = peak_idx * bin_freq
    # print 'fit offset', peak_freq, fit_offset * bin_freq, adjusted_freq
    # perr = np.sqrt(np.diag(pcov))
    # print 'estimated fit error (1 std dev)', perr

    # todo: add unit test
    return fit_offset


def freq_shift(fft, peak, delta):
    offset = peak + delta
    # offset = delta
    # f = np.fft.fftshift(np.fft.fftfreq(len(data)))
    f = np.arange(len(fft)) * 1. / len(fft) - 0.5
    freq_shift = np.exp(-2j * np.pi * offset * f)
    y3 = np.fft.ifft(fft) * freq_shift
    shifted_fft = np.fft.fft(y3)
    return shifted_fft


def fit(c, settings):
    c.offset = interpolate(c.peak, np.abs(c.fft), settings)
    c.shifted_fft = freq_shift(c.fft, c.peak, c.offset)
    # print 'offset', c.offset
    # c.shifted_fft = freq_shift(c.shifted_fft, 0, c.offset) # 0.05

    # n = 2
    # xdata = np.array(np.arange(-n, n+1))
    # print np.abs(c.shifted_fft[xdata])

