#!/usr/bin/env python

import matplotlib.pyplot as plt

def signed_index(i, N):
    if i < 0: return i
    return i if i <= (2*N - 1) / 4 else i - N


class CarrierSyncResults:
    def __init__(self):
        self.detected = False

    def plot(s, settings):
        f = np.fft.fftfreq(len(s.fft), 1.0/settings.sample_rate)
        ff = np.fft.fftshift(f) / 1000
        fft_mag = np.abs(s.fft)
        plt.plot(ff, np.fft.fftshift(fft_mag), label="FFT (unshifted)")

        # plt.hlines(s.noise, ff[0], ff[-1], label='Noise')
        plt.plot([ff[0], ff[-1]], [s.noise, s.noise], label="Noise")
        plt.plot([ff[0], ff[-1]], [s.threshold, s.threshold], label="Threshold")

        peak_mag = np.abs(s.fft[s.peak])

        plt.vlines([settings.carrier_freq_min / 1000, settings.carrier_freq_max / 1000], 0, np.max(fft_mag), label="Window")

        N = len(s.fft)
        peak_freq = signed_index(s.peak, N) * settings.sample_rate / N / 1000.0
        plt.annotate('(%.3f kHz, %.0f)' % (peak_freq, peak_mag), xy=(peak_freq, peak_mag))

        if s.detected:
            mag = np.abs(s.shifted_fft)
            plt.plot(ff, np.fft.fftshift(mag), label="FFT (shifted)")
            plt.annotate('(%.3f kHz, %.0f)' % (0, mag[0]), xy=(0, mag[0]))

        plt.legend(loc='best')
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Magnitude')
        plt.title('Carrier sync')

        plt.figtext(0.5, 0.95, s.summary(settings.sample_rate), horizontalalignment='center')

        # Zoom in
        lim_low = min(0, settings.carrier_freq_min) - 20e3
        lim_high = max(0, settings.carrier_freq_max) + 20e3
        plt.xlim([lim_low / 1000.0, lim_high / 1000.0])
        plt.ylim([0, N])


    def summary(s, sample_rate):
        if s.detected:
            peak_mag = np.abs(s.shifted_fft[0])
        else:
            peak_mag = np.abs(s.fft[s.peak])

        N = len(s.fft)
        signed_peak = signed_index(s.peak, N)
        peak_freq = signed_peak * sample_rate / N

        SNR = 20 * np.log10(peak_mag / s.noise)
        return "detected: %d, SNR: %.2f dB, f: %.3f kHz, peak: %.0f, thres: %.0f, noise: %.0f" % (s.detected, SNR, peak_freq / 1000, peak_mag, s.threshold, s.noise)
