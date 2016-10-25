"""A host of different cross-correlation peak interpolation methods."""

import numpy as np
import scipy.optimize


def _time_shift(samples, shift):
    """Shift samples by shift samples."""
    freqs = np.fft.fftfreq(len(samples))
    fft_shift = np.exp(-2j * np.pi * shift * freqs)
    fft = np.fft.fft(samples)
    return np.fft.ifft(fft * fft_shift)


def _partial_xcorr(template, signal, indices):
    assert len(template) == len(signal)
    N = len(template)
    corr = np.zeros(len(indices), dtype=signal.dtype)
    template_conj = np.conj(template)
    for i, index in enumerate(indices):
        signal_slice = signal[max(0, index) : min(N, N+index)]
        template_slice = template_conj[max(0, -index) : min(N, N-index)]
        corr[i] = np.sum(signal_slice * template_slice)
    return corr


def _clip_offset(offset, max_=0.5):
    return -max_ if offset < -max_ else max_ if offset > max_ else offset


def none(corr_mag, peak):
    return 0


def parabolic(corr_mag, peak):
    a, b, c = corr_mag[peak-1], corr_mag[peak], corr_mag[peak+1]
    offset = 0.5 * (c - a) / (2 * b - a - c)
    return offset


def gaussian(corr_mag, peak):
    a, b, c = corr_mag[peak-1], corr_mag[peak], corr_mag[peak+1]
    a, b, c = np.log(a), np.log(b), np.log(c)
    offset = 0.5 * (c - a) / (2 * b - a - c)
    return offset


def cosine(corr_mag, peak):
    a, b, c = corr_mag[peak-1], corr_mag[peak], corr_mag[peak+1]
    cos_omega = (a + c) / (2*b)
    if cos_omega > 1:
        return 0
    omega = np.arccos(cos_omega)
    theta = np.arctan((a - c) / (2*b*np.sin(omega)))
    offset = -theta / omega
    return offset


def make_autocorr_fit(template):
    ook_template = (template - np.min(template)) * 2

    def autocorr_fit(corr_mag, peak, n=2):
        initial_offset = _clip_offset(gaussian(corr_mag, peak))

        rel = np.arange(-n, n+1)
        xcorr = corr_mag[peak+rel]
        autocorr = _partial_xcorr(ook_template, template, rel)
        autocorr *= np.sum(xcorr) / np.sum(autocorr)

        def func(xdata, amplitude, offset):
            return amplitude * np.abs(_time_shift(xcorr, -offset))

        weights = np.abs(rel)+1
        try:
            popt, _ = scipy.optimize.curve_fit(func, rel, autocorr,
                                               p0=(1, initial_offset),
                                               bounds=([0.1, -0.55],
                                                       [2, 0.55]),
                                               sigma=weights)
            _, offset_lsq = popt
        except RuntimeError:  # "Optimal parameters not found"
            return initial_offset

        # amplitude_lsq, offset_lsq = popt
        # plt.plot(func(None, amplitude_lsq, offset_lsq))
        # plt.plot(autocorr)
        # plt.show()

        return offset_lsq

    return autocorr_fit


def make_maximise(template):
    template_fft = np.fft.fft(template).conjugate()

    def iterative(signal, peak, guess=0):
        signal = signal[peak:peak+len(template)]
        signal_fft = np.fft.fft(signal)
        xcorr_fft = signal_fft * template_fft

        def func(offset):
            freqs = np.fft.fftfreq(len(xcorr_fft))
            fft_shift = np.exp(2j * np.pi * offset * freqs)
            return -np.abs(np.sum(xcorr_fft * fft_shift))

        res = scipy.optimize.minimize(func, guess, bounds=[(-0.55, 0.55)])
        return res.x[0]

    return iterative


INTERPOLATORS = {
    'none': none,
    'parabolic': parabolic,
    'gaussian': gaussian,
    'cosine': cosine,
    'autocorr': make_autocorr_fit,
    'maximise': make_maximise,
    }


# def interpolate_phase(signal):
#     # FIXME: Problems with phase unwrapping!
#     signal_fft = np.fft.fft(signal)
#     etfe = np.fft.fftshift(signal_fft / template_fft)
#     angle = np.unwrap(np.angle(etfe), discont=np.pi*1.5)
#     mag = np.abs(np.fft.fftshift(signal_fft))
#
#     angle = angle[1000:4000]
#     mag = mag[1000:4000]
#
#     coeffs = np.polyfit(np.arange(len(angle)), angle, 1, w=mag**2)
#     slope = coeffs[0]
#
#     # slope = np.sum(np.diff(angle)) / len(angle)
#
#     offset = slope * len(signal) / (2 * np.pi)
#
#     angle2 = coeffs[0] * np.arange(len(angle)) + coeffs[1]
#     plt.plot(angle)
#     plt.plot(angle2)
#     plt.show()
#     # plt.plot(mag)
#     # plt.show()
#
#     return -offset
#
#
# def interpolate_phase2(signal):
#     # FIXME: Problems with phase unwrapping!
#     signal_fft = np.fft.fft(signal)
#     etfe = np.fft.fftshift(signal_fft / template_fft)
#     angle = np.angle(etfe)
#     weights = np.abs(np.fft.fftshift(signal_fft))**2
#     x0 = np.polyfit(np.arange(len(angle)), angle, 1, w=weights)
#
#     def fun(coeffs):
#         slope, offset = coeffs
#         angle2 = slope * np.arange(len(angle)) + offset
#         errors = (angle - angle2) % (2 * np.pi)
#         leastsq = np.sum(errors**2 * weights)
#         # print(slope, offset, leastsq)
#         return leastsq
#
#     res = scipy.optimize.minimize(fun, x0)
#     slope = res.x[0]
#     offset = slope * len(signal) / (2 * np.pi)
#
#     angle2 = res.x[0] * np.arange(len(angle)) + res.x[1]
#     plt.plot(angle)
#     plt.plot(angle2)
#     plt.show()
#
#     return -offset


# def interpolate_polyfit(corr, peak, n=2, deg=2):
#     rel = np.arange(-n, n+1)
#
#     weights = np.abs(rel)**2+1
#     coef = np.polyfit(rel, np.sqrt(corr)[peak + rel], deg, w=weights)
#     der = np.polyder(coef)
#     roots = np.roots(der)
#
#     return roots[0]


# def interpolate_com(corr, peak, n=3):
#     rel = np.array(np.arange(-n, n+1))
#     noms = (rel + n + 1) * corr[peak + rel]
#     nom = np.sum(noms)
#     den = np.sum(corr[peak + rel])
#     offset = nom / den - n - 1
#     return float(np.real(offset)) * 2
