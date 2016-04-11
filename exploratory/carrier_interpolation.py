#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class Settings:
    sample_rate = 2.2e6

def dirichlet_kernel(k, settings):
    """Calculate sinc / dirichlet weights.
    
    From: https://en.wikipedia.org/wiki/Discrete_Fourier_transform:
    DFT of a rect function is a sinc-like function (Dirichlet kernel)
    """
    N, W = settings.data_len, settings.code_len
    k = np.array(k, dtype=np.float64)
    f = lambda k: np.sin(np.pi*W*k/N)/W/np.sin(np.pi*k/N)
    # sinc approx (easier for mathematical analysis)
    # f = lambda k: np.sin(np.pi*W*k/N)/(np.pi*W*k/N)
    return np.piecewise(k, [k == 0, k != 0], [lambda x: 1, f])


def dirichlet_fit(k, a, t):
    N, W = 8192, 2085
    k = np.array(k, dtype=np.float64)
    f = lambda k: a * np.abs(np.sin(np.pi*W*(k-t)/N)/W/np.sin(np.pi*(k-t)/N))
    return np.piecewise(k, [k-t == 0, k-t != 0], [lambda x: a, f])


def sinc_fit(k, a, t):
    N, W = 8192, 2085
    k = np.array(k, dtype=np.float64)
    f = lambda k: a * np.abs(np.sin(np.pi*W*(k-t)/N)/W/(np.pi*(k-t)/N))
    result = np.piecewise(k, [k - t == 0, k - t != 0], [lambda x: a, f])
    # print "sinc_fit", k, a, t, result
    return result


def testInterpolate():
    carrier_freq = -80.060e3 # 80029.296875 # 80.109e3
    s = Settings()
    N, W = 8192, 2085
    s.data_len = N
    s.code_len = W

    # code_samples = np.load(open('../template.npy', 'r'))
    # code_samples += 1
    # code_samples /= 2
    # plt.plot(code_samples)
    # plt.show()
    
    c = np.exp(2j * np.pi * carrier_freq * np.arange(W) / s.sample_rate)
    # c = code_samples * c
    # phase_offset = np.exp(2j * np.pi / 3.333)
    # c *= phase_offset
    y = np.concatenate([c, np.zeros(N - len(c))])
    # y += np.random.normal(0, 0.5, len(y))
    
    fft = np.fft.fft(y)
    fft_mag = np.abs(fft)

    peak = np.argmax(fft_mag)

    # center of mass
    # Werk slegs vir N=2 -- verstaan nie hoekom nie
    # Bewys hoekom "center of mass" nie werk nie (waarskynlik omdat sinc nie lineer is nie)
    # Bewys hoekom "center of mass gedeel deur d" werk vir N=2
    n = 2
    rel = np.array(np.arange(-n, n+1))
    d = dirichlet_kernel(rel, s)
    print 'd', d
    print 'fft_mag', fft_mag[peak + rel]
    noms = (rel + n + 1) * fft_mag[peak + rel] / np.abs(d)
    nom = np.sum(noms)
    den = np.sum(fft_mag[peak + rel] / np.abs(d))
    print 'noms', noms / den
    offset = nom / den - n - 1
    offset *= 2 # not sure why
    print 'offset', offset
    peak_freq = peak * 2.2e6 / N
    adjusted_freq = (peak + offset) * 2.2e6 / N
    print 'offset', carrier_freq, peak_freq, offset * 2.2e6 / N, adjusted_freq, adjusted_freq - carrier_freq

    # ander opsies:
    # - skuif fft totdat dit die beste pas
    # - werk dirichlet kernel / sinc uit met verskillende "frequency shifts",
    #   kyk waar pas dit die beste.

    # plt.plot(noms / den)
    # plt.show()

    # perform fraction frequency shift in the frequency domain by performing
    # This is an approximation (sein gaan vervorm word!).
    # TODO: we probably screw up phase information
    # TODO: Is this the same as linear interpolation between two FFT samples?
    # Using proper interpol interpolation might be better
    #   (see https://www.mathworks.com/matlabcentral/answers/78746-how-do-i-freqency-shift-a-signal-by-a-fractional-amount-using-ifft-and-fft)
    # TODO: (kontrolleer met wiskunde -- kyk na fft van freq_shift. Hierdie gewigte behoort 'n 
    # vereenvoudiging te wees van die gewigte van freq_shift se fft
    # (hulle behoort dieselfde verhouding te he).
    if offset > 0:
        fft2 = (1 - offset) * fft + offset * np.roll(fft, -1) # * (1 + offset)
    else:
        fft2 = (1 + offset) * fft - offset * np.roll(fft, 1) # * (1 - offset)
    fft_mag2 = np.abs(fft2)

    # Check against freq shift in time-domain
    freq_shift = np.exp(-2j * np.pi * offset * np.fft.fftfreq(N))
    y3 = y * freq_shift
    fft3 = np.fft.fft(y3)
    # fft3 = fft3 / np.abs(fft3[peak]) * np.abs(fft[peak]) # scale
    fft_mag3 = np.abs(fft3)

    # y * freq_shift is that same as convolve(Y, FREQ_SHIFT)
    # Check to see whether offset and (1 - offset) is an approximateion of
    # FREQ_SHIFT[-1:2]
    # TODO: werk gewigte uit met DFT vergelyking
    freq_shift_fft = np.fft.fft(freq_shift)
    freq_shift_energy = np.sum(np.abs(freq_shift_fft)**2)
    frac = np.abs(freq_shift_fft)**2 / freq_shift_energy
    if offset > 0:
        print frac[-1], frac[0], offset, (1 - offset)
        print "Freq-domain shift fraction of energy:", frac[-1] + frac[0]
    else:
        print frac[0], frac[1], offset, (1 + offset)
        print "Freq-domain shift fraction of energy:", frac[0] + frac[1]
    # plt.plot(np.abs(freq_shift_fft), '.-')
    # plt.show()

    # Interpolate
    xp = np.arange(len(fft))
    # delta = 1 - o if o >=0 else 1 + o
    fft4 = np.interp(xp + offset, xp, fft.real) + 1j * np.interp(xp + offset, xp, fft.imag)
    # fft4 = fft4 / np.abs(fft4[peak]) * np.abs(fft[peak]) # scale
    fft_mag4 = np.abs(fft4)
    print "fft2 and fft4 are essentially the same:", np.sum(np.abs(fft2 - fft4))

    # Curve fit
    # Kan dalk 'n vereenvoudigde benaderde vergelyking uitwerk deur die Taylor-uitbreiding van sinc te gebruik.
    # Kyk ook na https://stackoverflow.com/questions/22950301/fitting-a-variable-sinc-function-in-python
    n = 3
    xdata = np.array(np.arange(-n, n+1))
    ydata = fft_mag[peak + xdata]
    p0 = (fft_mag[peak], 0)
    print p0
    popt, pcov = curve_fit(dirichlet_fit, xdata, ydata, p0=p0)
    # bounds=([fft_mag[peak]*0.6, -0.5], [fft_mag[peak]*1.6, 0.5])
    fit_ampl, fit_offset = popt
    print "Curve fit:", fit_offset, popt, pcov
    adjusted_freq = (peak + fit_offset) * 2.2e6 / N
    print 'fit offset', carrier_freq, peak_freq, fit_offset * 2.2e6 / N, adjusted_freq, adjusted_freq - carrier_freq
    perr = np.sqrt(np.diag(pcov))
    print 'estimated fit error (1 std dev)', perr

    # Time-domain fit shift
    freq_shift = np.exp(-2j * np.pi * fit_offset * np.fft.fftfreq(N))
    y5 = y * freq_shift
    fft5 = np.fft.fft(y5)
    # fft5 = fft5 / np.abs(fft5[peak]) * np.abs(fft[peak]) # scale
    fft_mag5 = np.abs(fft5)

    # Check energy
    energy_time = np.sum(np.abs(y)**2)
    energy_fft = np.sum(np.abs(fft)**2) / N
    energy_fft2 = np.sum(np.abs(fft2)**2) / N
    energy_fft3 = np.sum(np.abs(fft3)**2) / N
    energy_fft4 = np.sum(np.abs(fft4)**2) / N
    print "Energy:", energy_fft, energy_fft2, energy_fft3, energy_fft4

    rel = np.array(np.arange(-20, 20+1))
    d = dirichlet_kernel(rel, s)
    # print d

    plt.plot(fft_mag, '.-', label="before shift")
    plt.plot(np.real(fft), '.-', label="real")
    plt.plot(np.imag(fft), '.-', label="imag")
    t_offset = np.arange(len(fft_mag)) + offset
    plt.plot(t_offset, fft_mag2, '.-', label="after shift (mean)")
    plt.plot(t_offset, fft_mag3, '.-', label="after shift (time)")
    plt.plot(t_offset, fft_mag4, '.-', label="after shift (lin interpol)")
    plt.plot(t_offset, fft_mag5, '.-', label="after shift fit_offset (time)")
    plt.plot(rel + peak + fit_offset, np.abs(d) * fit_ampl, '.-', label="dirichlet")
    # plt.plot(rel + peak + offset, np.abs(d * fft_mag[peak]), '.-', label="dirichlet")
    plt.plot([peak+offset, peak+offset], [0, fft_mag[peak]])
    plt.xlim([peak-10, peak+10])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    testInterpolate()
