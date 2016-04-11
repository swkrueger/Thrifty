#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.fft import fftshift

def freq_shift(data, delta):
    # f = np.fft.fftfreq(len(data)) # buggy
    # f = np.fft.fftshift(np.fft.fftfreq(len(data)))
    f = np.arange(len(data)) * 1. / len(data) - 0.5
    freq_shift = np.exp(-2j * np.pi * delta * f)
    y = data * freq_shift
    shifted_fft = np.fft.fft(y)
    return shifted_fft

sample_rate = 2.2e6

b = np.load(open('gmf.npy', 'r'))
fft = np.fft.fft(b)

f = np.fft.fftfreq(len(fft), 1.0/sample_rate) / 1000
ff = np.fft.fftshift(f)

fft_mag = np.abs(fft)
fig = plt.figure()
ax = plt.subplot(2, 1, 1, xlim=(-31.5, -27), ylim=(-1000, 1000))
ax.plot(ff, fftshift(fft_mag), marker='.', label="FFT (unshifted)")
ax.plot(ff, fftshift(np.real(fft)), marker='.', label="FFT real (unshifted)")
ax.plot(ff, fftshift(np.imag(fft)), marker='.', label="FFT imag (unshifted)")
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude')

annotation = ax.annotate('gar', xy=(-28, 0))
annotation.set_animated(True)

print len(f)
ang = fftshift(np.angle(fft))[3978:3998]
ax2 = plt.subplot(2, 1, 2, xlim=(-31.5, -27)) #, ylim=(np.min(ang)-2, np.max(ang)+2))
ax2.plot(ff[3978:3998], ang)
plt.xlabel('Phase')
plt.ylabel('Magnitude')

lines = [
    ax.plot(ff, fftshift(fft_mag), marker='.', label="FFT (shifted)")[0],
    ax.plot(ff, fftshift(np.real(fft)), marker='.', label="FFT real (shifted)")[0],
    ax.plot(ff, fftshift(np.imag(fft)), marker='.', label="FFT imag (shifted)")[0],
    annotation,
    ax2.plot(ff[3978:3998], ang)[0],
    ]

frames = 400

def init():
    return lines

def step(i):
    dt = 2.0 * i / frames - 1
    print dt
    shifted = freq_shift(b, dt)
    shifted_abs = np.abs(shifted)
    lines[0].set_ydata(fftshift(shifted_abs))
    lines[1].set_ydata(fftshift(np.real(shifted)))
    lines[2].set_ydata(fftshift(np.imag(shifted)))

    peak = np.argmax(shifted_abs)
    lines[3].set_text('(%.3f kHz, %.0f)' % (f[peak], shifted_abs[peak]))
    lines[3].set_position((f[peak], shifted_abs[peak]))
    ax.set_title("dt = %f" % (dt))

    ang = fftshift(np.angle(shifted))[3978:3998]
    lines[4].set_ydata(ang)

    # TODO: annotate peaks
    # TODO: plot "dt = x"
    return lines

anim = animation.FuncAnimation(fig, step, frames=frames, init_func=init, interval=30)

writer = animation.AVConvWriter()
anim.save('freq_shift_animation.mp4', fps=30, writer=writer)
# anim.save('/tmp/freq_shift_animation.gif', writer='imagemagick', fps=30)

plt.show()

