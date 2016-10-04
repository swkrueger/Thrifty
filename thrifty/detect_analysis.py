"""Like detect.py, but plots stuff."""

from __future__ import division
from __future__ import print_function

import argparse
import sys
import re
from collections import namedtuple

from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT
from matplotlib.figure import Figure

import numpy as np

from PyQt4 import QtGui as qt
from PyQt4 import QtCore

from thrifty.settings import load_args
from thrifty.signal_utils import Signal, compute_fft, compute_ifft
from thrifty import block_data
from thrifty import detect
from thrifty import carrier_detect
from thrifty import soa_estimator
from thrifty import util
from thrifty.setting_parsers import normalize_freq_range


def _time_shift(samples, shift):
    freqs = np.fft.fftfreq(len(samples))
    fft_shift = np.exp(2j * np.pi * shift * freqs)
    fft = compute_fft(samples)
    return compute_ifft(fft * fft_shift)


DetectorResult = namedtuple('DetectorResult', [
    'detected', 'result', 'unsynced', 'synced', 'corr'])


class ForcibleDetector(object):
    def __init__(self, settings, force_carrier=False, force_corr=False):
        override = {}
        if force_carrier:
            override['carrier_thresh'] = (0, 0, 0)
        if force_corr:
            override['corr_thresh'] = (0, 0, 0)
        dsettings = settings._replace(**override)

        self.detector = detect.Detector(dsettings, yield_data=True)

    def __call__(self, timestamp, block_idx, block):
        detected, result, shifted_fft, corr = self.detector.detect(
            timestamp, block_idx, block)

        return DetectorResult(detected,
                              result,
                              block,
                              shifted_fft.ifft,
                              corr)


class Plotter(object):
    def __init__(self, detection, settings, sample_rate):
        self.result = detection.result
        self.unsynced = detection.unsynced
        self.synced = detection.synced
        self.corr = detection.corr
        self.settings = settings
        self.sample_rate = sample_rate

    def plot_sample_histogram(self, ax):
        """Plot sample value histogram."""
        raw = block_data.complex_to_raw(self.unsynced)
        hist = [0] * 256
        for value in raw:
            hist[value] += 1
        ax.plot(np.array(hist) / len(self.unsynced) / 2 * 100)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 255)
        ax.set_xlabel('Raw byte value')
        ax.set_ylabel('Occurrence (%)')
        ax.set_title('Sample value histogram')
        ax.grid()

    def _plot_samples(self, signal, ax, mag, real, imag, rms):
        if mag:
            ax.plot(np.abs(signal), label='Mag')
        if real:
            ax.plot(np.real(signal), label='Real')
        if imag:
            ax.plot(np.imag(signal), label='Imag')
        if rms:
            ax.axhline(signal.rms, label='RMS', linestyle='--')
        ax.legend()
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        # ax2 = ax.twiny()
        # ax2.set_xlim(0, len(signal) / self.sample_rate * 1e3)
        # ax2.set_xlabel('Time (ms)')
        ax.grid()

    def plot_unsynced_iq(self, ax, real=True, imag=True):
        """Plot samples (I/Q)."""
        self._plot_samples(self.unsynced, ax, False, real, imag, False)
        ax.set_title('Unsynchronised samples (I/Q)')

    def plot_unsynced_mag(self, ax, rms=True):
        """Plot samples (magnitude)."""
        self._plot_samples(self.unsynced, ax, True, False, False, rms)
        ax.set_title('Unsynchronised samples (magnitude)')

    def plot_synced_iq(self, ax, real=True, imag=True):
        """Plot frequency-compensated samples (I/Q)."""
        self._plot_samples(self.synced, ax, False, real, imag, False)
        ax.set_title('Frequency-compensated samples (I/Q)')

    def plot_synced_mag(self, ax, rms=True):
        """Plot frequency-compensated samples (magnitude)."""
        self._plot_samples(self.synced, ax, True, False, False, rms)
        ax.set_title('Frequency-compensated samples (magnitude)')

    def _scaled_ook_template(self, signal):
        ook_template = self.settings.template - np.min(ook_template)
        ook_template *= signal.rms / Signal(ook_template).rms
        return ook_template

    def plot_template_overlay(self, ax, zoom='no',
                              padding=0, zoom_length=100):
        """Plot positioning signal with template overlayed on top."""
        start = self.result.corr_info.sample
        stop = start + len(self.settings.template)
        offset = self.result.corr_info.offset

        padded_start = max(0, start - padding)
        padded_stop = min(len(self.synced), stop + padding)

        cut = self.synced[padded_start:padded_stop]
        ook_template = self._scaled_ook_template(cut)

        ax.plot(np.arange(len(ook_template)) + start - padded_start + offset,
                ook_template,
                label='Template')

        if zoom == 'no':
            title_suffix = ' (full)'
        elif zoom == 'start':
            ax.set_xlim(0, zoom_length)
            title_suffix = ' (start)'
        elif zoom == 'center':
            center = len(cut) / 2.
            ax.set_xlim(center - zoom_length / 2., center + zoom_length / 2.)
            title_suffix = ' (center)'
        elif zoom == 'end':
            ax.set_xlim(len(cut) - zoom_length, len(cut))
            title_suffix = ' (end)'
        else:
            raise ValueError("invalid value for 'zoom'")

        ax.set_title('Template overlay' + title_suffix)
        self._plot_samples(cut, ax, True, False, False, False)

    def _plot_spectrum(self, fft, ax, scaled=False, power=False,
                       db_scale=False, plot_thresh=True, plot_noise=True,
                       **kwargs):
        bins = np.arange(len(fft)) - (len(fft) // 2)

        if scaled:
            y_scale = np.sqrt(1 / (len(fft) * self.sample_rate))
            x_scale = self.sample_rate / len(fft) / 1e3
        else:
            y_scale = 1
            x_scale = 1

        if db_scale:
            transf = lambda v: 20 * np.log10(v * y_scale)
        elif power:
            transf = lambda v: (v * y_scale)**2
        else:
            transf = lambda v: v * y_scale

        fft_mag = np.abs(fft)
        y = np.fft.fftshift(fft_mag)
        ax.plot(bins * x_scale, transf(y), **kwargs)

        if plot_thresh:
            threshold = carrier_detect._calculate_threshold(
                fft_mag, self.settings.carrier_thresh,
                self.result.carrier_info.noise)
            ax.axhline(transf(threshold), label='Threshold', linestyle='--')
        if plot_noise:
            ax.axhline(transf(self.result.carrier_info.noise),
                       label='Noise', linestyle=':')

        if scaled:
            ax.set_xlabel('Frequency (kHz)')
            if db_scale:
                ax.set_ylabel('Power/frequency (dB/Hz)')
            elif power:
                ax.set_ylabel('Power/frequency')
            else:
                ax.set_ylabel('Magnitude/frequency')
        else:
            ax.set_xlabel('FFT bin')
            if db_scale:
                ax.set_ylabel('Power/bin (dB)')
            elif power:
                ax.set_ylabel('Power/bin')
            else:
                ax.set_ylabel('Magnitude/bin')

        ax.set_xlim(bins[0] * x_scale, bins[-1] * x_scale)
        ax.grid()

        # if scaled:
        #     freqs = np.fft.fftfreq(len(fft), 1. / self.sample_rate)
        #     freqs_shifted = np.fft.fftshift(freqs)
        #     np.testing.assert_allclose(freqs_shifted, bins * x_scale)

    def _plot_fft_window(self, ax, zoom_to_window=False, zoom_padding=10):
        window = self.settings.carrier_window
        if window:
            N = self.settings.block_len
            start_idx, stop_idx = carrier_detect.fft_range_index(
                window[0], window[1], N)
            start = util.fft_bin(start_idx, N)
            stop = util.fft_bin(stop_idx, N)
            if zoom_padding > 0 or not zoom_to_window:
                ax.axvline(start, linestyle='-.')
                ax.axvline(stop, linestyle='-.')
            if zoom_to_window:
                ax.set_xlim(start - zoom_padding, stop + zoom_padding)

    def plot_synced_fft(self, ax):
        """Plot the FFT of the synchronized signal."""
        self._plot_spectrum(self.synced.fft, ax,
                            scaled=False, power=False, db_scale=False)

    def plot_unsynced_fft(self, ax, zoom_to_window=False, zoom_padding=10):
        """Plot the FFT of the unsynchronized signal."""
        self._plot_spectrum(self.unsynced.fft, ax,
                            scaled=False, power=False, db_scale=False,
                            label='FFT')
        self._plot_fft_window(ax, zoom_to_window, zoom_padding)
        ax.legend(loc='best')
        ax.set_title('Unsynchronised FFT' +
                     ' (window)' if zoom_to_window else '')

    def plot_unsynced_fft_window(self, ax, zoom_padding=10):
        """Plot the carrier detection window within the FFT."""
        self.plot_unsynced_fft(ax, True, zoom_padding)

    def plot_synced_psd(self, ax):
        """Plot the estimated PSD after frequency compensation."""
        self._plot_spectrum(self.synced.fft, ax,
                            scaled=True, power=True, db_scale=True)

    def plot_unsynced_psd(self, ax):
        """Plot the estimated power spectral density."""
        self._plot_spectrum(self.unsynced.fft, ax,
                            scaled=True, power=True, db_scale=True)

    def plot_corr(self, ax):
        """Plot correlation signal."""
        start, stop = soa_estimator.calculate_window(
            self.settings.block_len,
            self.settings.history_len,
            len(self.settings.template))

        corr_mag = np.abs(self.corr)
        ax.plot(corr_mag, label='Corr')
        ax.axvline(start, linestyle='--')
        ax.axvline(stop, linestyle='--')
        peak_pos = self.result.corr_info.sample + self.result.corr_info.offset
        ax.axvline(peak_pos, linestyle='--')
        ax.axhline(self.result.corr_info.noise, label='Noise', linestyle='--')
        # TODO: plot threshold
        ax.set_xlabel('Delay (samples)')
        ax.set_ylabel('Corr magnitude')

    def plot_corr_zoomed(self, ax, zoom_length=400):
        """Plot correlation signal, zoomed to exhibit multipath effects."""
        peak_idx = self.result.corr_info.sample
        offset = self.result.corr_info.offset
        start = max(0, peak_idx - zoom_length // 2)
        stop = min(len(self.corr), peak_idx + zoom_length // 2)
        corr_mag = np.abs(self.corr[start:stop])
        ax.plot(np.arange(len(corr_mag)) + start, corr_mag, label='Corr')
        ax.axvline(peak_idx + offset, linestyle='--')
        ax.set_title('Cross-correlation with template')
        ax.grid()

    def _generate_autocorr(self, indices, shift=0):
        template = self.settings.template
        autocorr = []
        for index in indices:
            x = abs(index)
            right = -x if x != 0 else None
            autocorr.append(np.sum(template[x:] * template[:right]))
        autocorr = np.array(autocorr)
        if shift != 0:
            autocorr = np.real(_time_shift(autocorr, shift))
        return autocorr

    def _plot_corr_interpolation(self, ax, corr_mag, peak_idx, **args):
        y1, y2, y3 = (corr_mag[peak_idx-1],
                      corr_mag[peak_idx],
                      corr_mag[peak_idx+1])
        y1, y2, y3 = np.log(y1), np.log(y2), np.log(y3)  # Gaussian
        offset = 0.5 * (y3 - y1) / (2 * y2 - y1 - y3)
        a = (y1-y2) / ((offset+1)**2 - offset**2)
        y0 = y1 - a*(offset+1)**2

        x = np.linspace(offset-2, offset+2, 50)
        y = a*(x - offset)**2 + y0
        y = np.exp(y)  # Gaussian
        ax.plot(x, y, **args)

    def plot_corr_peak_interpol(self, ax, length=10):
        """Plot the correlation peak and the interpolation function."""
        peak_sample = self.result.corr_info.sample
        offset = self.result.corr_info.offset
        start = max(0, peak_sample - length // 2)
        stop = min(len(self.corr), peak_sample + length // 2 + 1)

        peak_idx = peak_sample - start
        corr_mag = np.abs(self.corr[start:stop])
        ax.plot(np.arange(len(corr_mag)) - peak_idx, corr_mag,
                marker='.', label='Xcorr')
        # TODO: plot noise and threshold

        ax.axvline(offset, linestyle='--')

        self._plot_corr_interpolation(ax, corr_mag, peak_idx,
                                      label='Interpolation')

        ax.set_xlabel('Offset relative to peak (samples)')
        ax.set_ylabel('Correlation magnitude')
        ax.legend()

    def plot_corr_peak_shifted(self, ax, length=10):
        """Plot the offset-compensated xcorr peak and the autocorr peak."""
        peak_sample = self.result.corr_info.sample
        offset = self.result.corr_info.offset
        start = max(0, peak_sample - length // 2)
        stop = min(len(self.corr), peak_sample + length // 2 + 1)

        corr_mag = np.abs(_time_shift(self.corr[start:stop], offset))
        peak_idx = peak_sample - start
        indices = np.arange(len(corr_mag)) - peak_idx
        ax.plot(indices, corr_mag, marker='.', label='Shifted xcorr')

        autocorr = self._generate_autocorr(indices)
        # pidx = np.argmax(autocorr)
        # scale = (autocorr[pidx] * (1-offset) + autocorr[pidx+1] * offset
        #          if offset > 0 else
        #          autocorr[pidx] * (1+offset) + autocorr[pidx-1] * -offset)
        scale = corr_mag[peak_idx] / autocorr[peak_idx]
        ax.plot(indices, autocorr * scale, marker='.', label='Autocorr')
        print(autocorr)

        ax.set_xlabel('Offset relative to peak (samples)')
        ax.set_ylabel('Correlation magnitude')
        ax.legend()
        ax.grid()

    def plot_corr_peak_shifted_autocorr(self, ax, length=20):
        """Plot the xcorr peak and the shifted autocorr peak."""
        peak_sample = self.result.corr_info.sample
        offset = self.result.corr_info.offset
        start = max(0, peak_sample - length // 2)
        stop = min(len(self.corr), peak_sample + length // 2 + 1)

        peak_idx = peak_sample - start
        corr_mag = np.abs(self.corr[start:stop])
        ax.plot(np.arange(len(corr_mag)) - peak_idx, corr_mag,
                marker='.', label='Xcorr')

        indices = np.arange(-8, 9)
        autocorr = self._generate_autocorr(indices, -offset)
        autocorr_peak = np.argmax(autocorr)
        scale = ((np.sum(corr_mag[peak_idx-1:peak_idx+2]))
                 / (np.sum(autocorr[autocorr_peak-1:autocorr_peak+2])))
        ax.plot(indices, autocorr * scale, marker='.',
                label='Shifted autocorr')
        ax.legend()
        ax.grid()

    def plot_overview(self, fig):
        """Plot hist, mag_synced, fft_window, corr."""
        self.plot_sample_histogram(fig.add_subplot(221))
        self.plot_synced_mag(fig.add_subplot(222))
        self.plot_unsynced_fft_window(fig.add_subplot(223))
        self.plot_corr(fig.add_subplot(224))
        fig.suptitle('Overview')

    def plot_time(self, fig):
        """Plot iq, mag_synced."""
        self.plot_unsynced_iq(fig.add_subplot(211))
        self.plot_synced_mag(fig.add_subplot(212))
        fig.suptitle('Time-domain')

    def plot_overlays(self, fig):
        """Plot template overlay (full, start, center, and end)."""
        self.plot_template_overlay(fig.add_subplot(221), 'no')
        self.plot_template_overlay(fig.add_subplot(222), 'start')
        self.plot_template_overlay(fig.add_subplot(223), 'center')
        self.plot_template_overlay(fig.add_subplot(224), 'end')
        fig.suptitle('Template overlays')

    def plot_spectra(self, fig):
        """Plot fft, psd_synced."""
        self.plot_unsynced_fft(fig.add_subplot(221), zoom_to_window=True)
        self.plot_synced_psd(fig.add_subplot(222))
        # TODO: unsynced_fft_peak (i.t.o. bins)
        # TODO: synced_fft_peak
        fig.suptitle('Spectra')

    def plot_corrs(self, fig):
        """Plot corr, corr_zoomed, corr_interpol, corr_shifted."""
        self.plot_corr(fig.add_subplot(221))
        self.plot_corr_zoomed(fig.add_subplot(222))
        self.plot_corr_peak_interpol(fig.add_subplot(223))
        self.plot_corr_peak_shifted(fig.add_subplot(224))
        fig.suptitle('Correlation')


_PLOT_COMMAND_STRINGS = {
    'hist': Plotter.plot_sample_histogram,
    'iq': Plotter.plot_unsynced_iq,
    'mag': Plotter.plot_unsynced_mag,
    'iq_synced': Plotter.plot_synced_iq,
    'mag_synced': Plotter.plot_synced_mag,
    'template': Plotter.plot_template_overlay,
    'fft': Plotter.plot_unsynced_fft,
    'fft_window': Plotter.plot_unsynced_fft_window,
    'fft_synced': Plotter.plot_synced_fft,
    'psd': Plotter.plot_unsynced_psd,
    'psd_synced': Plotter.plot_synced_psd,
    'corr': Plotter.plot_corr,
    'corr_zoomed': Plotter.plot_corr_zoomed,
    'corr_interpol': Plotter.plot_corr_peak_interpol,
    'corr_shifted': Plotter.plot_corr_peak_shifted,
}


_FIGURE_COMMAND_STRINGS = {
    'overview': Plotter.plot_overview,
    'time': Plotter.plot_time,
    'overlays': Plotter.plot_overlays,
    'spectra': Plotter.plot_spectra,
    'corrs': Plotter.plot_corrs,
}


def _plot(fig, plotter, cmd):
    if cmd in _PLOT_COMMAND_STRINGS:
        ax = fig.add_subplot(111)
        _PLOT_COMMAND_STRINGS[cmd](plotter, ax)
    elif cmd in _FIGURE_COMMAND_STRINGS:
        _FIGURE_COMMAND_STRINGS[cmd](plotter, fig)


class DetectionViewer(qt.QWidget):
    def __init__(self, detections, cmds, settings, sample_rate, parent=None):
        qt.QWidget.__init__(self, parent)

        self.plotters = [Plotter(detection, settings, sample_rate)
                         for detection in detections]
        self.detections = detections

        self.block_selector = qt.QTabBar()
        for detection in detections:
            title = str(detection.result.block)
            self.block_selector.addTab(title)
        self.cmds = cmds
        self.cmd_selector = qt.QTabBar()
        for cmd in cmds:
            self.cmd_selector.addTab(cmd)

        self.block_selector.currentChanged.connect(self.plot)
        self.cmd_selector.currentChanged.connect(self.plot)

        self.fig = Figure(frameon=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, parent)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.setSizePolicy(qt.QSizePolicy.Expanding,
                                  qt.QSizePolicy.Expanding)
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.summary_liner = detect.SummaryLineFormatter(sample_rate,
                                                         settings.block_len,
                                                         add_dt=False)
        self.summary_line = qt.QLabel()
        self.summary_line.setAlignment(QtCore.Qt.AlignHCenter)

        vbox = qt.QVBoxLayout()
        vbox.addWidget(qt.QLabel('Detect Analysis'))
        vbox.addWidget(self.block_selector)
        vbox.addWidget(self.cmd_selector)
        vbox.addWidget(self.summary_line)
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.toolbar)
        self.setLayout(vbox)

        self.plot()
        self.canvas.setFocus()

    def plot(self):
        detection_idx = self.block_selector.currentIndex()
        if detection_idx == -1:
            return
        plotter = self.plotters[detection_idx]
        detection = self.detections[detection_idx]

        cmd = self.cmds[self.cmd_selector.currentIndex()]
        self.fig.gca().cla()
        self.fig.clf(keep_observers=False)
        _plot(self.fig, plotter, cmd)
        self.fig.set_facecolor('none')
        self.canvas.draw()

        summary_text = self.summary_liner(detection.detected, detection.result)
        self.summary_line.setText(summary_text)

    def on_key_press(self, event):
        # implement the default mpl key press events described at
        # http://matplotlib.org/users/navigation_toolbar.html
        key_press_handler(event, self.canvas, self.toolbar)


def parse_range_list(string):
    """Parse a comma-separated list of integer ranges.

    Ranges may be open-ended.

    Examples
    --------
    >>> parse_range_list("1")
    [(1, 1)]
    >>> parse_range_list("1-100")
    [(1, 100)]
    >>> parse_range_list("1-")
    [(1, None)]
    >>> parse_range_list("-100")
    [(None, 100)]
    >>> parse_range_list("1-5, 20-30")
    [(1, 5), (20, 30)]
    """
    ranges = []
    for range_string in string.split(','):
        m = re.match(r'^\s*(\d+)?(?:(-)?(\d+)?)\s*$', range_string)
        if not m:
            raise ValueError("'" + string + "' is not a range of numbers.")
        start = m.group(1)
        end = start if not m.group(2) else m.group(3)
        if start is not None:
            start = int(start)
        if end is not None:
            end = int(end)
        ranges.append((start, end))
    return ranges


def block_in_range(block_idx, ranges):
    for start, end in ranges:
        if ((start is None or block_idx >= start) and
           (end is None or block_idx <= end)):
            return True
    return False


def _main():
    # pylint: disable=too-many-locals

    plot_cmd_descriptions = ["  {:14s} {}".format(c, f.__doc__.split('\n')[0])
                             for c, f in
                             sorted(_PLOT_COMMAND_STRINGS.iteritems())]

    figure_cmd_descriptions = ["  {:14s} {}".format(c, f.__doc__.split('\n')[0])
                             for c, f in
                             sorted(_FIGURE_COMMAND_STRINGS.iteritems())]

    description = "{}\n\nplot commands:\n{}\n\nfigure commands:\n{}".format(
                  __doc__,
                  '\n'.join(plot_cmd_descriptions),
                  '\n'.join(figure_cmd_descriptions))

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('input',
                        type=argparse.FileType('rb'), default='data.card',
                        help="input data ('-' streams from stdin)")
    parser.add_argument('--raw', dest='raw', action='store_true',
                        help="input data is raw binary data")
    parser.add_argument('-f', '--force_cardet', action='store_true',
                        help="force carrier detection")
    parser.add_argument('-F', '--force_corrdet', action='store_true',
                        help="force correlation peak detection")
    parser.add_argument('-i', '--blocks', type=parse_range_list,
                        help="indices of blocks to plot", default='0-')
    parser.add_argument('-m', '--max', type=int,
                        help="maximum number of blocks", default=20)
    parser.add_argument('-p', '--plot', type=str,
                        help="what to plot",
                        default="overview,time,overlays,spectra,corrs")
    parser.add_argument('--export', type=str, nargs='?', const='plot',
                        help="export plots to .PDF files "
                             "with the given prefix")

    setting_keys = ['sample_rate', 'block_size', 'block_history',
                    'carrier_window', 'carrier_threshold',
                    'corr_threshold', 'template']
    config, args = load_args(parser, setting_keys)

    window = normalize_freq_range(config.carrier_window,
                                  config.sample_rate / config.block_size)

    if args.raw:
        blocks = block_data.block_reader(args.input, config.block_size,
                                         config.block_history)
    else:
        blocks = block_data.card_reader(args.input)

    cmds = args.plot.split(',')
    template = np.load(config.template)
    settings = detect.DetectorSettings(block_len=config.block_size,
                                       history_len=config.block_history,
                                       carrier_len=len(template),
                                       carrier_thresh=config.carrier_threshold,
                                       carrier_window=window,
                                       template=template,
                                       corr_thresh=config.corr_threshold)
    detector = ForcibleDetector(settings,
                                force_carrier=args.force_cardet,
                                force_corr=args.force_corrdet)

    detections = []
    for timestamp, block_idx, block in blocks:
        if not block_in_range(block_idx, args.blocks):
            continue
        print("Generating plotter for block #{}".format(block_idx))
        detection = detector(timestamp, block_idx, block)
        if detection.detected:
            detections.append(detection)
            if len(detections) >= args.max:
                break
        else:
            print("Skipping block #{}".format(block_idx))

    if args.export:
        for detection in detections:
            plotter = Plotter(detection, settings, config.sample_rate)
            for cmd in cmds:
                filename = "{}_{}_{}.pdf".format(args.export,
                                                 detection.result.block,
                                                 cmd)
                print("Exporting", filename)
                fig = Figure()
                FigureCanvas(fig)
                _plot(fig, plotter, cmd)
                fig.set_tight_layout(True)
                fig.savefig(filename)
    else:
        app = qt.QApplication(sys.argv)
        ui = DetectionViewer(detections, cmds, settings, config.sample_rate)
        ui.show()
        app.exec_()


if __name__ == '__main__':
    _main()
