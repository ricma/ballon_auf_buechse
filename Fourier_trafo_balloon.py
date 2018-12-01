"""
Read in an audio recording of one of the ballons, do a Fourier transform

See Also
--------
[[https://docs.python.org/3/library/aifc.html]]
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import pyaudio
import aifc


class ToneCallback:
    """
    Register as a callback to play a tone
    """
    def __init__(
            self, ax,
            volume=1.0, fs=100000, duration=5, **kwargs):
        """
        Parameters
        ----------
        ax : Subplot
            Subplot with spectrum
        volume : double between [0, 1]
            Volume to play sound with.
        fs : integer
            Sampling rate in Hz
        duration: float
            Time in seconds the tone is played
        bind : boolean, optional
            If True, then call fig.canvas.mpl_connect on self.
        """
        self.ax = ax
        self.volume = volume
        self.fs = fs
        self.duration = duration

        self.p = pyaudio.PyAudio()

        if kwargs.get("bind", True):
            fig = self.ax.get_figure()
            fig.canvas.mpl_connect("button_press_event", self)

    def __call__(self, event):
        """
        Play tone
        """
        if not event.inaxes is self.ax:
            return
        if event.button != 1:
            return
        if plt.get_current_fig_manager().toolbar.mode != '':
            return

        frequency = event.xdata
        self.play_tone(frequency)

    def play_tone(self, freq):
        """
        Play a tone with the given frequency

        Parameters
        ----------
        freq : float
            Frequency to play

        See Also
        --------
        [[https://stackoverflow.com/a/27978895/2959456]]
        """
        print("Playing {0:1.3f} Hz".format(freq))
        # generate samples, note conversion to float32 array
        #
        samples = (np.sin(
            2 * np.pi * np.arange(self.fs * self.duration) *
            freq / self.fs)).astype(np.float32)

        # for paFloat32 sample values must be in range [-1.0, 1.0]
        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.fs,
            output=True)

        stream.write(self.volume * samples)
        stream.stop_stream()
        stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.p.terminate()


def analyse(filename, fig=None):
    """
    Analyse spectrum of raw audio data
    """
    framerate = 44100
    integers = np.memmap(filename, dtype='h', mode='r')

    # The last bit is looking a bit weird
    integers = integers[:-1000]

    t = np.arange(len(integers)) / framerate

    # remove noise from the signal
    smooth = scipy.signal.savgol_filter(
        integers,
        window_length=1 + 2 * (len(t) // 2000),
        polyorder=7,
        mode="nearest")
    w = scipy.signal.windows.hann(1 + 2 * (len(t) // 2000), sym=True)
    smooth = np.convolve(w / w.sum(), smooth, mode="same")

    # do the fourier transform
    #
    # In the Fourier trafo the exponent is i⋅w_k⋅t_j.
    # In the FFT we have 2π⋅i⋅j⋅k / N:
    # i⋅w_k⋅t_j = 2π⋅i⋅f_k⋅t_j = 2π⋅i⋅j⋅f_k / frate = 2π⋅i⋅j⋅k / N
    # Hence, the f_k are given by: f_k = k ⋅ frate / N
    # where `frate` is the framerate
    nu = np.arange(len(t)) / t.max()

    # only plot the positive frequencies
    nu = nu[:len(t) // 2]
    a_c = np.fft.fft(integers)[:len(t) // 2]
    a_c_smooth = np.fft.fft(smooth)[:len(t) // 2]

    a_c_smooth2 = scipy.signal.savgol_filter(
        a_c_smooth,
        window_length=55,
        polyorder=5,
        mode="nearest")
    w = scipy.signal.windows.hann(45, sym=True)
    a_c_smooth2 = np.convolve(5 * w / w.sum(), a_c_smooth2, mode="same")

    if fig is None:
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(121)
        ax.set_xlabel(r"$t / \mathrm{ms}$")
        ax.set_ylabel(r"$g(t)$")
        ax.set_yticks([])

        ax_f = fig.add_subplot(122)
        ax_f.set_xlabel(r"$\nu / \mathrm{Hz}$")
        ax_f.set_ylabel(r"$|g(\nu)|$")
        ax_f.set_yscale("log")

        fig.tight_layout()

    else:
        ax, ax_f = fig.get_axes()

    t_max_plot = 0.2
    slc = slice(None, np.argmin(np.abs(t - t_max_plot)))
    plot_1 = ax.plot(t[slc] * 1000, integers[slc],
            zorder=-10)[0]
    plot_2 = ax.plot(t[slc] * 1000, smooth[slc])[0]

    # Only plot low frequencies
    nu_min = 200
    nu_max = 650
    where = slice(*np.nonzero(
        (nu_min <= nu) & (nu <= nu_max))[0][[0, -1]])
    ax_f.plot(nu[where],
              np.abs(a_c[where]),
              alpha=0.2, zorder=-10,
              color=plot_1.get_color())
    ax_f.plot(nu[where],
              np.abs(a_c_smooth2[where]),
              color=plot_2.get_color())

    return fig


if __name__ == "__main__":

    fig = None
    for filename in [
            # "ballon_auf_Dose_2.wav",
            "ballon_auf_Dose_3.wav",
            "ballon_auf_Dose_4.wav"
            ]:

        fig = analyse("audio/" + filename, fig=fig)

    filename = "pictures/ballon_auf_Dose.png"
    plt.savefig(filename, transparent=True)
    print("[[file:{0}]]".format(filename))

    ax_frequency = fig.get_axes()[1]
    with ToneCallback(ax_frequency) as tc:
        plt.show()
