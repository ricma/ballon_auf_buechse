"""
Read in an audio recording of one of the ballons, do a Fourier transform

See Also
--------
[[https://docs.python.org/3/library/aifc.html]]
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

def analyse(filename, fig=None):

    integers = np.memmap(filename, dtype='h', mode='r')[:-1000]
    t = np.arange(len(integers)) / params.framerate

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
    a_c = np.fft.fft(integers)[:len(t) // 2]
    a_c_smooth = np.fft.fft(smooth)[:len(t) // 2]

    a_c_smooth2 = scipy.signal.savgol_filter(
        a_c_smooth,
        window_length=55,
        polyorder=5,
        mode="nearest")
    w = scipy.signal.windows.hann(45, sym=True)
    a_c_smooth2 = np.convolve(5 * w / w.sum(), a_c_smooth2, mode="same")

    nu = nu[:len(t) // 2]

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
        ax_f.set_ylim(1e5, 1e8)

        fig.tight_layout()

    else:
        ax, ax_f = fig.get_axes()

    frac = 30
    slc = slice(None, len(t) // frac)
    plot_1 = ax.plot(t[slc] * 1000, integers[slc],
            zorder=-10)[0]
    plot_2 = ax.plot(t[slc] * 1000, smooth[slc])[0]

    # Only plot low frequencies
    frac = 35
    ax_f.plot(nu[:len(nu) // frac],
              np.abs(a_c[:len(nu) // frac]),
              alpha=0.2, zorder=-10,
              color=plot_1.get_color())
    # ax_f.plot(nu[:len(nu) // frac],
    #           np.abs(a_c_smooth[:len(nu) // frac]),
    #           alpha=0.6, zorder=-5,
    #           color=plot_2.get_color())
    ax_f.plot(nu[:len(nu) // frac],
              np.abs(a_c_smooth2[:len(nu) // frac]),
              color=plot_2.get_color())

    return fig


# filename = "ballon_auf_Dose.aiff"
# filename =
fig = None
for filename in [
        "ballon_auf_Dose_2.wav",
        "ballon_auf_Dose_3.wav",
        "ballon_auf_Dose_4.wav"]:

    fig = analyse("audio/" + filename, fig=fig)

filename = "pictures/ballon_auf_Dose.png"
plt.savefig(filename, transparent=True)
print("[[file:{0}]]".format(filename))
plt.show()
