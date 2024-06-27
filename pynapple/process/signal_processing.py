"""
Signal processing tools for Pynapple.

Contains functionality for signal processing pynapple object; fourier transforms and wavelet decomposition.
"""

import numpy as np
import pynapple as nap
from math import ceil, floor
import json
from scipy.signal import welch
from itertools import repeat

with open('wavelets.json') as f:
    WAVELET_DICT = json.load(f)


def compute_spectrum(sig, fs=None):
    """  
    Performs numpy fft on sig, returns output  
    ..todo: Make sig handle TsdFrame, TsdTensor  

    ----------
    sig : pynapple.Tsd or pynapple.TsdFrame
        Time series.
    fs : float, optional
        Sampling rate, in Hz. If None, will be calculated from the given signal
    """
    if not isinstance(sig, nap.Tsd):
        raise TypeError("Currently compute_fft is only implemented for Tsd")
    if fs is None:
        fs = sig.index.shape[0]/(sig.index.max() - sig.index.min())
    fft_result = np.fft.fft(sig.values)
    fft_freq = np.fft.fftfreq(len(sig.values), 1 / fs)
    return fft_result, fft_freq


def compute_welch_spectrum(sig, fs=None):
    """
    Performs scipy Welch's decomposition on sig, returns output
    ..todo: Make sig handle TsdFrame, TsdTensor

    ----------
    sig : pynapple.Tsd or pynapple.TsdFrame
        Time series.
    fs : float, optional
        Sampling rate, in Hz. If None, will be calculated from the given signal
    """
    if not isinstance(sig, nap.Tsd):
        raise TypeError("Currently compute_fft is only implemented for Tsd")
    if fs is None:
        fs = sig.index.shape[0]/(sig.index.max() - sig.index.min())
    freqs, spectogram = welch(sig.values, fs=fs)
    return spectogram, freqs


def morlet(M=1024, ncycles=1.5, scaling=1.0, precision=8):
    """
    Defines the complex Morelet wavelet kernel

    Parameters
    ----------
    M : int
        Length of the wavelet
    ncycles : float
        number of wavelet cycles to use. Default is 5
    scaling: float
        Scaling factor. Default is 1.5
    precision: int
        Precision of wavelet to use

    Returns
    -------
    np.ndarray
        Morelet wavelet kernel
    """
    x = np.linspace(-precision, precision, M)
    return ((np.pi*ncycles) ** (-0.25)) * np.exp(-x**2 / ncycles) * np.exp(1j * 2*np.pi * scaling * x)

def _check_n_cycles(n_cycles, len_cycles=None):
    """
    Check an input as a number of cycles, and make it iterable.

    Parameters
    ----------
    n_cycles : float or list
        Definition of number of cycles to check. If a single value, the same number of cycles is used for each
        frequency value. If a list or list_like, then should be a n_cycles corresponding to each frequency.
    len_cycles: int, optional
        What the length of `n_cycles` should be, if it's a list.

    Returns
    -------
    iter
        An iterable version of the number of cycles.
    """
    if isinstance(n_cycles, (int, float, np.number)):
        if n_cycles <= 0:
            raise ValueError("Number of cycles must be a positive number.")
        n_cycles = repeat(n_cycles)
    elif isinstance(n_cycles, (tuple, list, np.ndarray)):
        for cycle in n_cycles:
            if cycle <= 0:
                raise ValueError("Each number of cycles must be a positive number.")
        if len_cycles and len(n_cycles) != len_cycles:
            raise ValueError(
                "The length of number of cycles does not match other inputs."
            )
        n_cycles = iter(n_cycles)
    return n_cycles


def _create_freqs(freq_start, freq_stop, freq_step=1):
    """
    Creates an array of frequencies.

    ..todo:: Implement log scaling

    Parameters
    ----------
    freq_start : float
        Starting value for the frequency definition.
    freq_stop: float
        Stopping value for the frequency definition, inclusive.
    freq_step: float, optional
        Step value, for linearly spaced values between start and stop.

    Returns
    -------
    freqs: 1d array
        Frequency indices.
    """
    return np.arange(freq_start, freq_stop + freq_step, freq_step)


def compute_wavelet_transform(sig, fs, freqs, n_cycles=1.5, scaling=1.0, norm="amp"):
    """
    Compute the time-frequency representation of a signal using morlet wavelets.

    Parameters
    ----------
    sig : pynapple.Tsd or pynapple.TsdFrame
        Time series.
    fs : float
        Sampling rate, in Hz.
    freqs : 1d array or list of float
        If array, frequency values to estimate with morlet wavelets.
        If list, define the frequency range, as [freq_start, freq_stop, freq_step].
        The `freq_step` is optional, and defaults to 1. Range is inclusive of `freq_stop` value.
    n_cycles : float or 1d array
        Length of the filter, as the number of cycles for each frequency.
        If 1d array, this defines n_cycles for each frequency.
    scaling : float
        Scaling factor.
    norm : {'sss', 'amp'}, optional
        Normalization method:

        * 'sss' - divide by the square root of the sum of squares
        * 'amp' - divide by the sum of amplitudes

    Returns
    -------
    mwt : 2d array
        Time frequency representation of the input signal.

    Notes
    -----
    This computes the continuous wavelet transform at specified frequencies across time.
    """
    if not isinstance(sig, nap.Tsd) and not isinstance(sig, nap.TsdFrame):
        raise TypeError("`sig` must be instance of Tsd or TsdFrame")
    if isinstance(freqs, (tuple, list)):
        freqs = _create_freqs(*freqs)
    if fs is None:
        fs = sig.index.shape[0]/(sig.index.max() - sig.index.min())
    n_cycles = _check_n_cycles(n_cycles, len(freqs))
    if isinstance(sig, nap.Tsd):
        mwt = np.zeros([len(freqs), len(sig)], dtype=complex)
        for ind, (freq, n_cycle) in enumerate(zip(freqs, n_cycles)):
            mwt[ind, :] = _convolve_wavelet(sig,
                                            fs,
                                            freq,
                                            n_cycle,
                                            scaling,
                                            norm=norm)
        return nap.TsdFrame(t=sig.index, d=np.transpose(mwt), time_support=sig.time_support)
    else:
        mwt = np.zeros(
            [sig.values.shape[0], len(freqs), sig.values.shape[1]], dtype=complex
        )
        for channel_i in range(sig.values.shape[1]):
            for ind, (freq, n_cycle) in enumerate(zip(freqs, n_cycles)):
                mwt[:, ind, channel_i] = _convolve_wavelet(
                    sig[:, channel_i], fs, freq, n_cycle, scaling, norm=norm
                )
        return nap.TsdTensor(t=sig.index, d=mwt, time_support=sig.time_support)


def _convolve_wavelet(
    sig, fs, freq, n_cycles=1.5, scaling=1.0, precision=10, norm="sss"
):
    """
    Convolve a signal with a complex wavelet.

    Parameters
    ----------
    sig : pynapple.Tsd
        Time series to filter.
    fs : float
        Sampling rate, in Hz.
    freq : float
        Center frequency of bandpass filter.
    n_cycles : float, optional, default: 7
        Length of the filter, as the number of cycles of the oscillation with specified frequency.
    scaling : float, optional, default: 0.5
        Scaling factor for the morlet wavelet.
    norm : {'sss', 'amp'}, optional
        Normalization method:

        * 'sss' - divide by the square root of the sum of squares
        * 'amp' - divide by the sum of amplitudes

    Returns
    -------
    array
        Complex- valued time series.

    Notes
    -----

    * The real part of the returned array is the filtered signal.
    * Taking np.abs() of output gives the analytic amplitude.
    * Taking np.angle() of output gives the analytic phase.
    """
    if norm not in ["sss", "amp"]:
        raise ValueError("Given `norm` must be `sss` or `amp`")
    morlet_f = morlet(int(2**precision), ncycles=n_cycles, scaling=scaling)
    x = np.linspace(-8, 8, int(2**precision))
    int_psi = np.conj(_integrate(morlet_f, x[1] - x[0]))
    scale = scaling / (freq/fs)
    j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * (x[1] - x[0]))
    j = j.astype(int)  # floor
    if j[-1] >= int_psi.size:
        j = np.extract(j < int_psi.size, j)
    int_psi_scale = int_psi[j][::-1]
    conv = np.convolve(sig, int_psi_scale)
    coef = - np.sqrt(scale) * np.diff(conv, axis=-1)
    # transform axis is always -1 due to the data reshape above
    d = (coef.shape[-1] - sig.shape[-1]) / 2.
    if d > 0:
        coef = coef[..., floor(d):-ceil(d)]
    elif d < 0:
        raise ValueError(
            f"Selected scale of {scale} too small.")
    return coef

def _integrate(arr, step):
    integral = np.cumsum(arr)
    integral *= step
    return integral
