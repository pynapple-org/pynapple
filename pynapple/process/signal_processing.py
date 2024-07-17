"""
Signal processing tools for Pynapple.

Contains functionality for signal processing pynapple object; fourier transforms and wavelet decomposition.
"""

import numpy as np
import pandas as pd
from scipy.signal import welch

import pynapple as nap


def compute_spectogram(sig, fs=None, ep=None, full_range=False):
    """
    Performs numpy fft on sig, returns output. Pynapple assumes a constant sampling rate for sig.

    ----------
    sig : pynapple.Tsd or pynapple.TsdFrame
        Time series.
    fs : float, optional
        Sampling rate, in Hz. If None, will be calculated from the given signal
    ep : pynapple.IntervalSet or None, optional
        The epoch to calculate the fft on. Must be length 1.
    full_range : bool, optional
        If true, will return full fft frequency range, otherwise will return only positive values

    Notes
    -----
    compute_spectogram computes fft on only a single epoch of data. This epoch be given with the ep
    parameter otherwise will be sig.time_support, but it must only be a single epoch.
    """
    if not isinstance(sig, (nap.Tsd, nap.TsdFrame)):
        raise TypeError(
            "Currently compute_spectogram is only implemented for Tsd or TsdFrame"
        )
    if not (ep is None or isinstance(ep, nap.IntervalSet)):
        raise TypeError("ep param must be a pynapple IntervalSet object, or None")
    if ep is None:
        ep = sig.time_support
    if len(ep) != 1:
        raise ValueError("Given epoch (or signal time_support) must have length 1")
    if fs is None:
        fs = sig.index.shape[0] / (sig.index.max() - sig.index.min())
    fft_result = np.fft.fft(sig.restrict(ep).values, axis=0)
    fft_freq = np.fft.fftfreq(len(sig.restrict(ep).values), 1 / fs)
    ret = pd.DataFrame(fft_result, fft_freq)
    ret.sort_index(inplace=True)
    if not full_range:
        return ret.loc[ret.index >= 0]
    return ret


def compute_welch_spectogram(sig, fs=None):
    """
    Performs Welch's decomposition on sig, returns output.
    Estimates the power spectral density of a signal by segmenting it into overlapping sections, applying a
    window function to each segment, computing their FFTs, and averaging the resulting periodograms to reduce noise.

    ..todo: remove this or add binsize parameter
    ..todo: be careful of border artifacts

    Parameters
    ----------
    sig : pynapple.Tsd or pynapple.TsdFrame
        Time series.
    fs : float, optional
        Sampling rate, in Hz. If None, will be calculated from the given signal
    """
    if not isinstance(sig, (nap.Tsd, nap.TsdFrame)):
        raise TypeError(
            "Currently compute_welch_spectogram is only implemented for Tsd or TsdFrame"
        )
    if fs is None:
        fs = sig.index.shape[0] / (sig.index.max() - sig.index.min())
    freqs, spectogram = welch(sig.values, fs=fs, axis=0)
    return pd.DataFrame(spectogram, freqs)


def _morlet(M=1024, ncycles=1.5, scaling=1.0, precision=8):
    """
    Defines the complex Morlet wavelet kernel

    Parameters
    ----------
    M : int
        Length of the wavelet
    ncycles : float
        number of wavelet cycles to use. Default is 1.5
    scaling: float
        Scaling factor. Default is 1.0
    precision: int.
        Precision of wavelet to use. Default is 8

    Returns
    -------
    np.ndarray
        Morelet wavelet kernel
    """
    x = np.linspace(-precision, precision, M)
    return (
        ((np.pi * ncycles) ** (-0.25))
        * np.exp(-(x**2) / ncycles)
        * np.exp(1j * 2 * np.pi * scaling * x)
    )


def _create_freqs(freq_start, freq_stop, freq_step=1, log_scaling=False, log_base=np.e):
    """
    Creates an array of frequencies.

    Parameters
    ----------
    freq_start : float
        Starting value for the frequency definition.
    freq_stop: float
        Stopping value for the frequency definition, inclusive.
    freq_step: float, optional
        Step value, for linearly spaced values between start and stop.
    log_scaling: Bool
        If True, will use log spacing with base log_base for frequency spacing. Default False.
    log_base: float
        If log_scaling==True, this defines the base of the log to use.

    Returns
    -------
    freqs: 1d array
        Frequency indices.
    """
    if not log_scaling:
        return np.arange(freq_start, freq_stop + freq_step, freq_step)
    else:
        return np.logspace(freq_start, freq_stop, base=log_base)


def compute_wavelet_transform(
    sig, freqs, fs=None, n_cycles=1.5, scaling=1.0, precision=10, norm=None
):
    """
    Compute the time-frequency representation of a signal using morlet wavelets.

    Parameters
    ----------
    sig : pynapple.Tsd, pynapple.TsdFrame or pynapple.TsdTensor
        Time series.
    freqs : 1d array or list of float
        If array, frequency values to estimate with morlet wavelets.
        If list, define the frequency range, as [freq_start, freq_stop, freq_step].
        The `freq_step` is optional, and defaults to 1. Range is inclusive of `freq_stop` value.
    fs : float or None
        Sampling rate, in Hz. Defaults to sig.rate if None is given.
    n_cycles : float or 1d array
        Length of the filter, as the number of cycles for each frequency.
        If 1d array, this defines n_cycles for each frequency.
    scaling : float
        Scaling factor.
    precision: int.
        Precision of wavelet to use. Default is 8
    norm : {None, 'sss', 'amp'}, optional
        Normalization method:
        * None - no normalization
        * 'sss' - divide by the square root of the sum of squares
        * 'amp' - divide by the sum of amplitudes

    Returns
    -------
    pynapple.TsdFrame or pynapple.TsdTensor : 2d array
        Time frequency representation of the input signal.

    Notes
    -----
    This computes the continuous wavelet transform at specified frequencies across time.
    """

    if not isinstance(sig, (nap.Tsd, nap.TsdFrame, nap.TsdTensor)):
        raise TypeError("`sig` must be instance of Tsd, TsdFrame, or TsdTensor")
    if isinstance(n_cycles, (int, float, np.number)):
        if n_cycles <= 0:
            raise ValueError("Number of cycles must be a positive number.")

    if isinstance(freqs, (tuple, list)):
        freqs = _create_freqs(*freqs)

    if fs is None:
        fs = sig.rate

    if isinstance(sig, nap.Tsd):
        sig = sig.reshape((sig.shape[0], 1))
        output_shape = (sig.shape[0], len(freqs))
    else:
        output_shape = (sig.shape[0], len(freqs), *sig.shape[1:])
        sig = sig.reshape((sig.shape[0], np.prod(sig.shape[1:])))

    filter_bank = generate_morlet_filterbank(freqs, fs, n_cycles, scaling, precision)
    convolved_real = sig.convolve(np.transpose(filter_bank.real))
    convolved_imag = sig.convolve(np.transpose(filter_bank.imag))
    convolved = convolved_real.values + convolved_imag.values * 1j
    coef = -np.diff(convolved, axis=0)
    if norm == "sss":
        coef *= coef * (-np.sqrt(scaling) / (freqs / fs))
    elif norm == "amp":
        coef *= -scaling / (freqs / fs)
    coef = np.insert(
        coef, 1, coef[0, :], axis=0
    )  # slightly hacky line, necessary to make output correct shape
    cwt = np.swapaxes(coef, 1, 2)

    if len(output_shape) == 2:
        return nap.TsdFrame(
            t=sig.index, d=cwt.reshape(output_shape), time_support=sig.time_support
        )

    return nap.TsdTensor(
        t=sig.index, d=cwt.reshape(output_shape), time_support=sig.time_support
    )


def generate_morlet_filterbank(freqs, fs, n_cycles=1.5, scaling=1.0, precision=10):
    """

    Parameters
    ----------
    freqs : 1d array or list of float
        If array, frequency values to estimate with morlet wavelets.
        If list, define the frequency range, as [freq_start, freq_stop, freq_step].
        The `freq_step` is optional, and defaults to 1. Range is inclusive of `freq_stop` value.
    fs : float
        Sampling rate, in Hz.
    n_cycles : float or 1d array
        Length of the filter, as the number of cycles for each frequency.
        If 1d array, this defines n_cycles for each frequency.
    scaling : float
        Scaling factor.
    precision: int.
        Precision of wavelet to use.

    Returns
    -------
    filter_bank : np.ndarray
        list of Morlet wavelet filters of the frequencies given
    """
    filter_bank = []
    morlet_f = _morlet(int(2**precision), ncycles=n_cycles, scaling=scaling)
    x = np.linspace(-8, 8, int(2**precision))
    int_psi = np.conj(_integrate(morlet_f, x[1] - x[0]))
    max_len = 0
    for freq in freqs:
        scale = scaling / (freq / fs)
        j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * (x[1] - x[0]))
        j = j.astype(int)  # floor
        if j[-1] >= int_psi.size:
            j = np.extract(j < int_psi.size, j)
        int_psi_scale = int_psi[j][::-1]
        if len(int_psi_scale) > max_len:
            max_len = len(int_psi_scale)
        filter_bank.append(int_psi_scale)
    filter_bank = [
        np.pad(
            arr,
            ((max_len - len(arr)) // 2, (max_len - len(arr) + 1) // 2),
            constant_values=0.0,
        )
        for arr in filter_bank
    ]
    return np.array(filter_bank)


def _integrate(arr, step):
    """
    Integrates an array with respect to some step param. Used for integrating complex wavelets.

    Parameters
    ----------
    arr : np.ndarray
        wave function to be integrated
    step : float
        Step size of vgiven wave function array

    Returns
    -------
    array
        Complex-valued integrated wavelet

    """
    integral = np.cumsum(arr)
    integral *= step
    return integral
