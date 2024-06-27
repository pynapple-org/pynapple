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


def compute_wavelet_transform(sig, freqs, fs=None):
    """
    Compute the time-frequency representation of a signal using morlet wavelets.

    Parameters
    ----------
    sig : pynapple.Tsd or pynapple.TsdFrame
        Time series.
    freqs : 1d array or list of float
        If array, frequency values to estimate with morlet wavelets.
        If list, define the frequency range, as [freq_start, freq_stop, freq_step].
        The `freq_step` is optional, and defaults to 1. Range is inclusive of `freq_stop` value.
    fs : float, optional
        Sampling rate, in Hz. If None, will be calculated from the given signal

    Returns
    -------
    mwt : 2d array
        Time frequency representation of the input signal.

    Notes
    -----
    This computes the continuous wavelet transform at specified frequencies across time.
    """
    if not isinstance(sig, nap.Tsd) and not isinstance(sig, nap.TsdFrame):
        raise TypeError("`sig` must be instance of Tsd, TsdFrame, or TsdTensor")
    if fs is None:
        fs = sig.index.shape[0]/(sig.index.max() - sig.index.min())
    assert fs/2 > np.max(freqs), "`freqs` contains values over the Nyquist frequency."
    if isinstance(sig, nap.Tsd):
        mwt, f = _cwt(sig,
                      freqs=freqs,
                      wavelet="cmor1.5-1.0",
                      sampling_period=1/fs)
        return nap.TsdFrame(t=sig.index, d=np.transpose(mwt), time_support=sig.time_support)
    elif isinstance(sig, nap.TsdFrame):
        mwt = np.zeros(
            [sig.values.shape[0], len(freqs), sig.values.shape[1]], dtype=complex
        )
        for channel_i in range(sig.values.shape[1]):
            mwt[:, :, channel_i] = np.transpose(_cwt(sig[:, channel_i],
                                                freqs=freqs,
                                                wavelet="cmor1.5-1.0",
                                                sampling_period=1/fs)[0])
        return nap.TsdTensor(t=sig.index, d=mwt, time_support=sig.time_support)
    elif isinstance(sig, nap.TsdTensor):
        raise NotImplemented("cwt for TsdTensor is not yet implemented")


def _cwt(data, freqs, wavelet, sampling_period, axis=-1):
    """
    cwt(data, scales, wavelet)

    One dimensional Continuous Wavelet Transform.

    Parameters
    ----------
    data : pynapple.Tsd or pynapple.TsdFrame
        Input time series signal.
    freqs : 1d array
        Frequency values to estimate with morlet wavelets.
    wavelet : Wavelet object or name
        Wavelet to use, only implemented for "cmor1.5-1.0".
    sampling_period : float
        Sampling period for the frequencies output.
        The values computed for ``coefs`` are independent of the choice of
        ``sampling_period`` (i.e. ``scales`` is not scaled by the sampling
        period).
    axis: int, optional
        Axis over which to compute the CWT. If not given, the last axis is
        used.

    Returns
    -------
    coefs : array_like
        Continuous wavelet transform of the input signal for the given scales
        and wavelet. The first axis of ``coefs`` corresponds to the scales.
        The remaining axes match the shape of ``data``.
    frequencies : array_like
        If the unit of sampling period are seconds and given, then frequencies
        are in hertz. Otherwise, a sampling period of 1 is assumed.

    ..todo:: This should use pynapple convolve but currently that cannot handle imaginary numbers as it uses scipy convolve
    """
    int_psi = np.array(WAVELET_DICT[wavelet]['int_psi_real'])*1j + np.array(WAVELET_DICT[wavelet]['int_psi_imag'])
    x = np.array(WAVELET_DICT[wavelet]["x"])
    central_freq = WAVELET_DICT[wavelet]["central_freq"]
    scales = central_freq/(freqs*sampling_period)
    out = np.empty((np.size(scales),) + data.shape, dtype=np.complex128)

    if data.ndim > 1:
        # move axis to be transformed last (so it is contiguous)
        data = data.swapaxes(-1, axis)
        # reshape to (n_batch, data.shape[-1])
        data_shape_pre = data.shape
        data = data.reshape((-1, data.shape[-1]))

    for i, scale in enumerate(scales):
        step = x[1] - x[0]
        j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step)
        j = j.astype(int)  # floor
        if j[-1] >= int_psi.size:
            j = np.extract(j < int_psi.size, j)
        int_psi_scale = int_psi[j][::-1]

        if data.ndim == 1:
            conv = np.convolve(data, int_psi_scale)
        else:
            # batch convolution via loop
            conv_shape = list(data.shape)
            conv_shape[-1] += int_psi_scale.size - 1
            conv_shape = tuple(conv_shape)
            conv = np.empty(conv_shape, dtype=np.complex128)
            for n in range(data.shape[0]):
                conv[n, :] = np.convolve(data[n], int_psi_scale)

        coef = - np.sqrt(scale) * np.diff(conv, axis=-1)
        if out.dtype.kind != 'c':
            coef = coef.real
        # transform axis is always -1 due to the data reshape above
        d = (coef.shape[-1] - data.shape[-1]) / 2.
        if d > 0:
            coef = coef[..., floor(d):-ceil(d)]
        elif d < 0:
            raise ValueError(
                f"Selected scale of {scale} too small.")
        if data.ndim > 1:
            # restore original data shape and axis position
            coef = coef.reshape(data_shape_pre)
            coef = coef.swapaxes(axis, -1)
        out[i, ...] = coef

    frequencies = central_freq/scales
    if np.isscalar(frequencies):
        frequencies = np.array([frequencies])
    frequencies /= sampling_period
    return out, frequencies






# -------------------------------------------------------------------------------

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

"""
The following code has been adapted from functions in the neurodsp package:
https://github.com/neurodsp-tools/neurodsp

..todo: reference licence in LICENCE directory
"""

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


def compute_wavelet_transform_og(sig, fs, freqs, n_cycles=7, scaling=0.5, norm="amp"):
    """
    Compute the time-frequency representation of a signal using morlet wavelets.

    ..todo:: better normalization between channels

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
    sig, fs, freq, n_cycles=7, scaling=0.5, wavelet_len=None, norm="sss"
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
    wavelet_len : int, optional
        Length of the wavelet. If defined, this overrides the freq and n_cycles inputs.
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
    * Taking np.angle() of output gives the analytic phase. ..todo: this this still true?
    """
    if norm not in ["sss", "amp"]:
        raise ValueError("Given `norm` must be `sss` or `amp`")
    if wavelet_len is None:
        wavelet_len = int(n_cycles * fs / freq)
    if wavelet_len > sig.shape[-1]:
        raise ValueError(
            "The length of the wavelet is greater than the signal. Can not proceed."
        )
    morlet_f = morlet(wavelet_len, ncycles=n_cycles, scaling=scaling)
    if norm == "sss":
        morlet_f = morlet_f / np.sqrt(np.sum(np.abs(morlet_f) ** 2))
    elif norm == "amp":
        morlet_f = morlet_f / np.sum(np.abs(morlet_f))
    mwt_real = sig.convolve(np.real(morlet_f))
    mwt_imag = sig.convolve(np.imag(morlet_f))
    return mwt_real.values + 1j * mwt_imag.values