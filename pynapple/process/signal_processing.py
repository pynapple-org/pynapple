"""
Signal processing tools for Pynapple.

Contains functionality for signal processing pynapple object; fourier transforms and wavelet decomposition.
"""

import numpy as np
import pynapple as nap
from math import ceil, floor
import json
from scipy.signal import welch

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
