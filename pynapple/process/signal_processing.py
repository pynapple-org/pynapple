"""
Signal processing tools for Pynapple.

Contains functionality for signal processing pynapple object; fourier transforms and wavelet decomposition.
"""

from numbers import Number

import numpy as np
import pandas as pd

from .. import core as nap


def compute_power_spectral_density(sig, fs=None, ep=None, full_range=False):
    """
    Performs numpy fft on sig, returns output. Pynapple assumes a constant sampling rate for sig.

    Parameters
    ----------
    sig : pynapple.Tsd or pynapple.TsdFrame
        Time series.
    fs : float, optional
        Sampling rate, in Hz. If None, will be calculated from the given signal
    ep : pynapple.IntervalSet or None, optional
        The epoch to calculate the fft on. Must be length 1.
    full_range : bool, optional
        If true, will return full fft frequency range, otherwise will return only positive values

    Returns
    -------
    pandas.DataFrame
        Time frequency representation of the input signal, indexes are frequencies, values
        are powers.

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
        fs = sig.rate
    fft_result = np.fft.fft(sig.restrict(ep).values, axis=0)
    fft_freq = np.fft.fftfreq(len(sig.restrict(ep).values), 1 / fs)
    ret = pd.DataFrame(fft_result, fft_freq)
    ret.sort_index(inplace=True)
    if not full_range:
        return ret.loc[ret.index >= 0]
    return ret


def compute_mean_power_spectral_density(
    sig, interval_size, fs=None, ep=None, full_range=False, time_unit="s"
):
    """Compute mean power spectral density by averaging FFT over epochs of same size.
    The parameter `interval_size` controls the duration of the epochs.

    Note that this function assumes a constant sampling rate for sig.

    Parameters
    ----------
    sig : Tsd or TsdFrame
        Signal with equispaced samples
    interval_size : Number
        Epochs size to compute to average the FFT across
    fs : None, optional
        Sampling frequency of `sig`. If `None`, `fs` is equal to `sig.rate`
    ep : None, optional
        The `IntervalSet` to calculate the fft on. Can be any length.
    full_range : bool, optional
        If true, will return full fft frequency range, otherwise will return only positive values
    time_unit : str, optional
        Time units for parameter `interval_size`. Can be ('s'[default], 'ms', 'us')

    Returns
    -------
    pandas.DataFrame
        Power spectral density.

    Raises
    ------
    RuntimeError
        If splitting the epoch with `interval_size` results in an empty set.
    TypeError
        If `ep` or `sig` are not respectively pynapple time series or interval set.
    """
    if not (ep is None or isinstance(ep, nap.IntervalSet)):
        raise TypeError("ep param must be a pynapple IntervalSet object, or None")
    if ep is None:
        ep = sig.time_support

    if not (fs is None or isinstance(fs, Number)):
        raise TypeError("fs must be of type float or int")
    if fs is None:
        fs = sig.rate

    if not isinstance(full_range, bool):
        raise TypeError("full_range must be of type bool or None")

    # Split the ep
    interval_size = nap.TsIndex.format_timestamps(np.array([interval_size]), time_unit)[
        0
    ]
    split_ep = ep.split(interval_size)

    if len(split_ep) == 0:
        raise RuntimeError(
            f"Splitting epochs with interval_size={interval_size} generated an empty IntervalSet. Try decreasing interval_size"
        )

    # Get the slices of each ep
    slices = np.zeros((len(split_ep), 2), dtype=int)

    for i in range(len(split_ep)):
        sl = sig.get_slice(split_ep[i, 0], split_ep[i, 1], mode="backward")
        slices[i, 0] = sl.start
        slices[i, 1] = sl.stop

    # Check what is the signal length
    N = np.min(np.diff(slices, 1))

    if N == 0:
        raise RuntimeError(
            "One interval doesn't have any signal associated. Check the parameter ep or the time support if no epoch is passed."
        )

    # Get the freqs
    fft_freq = np.fft.fftfreq(N, 1 / fs)

    # Compute the fft
    fft_result = np.zeros((N, *sig.shape[1:]), dtype=complex)

    for i in range(len(slices)):
        fft_result += np.fft.fft(sig[slices[i, 0] : slices[i, 1]].values[0:N], axis=0)

    ret = pd.DataFrame(fft_result, fft_freq)
    ret.sort_index(inplace=True)
    if not full_range:
        return ret.loc[ret.index >= 0]
    return ret


def _morlet(M=1024, gaussian_width=1.5, window_length=1.0, precision=8):
    """
    Defines the complex Morlet wavelet kernel.

    Parameters
    ----------
    M : int
        Length of the wavelet
    gaussian_width : float
        Defines width of Gaussian to be used in wavelet creation.
    window_length : float
        The length of window to be used for wavelet creation.
    precision: int.
        Precision of wavelet to use. Default is 8

    Returns
    -------
    np.ndarray
        Morelet wavelet kernel
    """
    x = np.linspace(-precision, precision, M)
    return (
        ((np.pi * gaussian_width) ** (-0.25))
        * np.exp(-(x**2) / gaussian_width)
        * np.exp(1j * 2 * np.pi * window_length * x)
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
    sig, freqs, fs=None, gaussian_width=1.5, window_length=1.0, precision=16, norm="l1"
):
    """
    Compute the time-frequency representation of a signal using Morlet wavelets.

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
    gaussian_width : float
        Defines width of Gaussian to be used in wavelet creation.
    window_length : float
        The length of window to be used for wavelet creation.
    precision: int.
        Precision of wavelet to use. . Defines the number of timepoints to evaluate the Morlet wavelet at.
        Default is 16
    norm : {None, 'l1', 'l2'}, optional
        Normalization method:
        * None - no normalization
        * 'l1' - divide by the sum of amplitudes
        * 'l2' - divide by the square root of the sum of amplitudes

    Returns
    -------
    pynapple.TsdFrame or pynapple.TsdTensor
        Time frequency representation of the input signal.

    Examples
    --------
    >>> import numpy as np
    >>> import pynapple as nap
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = nap.Tsd(d=np.sin(t * 50 * np.pi * 2), t=t)
    >>> freqs = np.linspace(10, 100, 10)
    >>> mwt = nap.compute_wavelet_transform(signal, fs=None, freqs=freqs)

    Notes
    -----
    This computes the continuous wavelet transform at specified frequencies across time.
    """

    if not isinstance(sig, (nap.Tsd, nap.TsdFrame, nap.TsdTensor)):
        raise TypeError("`sig` must be instance of Tsd, TsdFrame, or TsdTensor")
    if isinstance(gaussian_width, (int, float, np.number)):
        if gaussian_width <= 0:
            raise ValueError("gaussian_width must be a positive number.")
    if norm is not None and norm not in ["l1", "l2"]:
        raise ValueError("norm parameter must be 'l1', 'l2', or None.")

    if isinstance(freqs, (tuple, list)):
        freqs = _create_freqs(*freqs)

    if fs is None:
        fs = sig.rate

    if isinstance(sig, nap.Tsd):
        output_shape = (sig.shape[0], len(freqs))
    else:
        output_shape = (sig.shape[0], len(freqs), *sig.shape[1:])
        sig = sig.reshape((sig.shape[0], np.prod(sig.shape[1:])))

    filter_bank = generate_morlet_filterbank(
        freqs, fs, gaussian_width, window_length, precision
    )
    convolved_real = sig.convolve(filter_bank.real().values)
    convolved_imag = sig.convolve(filter_bank.imag().values)
    convolved = convolved_real.values + convolved_imag.values * 1j
    if norm == "l1":
        coef = convolved / (fs / freqs)
    elif norm == "l2":
        coef = convolved / (fs / np.sqrt(freqs))
    else:
        coef = convolved
    cwt = np.expand_dims(coef, -1) if len(coef.shape) == 2 else coef

    if len(output_shape) == 2:
        return nap.TsdFrame(
            t=sig.index, d=cwt.reshape(output_shape), time_support=sig.time_support
        )

    return nap.TsdTensor(
        t=sig.index, d=cwt.reshape(output_shape), time_support=sig.time_support
    )


def generate_morlet_filterbank(
    freqs, fs, gaussian_width=1.5, window_length=1.0, precision=16
):
    """
    Generates a Morlet filterbank using the given frequencies and parameters. Can be used purely for visualization,
    or to convolve with a pynapple Tsd, TsdFrame, or TsdTensor as part of a wavelet decomposition process.

    Parameters
    ----------
    freqs : 1d array or list of float
        If array, frequency values to estimate with morlet wavelets.
        If list, define the frequency range, as [freq_start, freq_stop, freq_step].
        The `freq_step` is optional, and defaults to 1. Range is inclusive of `freq_stop` value.
    fs : float
        Sampling rate, in Hz.
    gaussian_width : float
        Defines width of Gaussian to be used in wavelet creation.
    window_length : float
        The length of window to be used for wavelet creation.
    precision: int.
        Precision of wavelet to use. Defines the number of timepoints to evaluate the Morlet wavelet at.

    Returns
    -------
    filter_bank : pynapple.TsdFrame
        list of Morlet wavelet filters of the frequencies given
    """
    if len(freqs) == 0:
        raise ValueError("Given list of freqs cannot be empty.")
    if np.min(freqs) <= 0:
        raise ValueError("All frequencies in freqs must be strictly positive")
    filter_bank = []
    cutoff = 8
    morlet_f = _morlet(
        int(2**precision), gaussian_width=gaussian_width, window_length=window_length
    )
    x = np.linspace(-cutoff, cutoff, int(2**precision))
    int_psi = np.conj(morlet_f)
    max_len = -1
    for freq in freqs:
        scale = window_length / (freq / fs)
        j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * (x[1] - x[0]))
        j = j.astype(int)  # floor
        if j[-1] >= int_psi.size:
            j = np.extract(j < int_psi.size, j)
        int_psi_scale = int_psi[j][::-1]
        if len(int_psi_scale) > max_len:
            max_len = len(int_psi_scale)
            time = np.linspace(
                -cutoff * window_length / freq, cutoff * window_length / freq, max_len
            )
        filter_bank.append(int_psi_scale)
    filter_bank = [
        np.pad(
            arr,
            ((max_len - len(arr)) // 2, (max_len - len(arr) + 1) // 2),
            constant_values=0.0,
        )
        for arr in filter_bank
    ]
    return nap.TsdFrame(d=np.array(filter_bank).transpose(), t=time)


def _integrate(arr, step):
    """
    Integrates an array with respect to some step param. Used for integrating complex wavelets.

    Parameters
    ----------
    arr : np.ndarray
        wave function to be integrated
    step : float
        Step size of given wave function array

    Returns
    -------
    array
        Complex-valued integrated wavelet

    """
    integral = np.cumsum(arr)
    integral *= step
    return integral
