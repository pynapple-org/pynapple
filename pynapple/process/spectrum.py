"""
Functions to compute power spectral density and mean power spectral density.
"""

import inspect
from functools import wraps
from numbers import Number

import numpy as np
import pandas as pd
from numba import njit
from scipy import signal

from .. import core as nap


@njit
def _overlap_split(start, end, interval_size, overlap):
    N = int(
        np.ceil(np.sum(end - start) / (interval_size * (1 - overlap)))
    )  # upper bound
    slices = np.zeros((N + 1, 2))

    k = 0  # epochs
    n = 0
    while k < len(start):
        t = start[k]
        while t + interval_size < end[k]:
            slices[n, 0] = t
            slices[n, 1] = t + interval_size
            t += (1 - overlap) * interval_size
            n += 1
        k += 1

    return slices[0:n]


def _validate_spectrum_inputs(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Validate each positional argument
        sig = inspect.signature(func)
        kwargs = sig.bind_partial(*args, **kwargs).arguments

        parameters_type = {
            "sig": (nap.Tsd, nap.TsdFrame),
            "fs": Number,
            "ep": nap.IntervalSet,
            "full_range": bool,
            "norm": bool,
            "n": int,
            "time_unit": str,
            "interval_size": Number,
            "overlap": float,
        }
        for param, param_type in parameters_type.items():
            if param in kwargs:
                if not isinstance(kwargs[param], param_type):
                    raise TypeError(
                        f"Invalid type. Parameter {param} must be of type {param_type}."
                    )

        # Call the original function with validated inputs
        return func(**kwargs)

    return wrapper


@_validate_spectrum_inputs
def compute_fft(sig, fs=None, ep=None, full_range=False, norm=False, n=None):
    """
    Compute Fast Fourier Transform over a single epoch.
    Perform numpy fft on sig, returns output assuming a constant sampling rate for the signal.

    Parameters
    ----------
    sig : Tsd or TsdFrame
        Time series.
    fs : float, optional
        Sampling rate, in Hz. If None, will be calculated from the given signal
    ep : None or IntervalSet, optional
        The epoch to calculate the fft on. Must be length 1.
    full_range : bool, optional
        If true, will return full fft frequency range, otherwise will return only positive values
    norm: bool, optional
        Whether the FFT result is divided by the length of the signal to normalize the amplitude
    n: int, optional
        Length of the transformed axis of the output. If n is smaller than the length of the input,
        the input is cropped. If it is larger, the input is padded with zeros. If n is not given,
        the length of the input along the axis specified by axis is used.

    Returns
    -------
    pandas.DataFrame
        Time frequency representation of the input signal, indexes are frequencies, values
        are the FFT.

    Notes
    -----
    This function computes fft on only a single epoch of data. This epoch be given with the ep
    parameter otherwise will be sig.time_support, but it must only be a single epoch.

    If `full_range` is False, the nyquist frequency is excluded in the output due to how computes the FFT frequencies in `numpy.fft.fftfreq`.
    """
    if ep is None:
        ep = sig.time_support
    if len(ep) != 1:
        raise ValueError("Given epoch (or signal time_support) must have length 1")
    if fs is None:
        fs = sig.rate

    fft_result = np.fft.fft(sig.restrict(ep).values, n=n, axis=0)
    if n is None:
        n = len(sig.restrict(ep))
    fft_freq = np.fft.fftfreq(n, 1 / fs)

    if norm:
        fft_result = fft_result / fft_result.shape[0]

    ret = pd.DataFrame(fft_result, fft_freq)
    ret.sort_index(inplace=True)

    if full_range is False:
        return ret.loc[ret.index >= 0]
    return ret


@_validate_spectrum_inputs
def compute_power_spectral_density(sig, fs=None, ep=None, full_range=False, n=None):
    """
    Compute Power Spectral Density over a single epoch.
    Perform numpy fft on sig and obtain the periodogram, returns output assuming a constant sampling rate for the signal.

    Parameters
    ----------
    sig : Tsd or TsdFrame
        Time series.
    fs : float, optional
        Sampling rate, in Hz. If None, will be calculated from the given signal
    ep : None or IntervalSet, optional
        The epoch to calculate the fft on. Must be length 1.
    full_range : bool, optional
        If true, will return full fft frequency range, otherwise will return only positive values
    n: int, optional
        Length of the transformed axis of the output. If n is smaller than the length of the input,
        the input is cropped. If it is larger, the input is padded with zeros. If n is not given,
        the length of the input along the axis specified by axis is used.

    Returns
    -------
    pandas.DataFrame
        Power spectral density of the input signal, indexes are frequencies, values
        are powers/frequency.

    Notes
    -----
    This function computes fft on only a single epoch of data. This epoch be given with the ep
    parameter otherwise will be sig.time_support, but it must only be a single epoch.

    The power spectral density is calculated as the square of the absolute value of the FFT, scaled by the sampling rate and length of the signal.
    See [this tutorial](https://www.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html) for more information.

    If `full_range` is False, the nyquist frequency is excluded in the output due to how computes the FFT frequencies in `numpy.fft.fftfreq`.
    """

    if ep is None:
        ep = sig.time_support
    if len(ep) != 1:
        raise ValueError("Given epoch (or signal time_support) must have length 1")
    if fs is None:
        fs = sig.rate
    if n is None:
        n = len(sig.restrict(ep))

    fft = compute_fft(sig, fs=fs, ep=ep, n=n, full_range=full_range)

    # transform to power spectral density, power/Hz
    psd = (1 / (fs * n)) * (np.abs(fft) ** 2)

    if full_range is False:
        # frequencies not at 0 and not at the nyquist frequency occur twice
        # subtract from the nyquist frequency to adjust for floating point error in np.fft.fftfreq
        # nyquist freq may occur at negative end of frequencies if N is even
        doubled_freqs = (fft.index != 0) & (  # not 0
            fft.index < (fs / 2 - 1e-6)
        )  # less than positive nyquist freq
        psd[doubled_freqs] *= 2

    return psd


@_validate_spectrum_inputs
def compute_mean_power_spectral_density(
    sig, interval_size, fs=None, overlap=0.25, ep=None, full_range=False, time_unit="s"
):
    """
    Compute mean power spectral density over multiple epochs of same size.

    The parameter `interval_size` controls the duration of the epochs.

    To improve frequency resolution, the signal is multiplied by a Hamming window.

    Note that this function assumes a constant sampling rate for `sig`.

    Parameters
    ----------
    sig : Tsd or TsdFrame
        Signal with equispaced samples
    interval_size : Number
        Epochs size to compute to average the FFT across
    fs : Number, optional
        Sampling frequency of `sig`. If `None`, `fs` is equal to `sig.rate`
    overlap : float, optional
        Percentage of overlap between successive intervals.
        `0.0 <= overlap < 1.0`. Default is 0.25
    ep : None or IntervalSet, optional
        The `IntervalSet` to calculate the fft on. Can be any length.
    full_range : bool, optional
        If true, will return full fft frequency range, otherwise will return only positive values
    time_unit : str, optional
        Time units for parameter `interval_size`. Can be ('s'[default], 'ms', 'us')

    Returns
    -------
    pandas.DataFrame
        Return mean power spectral density of the input signal.
        Indexes are frequencies and values are powers/frequency.


    Notes
    -----
    The power spectral density is calculated as the square of the absolute value of the FFT, scaled by the sampling rate and length of the signal.
    See [this tutorial](https://www.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html) for more information.

    Raises
    ------
    RuntimeError
        If splitting the epoch with `interval_size` results in an empty set.
    ValueError
        If overlap is not within [0, 1).
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError("Overlap should be in intervals [0.0, 1.0).")

    if ep is None:
        ep = sig.time_support

    if fs is None:
        fs = sig.rate

    interval_size = nap.TsIndex.format_timestamps(np.array([interval_size]), time_unit)[
        0
    ]

    # Check if at least one epoch is larger than the interval size
    if np.max(ep.end - ep.start) < interval_size:
        raise RuntimeError(
            f"Splitting epochs with interval_size={interval_size} generated an empty IntervalSet. Try decreasing interval_size"
        )

    split_ep = _overlap_split(ep.start, ep.end, interval_size, overlap)

    # Get the slices of each ep
    slices = np.zeros((len(split_ep), 2), dtype=int)

    for i in range(len(split_ep)):
        sl = sig.get_slice(split_ep[i, 0], split_ep[i, 1])
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

    # Get the Hamming window
    window = signal.windows.hamming(N)
    if sig.ndim == 2:
        window = window[:, np.newaxis]

    # Compute the PSD
    psd_result = np.zeros((N, *sig.shape[1:]), dtype=float)  # Default

    for i in range(len(slices)):
        tmp = sig[slices[i, 0] : slices[i, 1]].values[0:N] * window
        fft = np.fft.fft(tmp, axis=0)

        # transform to power spectral density, power/Hz
        psd = (1 / (fs * N)) * (np.abs(fft) ** 2)

        psd_result += psd  # Default

    psd_result /= float(len(slices))
    ret = pd.DataFrame(psd_result, index=fft_freq)
    ret.sort_index(inplace=True)
    # frequencies not at 0 and not at the nyquist frequency occur twice
    # subtract from the nyquist frequency to adjust for floating point error in np.fft.fftfreq
    # nyquist freq may occur at negative end of frequencies if N is even
    if full_range is False:
        ret = ret.loc[ret.index >= 0]
        doubled_freqs = (ret.index != 0) & (  # not 0
            ret.index < (fs / 2 - 1e-6)
        )  # less than positive nyquist freq
        ret[doubled_freqs] *= 2

    return ret
