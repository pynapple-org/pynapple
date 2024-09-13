"""
# Power spectral density

This module contains functions to compute power spectral density and mean power spectral density.

"""

from numbers import Number

import numpy as np
import pandas as pd
from scipy import signal

from .. import core as nap


def compute_power_spectral_density(
    sig, fs=None, ep=None, full_range=False, norm=False, n=None
):
    """
    Perform numpy fft on sig, returns output assuming a constant sampling rate for the signal.

    Parameters
    ----------
    sig : pynapple.Tsd or pynapple.TsdFrame
        Time series.
    fs : float, optional
        Sampling rate, in Hz. If None, will be calculated from the given signal
    ep : None or pynapple.IntervalSet, optional
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
        are powers.

    Notes
    -----
    compute_spectogram computes fft on only a single epoch of data. This epoch be given with the ep
    parameter otherwise will be sig.time_support, but it must only be a single epoch.
    """
    if not isinstance(sig, (nap.Tsd, nap.TsdFrame)):
        raise TypeError("sig must be either a Tsd or a TsdFrame object.")
    if not (fs is None or isinstance(fs, Number)):
        raise TypeError("fs must be of type float or int")
    if not (ep is None or isinstance(ep, nap.IntervalSet)):
        raise TypeError("ep param must be a pynapple IntervalSet object, or None")
    if ep is None:
        ep = sig.time_support
    if len(ep) != 1:
        raise ValueError("Given epoch (or signal time_support) must have length 1")
    if fs is None:
        fs = sig.rate
    if not isinstance(full_range, bool):
        raise TypeError("full_range must be of type bool or None")
    if not isinstance(norm, bool):
        raise TypeError("norm must be of type bool")

    fft_result = np.fft.fft(sig.restrict(ep).values, n=n, axis=0)
    if n is None:
        n = len(sig.restrict(ep))
    fft_freq = np.fft.fftfreq(n, 1 / fs)

    if norm:
        fft_result = fft_result / fft_result.shape[0]

    ret = pd.DataFrame(fft_result, fft_freq)
    ret.sort_index(inplace=True)

    if not full_range:
        return ret.loc[ret.index >= 0]
    return ret


def compute_mean_power_spectral_density(
    sig,
    interval_size,
    fs=None,
    ep=None,
    full_range=False,
    norm=False,
    time_unit="s",
):
    """
    Compute mean power spectral density by averaging FFT over epochs of same size.

    The parameter `interval_size` controls the duration of the epochs.

    To imporve frequency resolution, the signal is multiplied by a Hamming window.

    Note that this function assumes a constant sampling rate for `sig`.

    Parameters
    ----------
    sig : pynapple.Tsd or pynapple.TsdFrame
        Signal with equispaced samples
    interval_size : Number
        Epochs size to compute to average the FFT across
    fs : None, optional
        Sampling frequency of `sig`. If `None`, `fs` is equal to `sig.rate`
    ep : None or pynapple.IntervalSet, optional
        The `IntervalSet` to calculate the fft on. Can be any length.
    full_range : bool, optional
        If true, will return full fft frequency range, otherwise will return only positive values
    norm: bool, optional
        Whether the FFT result is divided by the length of the signal to normalize the amplitude
    time_unit : str, optional
        Time units for parameter `interval_size`. Can be ('s'[default], 'ms', 'us')

    Returns
    -------
    pandas.DataFrame
        Power spectral density.

    Examples
    --------
    >>> import numpy as np
    >>> import pynapple as nap
    >>> t = np.arange(0, 1, 1/1000)
    >>> signal = nap.Tsd(d=np.sin(t * 50 * np.pi * 2), t=t)
    >>> mpsd = nap.compute_mean_power_spectral_density(signal, 0.1)

    Raises
    ------
    RuntimeError
        If splitting the epoch with `interval_size` results in an empty set.
    TypeError
        If `ep` or `sig` are not respectively pynapple time series or interval set.
    """
    if not isinstance(sig, (nap.Tsd, nap.TsdFrame)):
        raise TypeError("sig must be either a Tsd or a TsdFrame object.")

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

    if not isinstance(norm, bool):
        raise TypeError("norm must be of type bool")

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

    # Compute the fft
    fft_result = np.zeros((N, *sig.shape[1:]), dtype=complex)

    for i in range(len(slices)):
        tmp = sig[slices[i, 0] : slices[i, 1]].values[0:N] * window
        fft_result += np.fft.fft(tmp, axis=0)

    if norm:
        fft_result = fft_result / (float(N) * float(len(slices)))

    ret = pd.DataFrame(fft_result, fft_freq)
    ret.sort_index(inplace=True)
    if not full_range:
        return ret.loc[ret.index >= 0]
    return ret
