"""
# Signal processing tools.

- `nap.compute_power_spectral_density`
- `nap.compute_mean_power_spectral_density`
- `nap.compute_wavelet_transform`
- `nap.generate_morlet_filterbank`

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


def compute_wavelet_transform(
    sig, freqs, fs=None, gaussian_width=1.5, window_length=1.0, precision=16, norm="l1"
):
    """
    Compute the time-frequency representation of a signal using Morlet wavelets.

    Parameters
    ----------
    sig : pynapple.Tsd or pynapple.TsdFrame or pynapple.TsdTensor
        Time series.
    freqs : 1d array
        Frequency values to estimate with Morlet wavelets.
    fs : float or None
        Sampling rate, in Hz. Defaults to `sig.rate` if None is given.
    gaussian_width : float
        Defines width of Gaussian to be used in wavelet creation. Default is 1.5.
    window_length : float
        The length of window to be used for wavelet creation. Default is 1.0.
    precision: int.
        Precision of wavelet to use. Defines the number of timepoints to evaluate the Morlet wavelet at.
        Default is 16.
    norm : {None, 'l1', 'l2'}, optional
        Normalization method:
        - None - no normalization
        - 'l1' - (default) divide by the sum of amplitudes
        - 'l2' - divide by the square root of the sum of amplitudes

    Returns
    -------
    pynapple.TsdFrame or pynapple.TsdTensor
        Time frequency representation of the input signal.

    Examples
    --------
    >>> import numpy as np
    >>> import pynapple as nap
    >>> t = np.arange(0, 1, 1/1000)
    >>> signal = nap.Tsd(d=np.sin(t * 50 * np.pi * 2), t=t)
    >>> freqs = np.linspace(10, 100, 10)
    >>> mwt = nap.compute_wavelet_transform(signal, fs=1000, freqs=freqs)

    Notes
    -----
    This computes the continuous wavelet transform at specified frequencies across time.
    """

    if not isinstance(sig, (nap.Tsd, nap.TsdFrame, nap.TsdTensor)):
        raise TypeError("`sig` must be instance of Tsd, TsdFrame, or TsdTensor")

    if not isinstance(freqs, np.ndarray):
        raise TypeError("`freqs` must be a ndarray")
    if len(freqs) == 0:
        raise ValueError("Given list of freqs cannot be empty.")
    if np.min(freqs) <= 0:
        raise ValueError("All frequencies in freqs must be strictly positive")

    if fs is not None and not isinstance(fs, (int, float, np.number)):
        raise TypeError("`fs` must be of type float or int or None")

    if norm is not None and norm not in ["l1", "l2"]:
        raise ValueError("norm parameter must be 'l1', 'l2', or None.")

    if fs is None:
        fs = sig.rate

    output_shape = (sig.shape[0], len(freqs), *sig.shape[1:])
    sig = np.reshape(sig, (sig.shape[0], -1))

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
    Generates a Morlet filterbank using the given frequencies and parameters.

    This function can be used purely for visualization, or to convolve with a pynapple Tsd,
    TsdFrame, or TsdTensor as part of a wavelet decomposition process.

    Parameters
    ----------
    freqs : 1d array
        frequency values to estimate with Morlet wavelets.
    fs : float or int
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

    Notes
    -----
    This algorithm first computes a single, finely sampled wavelet using the provided hyperparameters.
    Wavelets of different frequencies are generated by resampling this mother wavelet with an appropriate step size.
    The step size is determined based on the desired frequency and the sampling rate.
    """
    if not isinstance(freqs, np.ndarray):
        raise TypeError("`freqs` must be a ndarray")
    if len(freqs) == 0:
        raise ValueError("Given list of freqs cannot be empty.")
    if np.min(freqs) <= 0:
        raise ValueError("All frequencies in freqs must be strictly positive")

    if not isinstance(fs, (int, float, np.number)):
        raise TypeError("`fs` must be of type float or int ndarray")

    if isinstance(gaussian_width, (int, float, np.number)):
        if gaussian_width <= 0:
            raise ValueError("gaussian_width must be a positive number.")
    else:
        raise TypeError("gaussian_width must be a float or int instance.")

    if isinstance(window_length, (int, float, np.number)):
        if window_length <= 0:
            raise ValueError("window_length must be a positive number.")
    else:
        raise TypeError("window_length must be a float or int instance.")

    if isinstance(precision, int):
        if precision <= 0:
            raise ValueError("precision must be a positive number.")
    else:
        raise TypeError("precision must be a float or int instance.")

    # Initialize filter bank and parameters
    filter_bank = []
    cutoff = 8  # Define cutoff for wavelet
    # Compute a single, finely sampled Morlet wavelet
    morlet_f = np.conj(
        _morlet(
            int(2**precision),
            gaussian_width=gaussian_width,
            window_length=window_length,
        )
    )
    x = np.linspace(-cutoff, cutoff, int(2**precision))
    max_len = -1  # Track maximum length of wavelet
    for freq in freqs:
        scale = window_length / (freq / fs)
        # Calculate the indices for subsampling the wavelet and achieve the right frequency
        # After the slicing the size will be reduced, therefore we will pad with 0s.
        j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * (x[1] - x[0]))
        j = np.ceil(j).astype(int)  # Ceil the values to get integer indices
        if j[-1] >= morlet_f.size:
            j = np.extract(j < morlet_f.size, j)
        scaled_morlet = morlet_f[j][::-1]  # Scale and reverse wavelet
        if len(scaled_morlet) > max_len:
            max_len = len(scaled_morlet)
            time = np.linspace(
                -cutoff * window_length / freq, cutoff * window_length / freq, max_len
            )
        filter_bank.append(scaled_morlet)
    # Pad wavelets to ensure all are of the same length
    filter_bank = [
        np.pad(
            arr,
            ((max_len - len(arr)) // 2, (max_len - len(arr) + 1) // 2),
            constant_values=0.0,
        )
        for arr in filter_bank
    ]
    # Return filter bank as a TsdFrame
    return nap.TsdFrame(d=np.array(filter_bank).transpose(), t=time)
