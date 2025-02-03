"""
Functions to compute wavelets decomposition of a time series.

The main function for doing wavelet decomposition is `nap.compute_wavelet_transform`

For now, pynapple only implements Morlet wavelets. To check the shape and quality of the wavelets, check out
the function `nap.generate_morlet_filterbank` to plot the wavelets.

"""

import numpy as np

from .. import core as nap


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

    output_shape = (sig.shape[0], *sig.shape[1:], len(freqs))
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
            t=sig.index,
            d=np.squeeze(cwt, axis=1),
            time_support=sig.time_support,
            columns=freqs,
        )
    else:
        return nap.TsdTensor(
            t=sig.index, d=np.reshape(cwt, output_shape), time_support=sig.time_support
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
