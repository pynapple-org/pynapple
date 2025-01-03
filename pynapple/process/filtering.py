"""Functions for highpass, lowpass, bandpass or bandstop filtering."""

import inspect
from collections.abc import Iterable
from functools import wraps
from numbers import Number

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, sosfreqz

from .. import core as nap


def _validate_filtering_inputs(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Validate each positional argument
        sig = inspect.signature(func)
        kwargs = sig.bind_partial(*args, **kwargs).arguments

        cutoff = kwargs["cutoff"]
        filter_type = kwargs["filter_type"]
        if filter_type in ["lowpass", "highpass"] and not isinstance(cutoff, Number):
            raise ValueError(
                f"{filter_type} filter require a single number. {cutoff} provided instead."
            )
        if filter_type in ["bandpass", "bandstop"]:
            if (
                not isinstance(cutoff, Iterable)
                or len(cutoff) != 2
                or not all(isinstance(fq, Number) for fq in cutoff)
            ):
                raise ValueError(
                    f"{filter_type} filter require a tuple of two numbers. {cutoff} provided instead."
                )

        if "fs" in kwargs:
            if kwargs["fs"] is not None and not isinstance(kwargs["fs"], Number):
                raise ValueError(
                    "Invalid value for 'fs'. Parameter 'fs' should be of type float or int"
                )

        if "order" in kwargs:
            if not isinstance(kwargs["order"], int):
                raise ValueError(
                    "Invalid value for 'order': Parameter 'order' should be of type int"
                )

        if "transition_bandwidth" in kwargs:
            if not isinstance(kwargs["transition_bandwidth"], float):
                raise ValueError(
                    "Invalid value for 'transition_bandwidth'. 'transition_bandwidth' should be of type float"
                )

        # Call the original function with validated inputs
        return func(**kwargs)

    return wrapper


def _get_butter_coefficients(cutoff, filter_type, sampling_frequency, order=4):
    """Calls scipy butter"""
    return butter(order, cutoff, btype=filter_type, fs=sampling_frequency, output="sos")


def _compute_butterworth_filter(
    data, cutoff, sampling_frequency=None, filter_type="bandpass", order=4
):
    """
    Apply a Butterworth filter to the provided signal.
    """
    sos = _get_butter_coefficients(cutoff, filter_type, sampling_frequency, order)

    if nap.utils.get_backend() == "jax":
        from pynajax.jax_process_filtering import jax_sosfiltfilt

        out = jax_sosfiltfilt(
            sos,
            data.index.values,
            data.values,
            data.time_support.start,
            data.time_support.end,
        )

    else:
        out = np.zeros_like(data.d)
        for ep in data.time_support:
            slc = data.get_slice(start=ep.start[0], end=ep.end[0])
            out[slc] = sosfiltfilt(sos, data.d[slc], axis=0)

    return data._define_instance(data.t, data.time_support, values=out)


def _compute_spectral_inversion(kernel):
    """
    Compute the spectral inversion.
    Parameters
    ----------
    kernel: ndarray

    Returns
    -------
    ndarray
    """
    kernel *= -1.0
    kernel[len(kernel) // 2] = 1.0 + kernel[len(kernel) // 2]
    return kernel


def _get_windowed_sinc_kernel(
    fc, filter_type, sampling_frequency, transition_bandwidth=0.02
):
    """
    Get the windowed-sinc kernel.
    Smith, S. (2003). Digital signal processing: a practical guide for engineers and scientists.
    Chapter 16, equation 16-4

    Parameters
    ----------
    fc: float or tuple of float
        Cutting frequency in Hz. Single float for 'lowpass' and 'highpass'. Tuple of float for
        'bandpass' and 'bandstop'.
    filter_type: str
        Either 'lowpass', 'highpass', 'bandstop' or 'bandpass'.
    sampling_frequency: float
        Sampling frequency in Hz.
    transition_bandwidth: float
        Percentage between 0 and 0.5
    Returns
    -------
    np.ndarray
    """
    M = int(np.rint(4.0 / transition_bandwidth))
    x = np.arange(-(M // 2), 1 + (M // 2))
    fc = np.transpose(np.atleast_2d(fc / sampling_frequency))
    kernel = np.sinc(2 * fc * x)
    kernel = kernel * np.blackman(len(x))
    kernel = np.transpose(kernel)
    kernel = kernel / kernel.sum(0)

    if filter_type == "lowpass":
        return kernel.flatten()
    elif filter_type == "highpass":
        return _compute_spectral_inversion(kernel.flatten())
    elif filter_type == "bandstop":
        kernel[:, 1] = _compute_spectral_inversion(kernel[:, 1])
        kernel = np.sum(kernel, axis=1)
        return kernel
    elif filter_type == "bandpass":
        kernel[:, 1] = _compute_spectral_inversion(kernel[:, 1])
        kernel = _compute_spectral_inversion(np.sum(kernel, axis=1))
        return kernel
    else:
        raise ValueError


def _compute_windowed_sinc_filter(
    data, freq, filter_type, sampling_frequency, transition_bandwidth=0.02
):
    """
    Apply a windowed-sinc filter to the provided signal.

    Parameters
    ----------
    data: Tsd, TsdFrame or TsdTensor

    freq: float or tuple of float
        Cutting frequency in Hz. Single float for 'lowpass' and 'highpass'. Tuple of float for
        'bandpass' and 'bandstop'.
    sampling_frequency: float
        Sampling frequency in Hz.
    filter_type: str
        Either 'lowpass', 'highpass', 'bandstop' or 'bandpass'.
    transition_bandwidth: float
        Percentage between 0 and 0.5
    Returns
    -------
    Tsd, TsdFrame or TsdTensor
    """
    kernel = _get_windowed_sinc_kernel(
        freq, filter_type, sampling_frequency, transition_bandwidth
    )
    return data.convolve(kernel)


@_validate_filtering_inputs
def _compute_filter(
    data,
    cutoff,
    fs=None,
    mode="butter",
    order=4,
    transition_bandwidth=0.02,
    filter_type="bandpass",
):
    """
    Filter the signal.
    """
    if not isinstance(data, nap.time_series._BaseTsd):
        raise ValueError(
            f"Invalid value: {data}. First argument should be of type Tsd, TsdFrame or TsdTensor"
        )

    if np.any(np.isnan(data)):
        raise ValueError(
            "The input signal contains NaN values, which are not supported for filtering. "
            "Please remove or handle NaNs before applying the filter. "
            "You can use the `dropna()` method to drop all NaN values."
        )

    if fs is None:
        fs = data.rate

    cutoff = np.array(cutoff, dtype=float)

    if mode == "butter":
        return _compute_butterworth_filter(
            data, cutoff, fs, filter_type=filter_type, order=order
        )
    if mode == "sinc":
        return _compute_windowed_sinc_filter(
            data, cutoff, filter_type, fs, transition_bandwidth=transition_bandwidth
        )
    else:
        raise ValueError("Unrecognized filter mode. Choose either 'butter' or 'sinc'")


def apply_bandpass_filter(
    data, cutoff, fs=None, mode="butter", order=4, transition_bandwidth=0.02
):
    """
    Apply a band-pass filter to the provided signal.
    Mode can be :

    - `"butter"` for Butterworth filter. In this case, `order` determines the order of the filter.
    - `"sinc"` for Windowed-Sinc convolution. `transition_bandwidth` determines the transition bandwidth.

    Parameters
    ----------
    data : Tsd, TsdFrame, or TsdTensor
        The signal to be filtered.
    cutoff : (Numeric, Numeric)
        Cutoff frequencies in Hz.
    fs : float, optional
        The sampling frequency of the signal in Hz. If not provided, it will be inferred from the time axis of the data.
    mode : {'butter', 'sinc'}, optional
        Filtering mode. Default is 'butter'.
    order : int, optional
        The order of the Butterworth filter. Higher values result in sharper frequency cutoffs.
        Default is 4.
    transition_bandwidth : float, optional
        The transition bandwidth. 0.2 corresponds to 20% of the frequency band between 0 and the sampling frequency.
        The smaller the transition bandwidth, the larger the windowed-sinc kernel.
        Default is 0.02.

    Returns
    -------
    filtered_data : Tsd, TsdFrame, or TsdTensor
        The filtered signal, with the same data type as the input.

    Raises
    ------
    ValueError
        If `data` is not a Tsd, TsdFrame, or TsdTensor.
        If `cutoff` is not a tuple of two floats for "bandpass" and "bandstop" filters.
        If `fs` is not float or None.
        If `mode` is not "butter" or "sinc".
        If `order` is not an int.
        If "transition_bandwidth" is not a float.
    Notes
    -----
    For the Butterworth filter, the cutoff frequency is defined as the frequency at which the amplitude of the signal
    is reduced by -3 dB (decibels).
    """
    return _compute_filter(
        data,
        cutoff,
        fs=fs,
        mode=mode,
        order=order,
        transition_bandwidth=transition_bandwidth,
        filter_type="bandpass",
    )


def apply_bandstop_filter(
    data, cutoff, fs=None, mode="butter", order=4, transition_bandwidth=0.02
):
    """
    Apply a band-stop filter to the provided signal.
    Mode can be :

    - `"butter"` for Butterworth filter. In this case, `order` determines the order of the filter.
    - `"sinc"` for Windowed-Sinc convolution. `transition_bandwidth` determines the transition bandwidth.

    Parameters
    ----------
    data : Tsd, TsdFrame, or TsdTensor
        The signal to be filtered.
    cutoff : (Numeric, Numeric)
        Cutoff frequencies in Hz.
    fs : float, optional
        The sampling frequency of the signal in Hz. If not provided, it will be inferred from the time axis of the data.
    mode : {'butter', 'sinc'}, optional
        Filtering mode. Default is 'butter'.
    order : int, optional
        The order of the Butterworth filter. Higher values result in sharper frequency cutoffs.
        Default is 4.
    transition_bandwidth : float, optional
        The transition bandwidth. 0.2 corresponds to 20% of the frequency band between 0 and the sampling frequency.
        The smaller the transition bandwidth, the larger the windowed-sinc kernel.
        Default is 0.02.

    Returns
    -------
    filtered_data : Tsd, TsdFrame, or TsdTensor
        The filtered signal, with the same data type as the input.

    Raises
    ------
    ValueError
        If `data` is not a Tsd, TsdFrame, or TsdTensor.
        If `cutoff` is not a tuple of two floats for "bandpass" and "bandstop" filters.
        If `fs` is not float or None.
        If `mode` is not "butter" or "sinc".
        If `order` is not an int.
        If "transition_bandwidth" is not a float.
    Notes
    -----
    For the Butterworth filter, the cutoff frequency is defined as the frequency at which the amplitude of the signal
    is reduced by -3 dB (decibels).
    """
    return _compute_filter(
        data,
        cutoff,
        fs=fs,
        mode=mode,
        order=order,
        transition_bandwidth=transition_bandwidth,
        filter_type="bandstop",
    )


def apply_highpass_filter(
    data, cutoff, fs=None, mode="butter", order=4, transition_bandwidth=0.02
):
    """
    Apply a high-pass filter to the provided signal.
    Mode can be :

    - `"butter"` for Butterworth filter. In this case, `order` determines the order of the filter.
    - `"sinc"` for Windowed-Sinc convolution. `transition_bandwidth` determines the transition bandwidth.

    Parameters
    ----------
    data : Tsd, TsdFrame, or TsdTensor
        The signal to be filtered.
    cutoff : Numeric
        Cutoff frequency in Hz.
    fs : float, optional
        The sampling frequency of the signal in Hz. If not provided, it will be inferred from the time axis of the data.
    mode : {'butter', 'sinc'}, optional
        Filtering mode. Default is 'butter'.
    order : int, optional
        The order of the Butterworth filter. Higher values result in sharper frequency cutoffs.
        Default is 4.
    transition_bandwidth : float, optional
        The transition bandwidth. 0.2 corresponds to 20% of the frequency band between 0 and the sampling frequency.
        The smaller the transition bandwidth, the larger the windowed-sinc kernel.
        Default is 0.02.

    Returns
    -------
    filtered_data : Tsd, TsdFrame, or TsdTensor
        The filtered signal, with the same data type as the input.

    Raises
    ------
    ValueError
        If `data` is not a Tsd, TsdFrame, or TsdTensor.
        If `cutoff` is not a number.
        If `fs` is not float or None.
        If `mode` is not "butter" or "sinc".
        If `order` is not an int.
        If "transition_bandwidth" is not a float.
    Notes
    -----
    For the Butterworth filter, the cutoff frequency is defined as the frequency at which the amplitude of the signal
    is reduced by -3 dB (decibels).
    """
    return _compute_filter(
        data,
        cutoff,
        fs=fs,
        mode=mode,
        order=order,
        transition_bandwidth=transition_bandwidth,
        filter_type="highpass",
    )


def apply_lowpass_filter(
    data, cutoff, fs=None, mode="butter", order=4, transition_bandwidth=0.02
):
    """
    Apply a low-pass filter to the provided signal.
    Mode can be :

    - `"butter"` for Butterworth filter. In this case, `order` determines the order of the filter.
    - `"sinc"` for Windowed-Sinc convolution. `transition_bandwidth` determines the transition bandwidth.

    Parameters
    ----------
    data : Tsd, TsdFrame, or TsdTensor
        The signal to be filtered.
    cutoff : Numeric
        Cutoff frequency in Hz.
    fs : float, optional
        The sampling frequency of the signal in Hz. If not provided, it will be inferred from the time axis of the data.
    mode : {'butter', 'sinc'}, optional
        Filtering mode. Default is 'butter'.
    order : int, optional
        The order of the Butterworth filter. Higher values result in sharper frequency cutoffs.
        Default is 4.
    transition_bandwidth : float, optional
        The transition bandwidth. 0.2 corresponds to 20% of the frequency band between 0 and the sampling frequency.
        The smaller the transition bandwidth, the larger the windowed-sinc kernel.
        Default is 0.02.

    Returns
    -------
    filtered_data : Tsd, TsdFrame, or TsdTensor
        The filtered signal, with the same data type as the input.

    Raises
    ------
    ValueError
        If `data` is not a Tsd, TsdFrame, or TsdTensor.
        If `cutoff` is not a number.
        If `fs` is not float or None.
        If `mode` is not "butter" or "sinc".
        If `order` is not an int.
        If "transition_bandwidth" is not a float.
    Notes
    -----
    For the Butterworth filter, the cutoff frequency is defined as the frequency at which the amplitude of the signal
    is reduced by -3 dB (decibels).
    """
    return _compute_filter(
        data,
        cutoff,
        fs=fs,
        mode=mode,
        order=order,
        transition_bandwidth=transition_bandwidth,
        filter_type="lowpass",
    )


@_validate_filtering_inputs
def get_filter_frequency_response(
    cutoff, fs, filter_type, mode, order=4, transition_bandwidth=0.02
):
    """
    Utility function to evaluate the frequency response of a particular type of filter. The arguments are the same
    as the function `apply_lowpass_filter`, `apply_highpass_filter`, `apply_bandpass_filter` and
    `apply_bandstop_filter`.

    This function returns a pandas Series object with the index as frequencies.

    Parameters
    ----------
    cutoff : Numeric or tuple of Numeric
        Cutoff frequency in Hz.
    fs : float
        The sampling frequency of the signal in Hz.
    filter_type: str
        Can be "lowpass", "highpass", "bandpass" or "bandstop"
    mode: str
        Can be "butter" or "sinc".
    order : int, optional
        The order of the Butterworth filter. Higher values result in sharper frequency cutoffs.
        Default is 4.
    transition_bandwidth : float, optional
        The transition bandwidth. 0.2 corresponds to 20% of the frequency band between 0 and the sampling frequency.
        The smaller the transition bandwidth, the larger the windowed-sinc kernel.
        Default is 0.02.

    Returns
    -------
    pandas.Series
    """
    cutoff = np.array(cutoff)

    if mode == "butter":
        sos = _get_butter_coefficients(cutoff, filter_type, fs, order)
        w, h = sosfreqz(sos, worN=1024, fs=fs)
        return pd.Series(index=w, data=np.abs(h))
    if mode == "sinc":
        kernel = _get_windowed_sinc_kernel(
            cutoff, filter_type, fs, transition_bandwidth
        )
        fft_result = np.fft.fft(kernel)
        fft_result = np.fft.fftshift(fft_result)
        fft_freq = np.fft.fftfreq(n=len(kernel), d=1 / fs)
        fft_freq = np.fft.fftshift(fft_freq)
        return pd.Series(
            index=fft_freq[fft_freq >= 0], data=np.abs(fft_result[fft_freq >= 0])
        )
    else:
        raise ValueError("Unrecognized filter mode. Choose either 'butter' or 'sinc'")
