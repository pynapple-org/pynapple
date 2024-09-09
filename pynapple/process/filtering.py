"""Filtering module."""

import inspect
from functools import wraps
from numbers import Number

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .. import core as nap


def _get_butter_coefficients(cutoff, filter_type, sampling_frequency, order=4):
    return butter(order, cutoff, btype=filter_type, fs=sampling_frequency, output="sos")


def _compute_butterworth_filter(
    data, cutoff, sampling_frequency=None, filter_type="bandpass", order=4
):
    """
    Apply a Butterworth filter to the provided signal.
    """
    sos = _get_butter_coefficients(cutoff, filter_type, sampling_frequency, order)
    out = np.zeros_like(data.d)
    for ep in data.time_support:
        slc = data.get_slice(start=ep.start[0], end=ep.end[0])
        out[slc] = sosfiltfilt(sos, data.d[slc], axis=0)

    kwargs = dict(t=data.t, d=out, time_support=data.time_support)
    if isinstance(data, nap.TsdFrame):
        kwargs["columns"] = data.columns
    return data.__class__(**kwargs)


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


def _get_windowed_sinc_kernel(fc, filter_type="lowpass", transition_bandwidth=0.02):
    """
    Get the windowed-sinc kernel.
    Smith, S. (2003). Digital signal processing: a practical guide for engineers and scientists.
    Chapter 16, equation 16-4

    Parameters
    ----------
    fc: float or tuple of float
        Cutting frequency between 0 and 0.5. Single float for 'lowpass' and 'highpass'. Tuple of float for
        'bandpass' and 'bandstop'.
    filter_type: str
        Either 'lowpass', 'highpass', 'bandstop' or 'bandpass'.
    transition_bandwidth: float
        Percentage between 0 and 0.5
    Returns
    -------
    np.ndarray
    """
    M = int(np.rint(20.0 / transition_bandwidth))
    x = np.arange(-(M // 2), 1 + (M // 2))
    fc = np.transpose(np.atleast_2d(fc))
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
    data, freq, sampling_frequency, filter_type="lowpass", transition_bandwidth=0.02
):
    """
    Apply a windowed-sinc filter to the provided signal.

    Parameters
    ----------
    filter_type
    """
    kernel = _get_windowed_sinc_kernel(
        freq / sampling_frequency, filter_type, transition_bandwidth
    )
    return data.convolve(kernel)


def _validate_filtering_inputs(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Validate each positional argument
        sig = inspect.signature(func)
        kwargs = sig.bind_partial(*args, **kwargs).arguments

        if "data" not in kwargs or "cutoff" not in kwargs:
            raise TypeError(
                "Function needs time series and cutoff frequency to be specified."
            )

        if not isinstance(kwargs["data"], nap.time_series.BaseTsd):
            raise ValueError(
                f"Invalid value: {args[0]}. First argument should be of type Tsd, TsdFrame or TsdTensor"
            )

        if not isinstance(kwargs["cutoff"], Number):
            if len(kwargs["cutoff"]) != 2 or not all(
                isinstance(fq, Number) for fq in kwargs["cutoff"]
            ):
                raise ValueError

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

        if np.any(np.isnan(kwargs["data"])):
            raise ValueError(
                "The input signal contains NaN values, which are not supported for filtering. "
                "Please remove or handle NaNs before applying the filter. "
                "You can use the `dropna()` method to drop all NaN values."
            )

        # Call the original function with validated inputs
        return func(**kwargs)

    return wrapper


@_validate_filtering_inputs
def compute_bandpass_filter(
    data, cutoff, fs=None, mode="butter", order=4, transition_bandwidth=0.02
):
    """
    Apply a band-pass filter to the provided signal.
    Mode can be :
        - 'butter' for Butterworth filter. In this case, `order` determines the order of the filter.
        - 'sinc' for Windowed-Sinc convolution. `transition_bandwidth` determines the transition bandwidth.

    Parameters
    ----------
    data : Tsd, TsdFrame, or TsdTensor
        The signal to be filtered.
    cutoff : tuple of (float, float)
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
    if fs is None:
        fs = data.rate

    cutoff = np.array(cutoff)

    if mode == "butter":
        return _compute_butterworth_filter(
            data, cutoff, fs, filter_type="bandpass", order=order
        )
    if mode == "sinc":
        return _compute_windowed_sinc_filter(
            data,
            cutoff,
            fs,
            filter_type="bandpass",
            transition_bandwidth=transition_bandwidth,
        )
    else:
        raise ValueError("Unrecognized filter mode. Choose either 'butter' or 'sinc'")


@_validate_filtering_inputs
def compute_bandstop_filter(
    data, cutoff, fs=None, mode="butter", order=4, transition_bandwidth=0.02
):
    """
    Apply a band-stop filter to the provided signal.
    Mode can be :
        - 'butter' for Butterworth filter. In this case, `order` determines the order of the filter.
        - 'sinc' for Windowed-Sinc convolution. `transition_bandwidth` determines the transition bandwidth.

    Parameters
    ----------
    data : Tsd, TsdFrame, or TsdTensor
        The signal to be filtered.
    cutoff : tuple of (float, float)
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
    if fs is None:
        fs = data.rate

    cutoff = np.array(cutoff)

    if mode == "butter":
        return _compute_butterworth_filter(
            data, cutoff, fs, filter_type="bandstop", order=order
        )
    elif mode == "sinc":
        return _compute_windowed_sinc_filter(
            data,
            cutoff,
            fs,
            filter_type="bandstop",
            transition_bandwidth=transition_bandwidth,
        )
    else:
        raise ValueError("Unrecognized filter mode. Choose either 'butter' or 'sinc'")


@_validate_filtering_inputs
def compute_highpass_filter(
    data, cutoff, fs=None, mode="butter", order=4, transition_bandwidth=0.02
):
    """
    Apply a high-pass filter to the provided signal.
    Mode can be :
        - 'butter' for Butterworth filter. In this case, `order` determines the order of the filter.
        - 'sinc' for Windowed-Sinc convolution. `transition_bandwidth` determines the transition bandwidth.

    Parameters
    ----------
    data : Tsd, TsdFrame, or TsdTensor
        The signal to be filtered.
    cutoff : float
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
    if fs is None:
        fs = data.rate

    if mode == "butter":
        return _compute_butterworth_filter(
            data, cutoff, fs, filter_type="highpass", order=order
        )
    elif mode == "sinc":
        return _compute_windowed_sinc_filter(
            data,
            cutoff,
            fs,
            filter_type="highpass",
            transition_bandwidth=transition_bandwidth,
        )
    else:
        raise ValueError("Unrecognized filter mode. Choose either 'butter' or 'sinc'")


@_validate_filtering_inputs
def compute_lowpass_filter(
    data, cutoff, fs=None, mode="butter", order=4, transition_bandwidth=0.02
):
    """
    Apply a low-pass filter to the provided signal.
    Mode can be :
        - 'butter' for Butterworth filter. In this case, `order` determines the order of the filter.
        - 'sinc' for Windowed-Sinc convolution. `transition_bandwidth` determines the transition bandwidth.

    Parameters
    ----------
    data : Tsd, TsdFrame, or TsdTensor
        The signal to be filtered.
    cutoff : float
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
    if fs is None:
        fs = data.rate

    if mode == "butter":
        return _compute_butterworth_filter(
            data, cutoff, fs, filter_type="lowpass", order=order
        )
    elif mode == "sinc":
        return _compute_windowed_sinc_filter(
            data,
            cutoff,
            fs,
            filter_type="lowpass",
            transition_bandwidth=transition_bandwidth,
        )
    else:
        raise ValueError("Unrecognized filter mode. Choose either 'butter' or 'sinc'")
