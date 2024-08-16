"""Filtering module."""

from numbers import Number

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .. import core as nap


def compute_filtered_signal(
    data, freq_band, fs=None, filter_type="bandpass", order=4
):
    """
    Apply a Butterworth filter to the provided signal.

    This function performs bandpass filtering on time series data
    using a Butterworth filter. The filter can be configured to be of
    type "bandpass", "bandstop", "highpass", or "lowpass".

    Parameters
    ----------
    data : Tsd, TsdFrame, or TsdTensor
        The signal to be filtered.
    freq_band : tuple of (float, float) or float
        Cutoff frequency(ies) in Hz.
        - For "bandpass" and "bandstop" filters, provide a tuple specifying the two cutoff frequencies.
        - For "lowpass" and "highpass" filters, provide a single float specifying the cutoff frequency.
    filter_type : {'bandpass', 'bandstop', 'highpass', 'lowpass'}, optional
        The type of frequency filter to apply. Default is "bandpass".
    order : int, optional
        The order of the Butterworth filter. Higher values result in sharper frequency cutoffs.
        Default is 4.
    sampling_frequency : float, optional
        The sampling frequency of the signal in Hz. If not provided, it will be inferred from the time axis of the data.

    Returns
    -------
    filtered_data : Tsd, TsdFrame, or TsdTensor
        The filtered signal, with the same data type as the input.

    Raises
    ------
    ValueError
        If `filter_type` is not one of {"bandpass", "bandstop", "highpass", "lowpass"}.
        If `freq_band` is not a float for "lowpass" and "highpass" filters.
        If `freq_band` is not a tuple of two floats for "bandpass" and "bandstop" filters.

    Notes
    -----
    The cutoff frequency is defined as the frequency at which the amplitude of the signal
    is reduced by -3 dB (decibels).
    """
    if sampling_frequency is None:
        sampling_frequency = data.rate

    if filter_type not in ["lowpass", "highpass", "bandpass", "bandstop"]:
        raise ValueError(
            f"Unrecognized filter type {filter_type}. "
            "filter_type must be either 'lowpass', 'highpass', 'bandpass',or 'bandstop'."
        )
    elif filter_type in ["lowpass", "highpass"] and not isinstance(freq_band, Number):
        raise ValueError(
            "Low/high-pass filter specification requires a single frequency. "
            f"{freq_band} provided instead!"
        )
    elif filter_type in ["bandpass", "bandstop"]:
        try:
            if len(freq_band) != 2 or not all(
                isinstance(fq, Number) for fq in freq_band
            ):
                raise ValueError
        except Exception:
            raise ValueError(
                "Band-pass/stop filter specification requires two frequencies. "
                f"{freq_band} provided instead!"
            )

    sos = butter(
        order, freq_band, btype=filter_type, fs=sampling_frequency, output="sos"
    )

    out = np.zeros_like(data.d)
    for ep in data.time_support:
        slc = data.get_slice(start=ep.start[0], end=ep.end[0])
        out[slc] = sosfiltfilt(sos, data.d[slc], axis=0)

    kwargs = dict(t=data.t, d=out, time_support=data.time_support)
    if isinstance(data, nap.TsdFrame):
        kwargs["columns"] = data.columns
    return data.__class__(**kwargs)
