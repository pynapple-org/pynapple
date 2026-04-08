"""
Functions to compute phases and envelopes
"""

import numpy as np

import pynapple as nap


def apply_hilbert_transform(data):
    """
    Apply the Hilbert transform to a time-series.

    This function wraps :func:`scipy.signal.hilbert` to compute the analytic signal,
    which represents the original signal plus its Hilbert transform.
    The Hilbert transform is commonly used for phase and envelope computations.

    Parameters
    ----------
    data : Tsd, TsdFrame
        The time-series to which the Hilbert transform will be applied.

    Returns
    -------
    Tsd, TsdFrame
        The analytic signal.

    Examples
    --------
    >>> import numpy as np
    >>> import pynapple as nap
    >>> times = np.arange(0, 20, 0.1)
    >>> data = nap.Tsd(d=np.sin(times), t=times)
    >>> analytic_signal = nap.apply_hilbert_transform(data)
    >>> analytic_signal
    Time (s)
    ----------  ---------------------------------------------
    0.0         (-7.105427357601002e-17+0.16863846755783507j)
    0.1         (0.09983341664682804-0.4242708612183068j)
    0.2         (0.1986693307950611-0.39707635680514186j)
    0.3         (0.2955202066613397-0.5595039695854611j)
    0.4         (0.3894183423086504-0.5111880915563278j)
    0.5         (0.4794255386042027-0.5737652837793169j)
    0.6         (0.5646424733950355-0.507679928825233j)
    ...
    19.3        (0.4353653603728936-0.8487963381972523j)
    19.4        (0.5230657651576995-0.8346984246178151j)
    19.5        (0.6055398697196014-0.6962597876075753j)
    19.6        (0.6819636200681357-0.673794027075586j)
    19.7        (0.7515734153521505-0.4510063664570575j)
    19.8        (0.8136737375071058-0.4288655114005536j)
    19.9        (0.8676441006416694+0.17871162129618953j)
    dtype: complex128, shape: (200,)

    Can be used for multiple signals in a `TsdFrame`:

    >>> data = nap.TsdFrame(d=np.stack([np.sin(times), np.cos(times)], axis=1), t=times)
    >>> analytic_signals = nap.apply_hilbert_transform(data)
    >>> analytic_signals
    Time (s)    0                                              1
    ----------  ---------------------------------------------  ------------------------------------------
    0.0         (-7.105427357601002e-17+0.16863846755783507j)  (0.9999999999999999-0.10933857636723118j)
    0.1         (0.09983341664682804-0.4242708612183068j)      (0.9950041652780255+0.2511675765027083j)
    0.2         (0.1986693307950611-0.39707635680514186j)      (0.9800665778412415+0.29226919446209765j)
    0.3         (0.2955202066613397-0.5595039695854611j)       (0.9553364891256061+0.45596645091122484j)
    0.4         (0.3894183423086504-0.5111880915563278j)       (0.9210609940028853+0.5095457107948864j)
    0.5         (0.4794255386042027-0.5737652837793169j)       (0.8775825618903729+0.6332440634658392j)
    0.6         (0.5646424733950355-0.507679928825233j)        (0.8253356149096783+0.6873614294191187j)
    ...
    19.3        (0.4353653603728936-0.8487963381972523j)       (0.9002538547473041+0.09873915660984472j)
    19.4        (0.5230657651576995-0.8346984246178151j)       (0.852292323865463+0.18298411058238345j)
    19.5        (0.6055398697196014-0.6962597876075753j)       (0.7958149698139438+0.19019044271235275j)
    19.6        (0.6819636200681357-0.673794027075586j)        (0.7313860956454965+0.2587502761655236j)
    19.7        (0.7515734153521505-0.4510063664570575j)       (0.6596494533734591+0.1992088667212628j)
    19.8        (0.8136737375071058-0.4288655114005536j)       (0.5813218118144358+0.24323915864085138j)
    19.9        (0.8676441006416694+0.17871162129618953j)      (0.49718579487120196-0.09195658451657955j)
    dtype: complex128, shape: (200, 2)
    """
    from scipy.signal import hilbert

    if isinstance(data, nap.Tsd):
        return nap.Tsd(
            d=hilbert(data.values),
            t=data.times(),
            time_support=data.time_support,
        )
    elif isinstance(data, nap.TsdFrame):
        return nap.TsdFrame(
            d=hilbert(data.values, axis=0),
            t=data.times(),
            columns=data.columns,
            time_support=data.time_support,
        )
    else:
        raise TypeError("data should be a Tsd or TsdFrame.")


def compute_hilbert_envelope(data):
    """
    Compute the Hilbert envelope of a time-series.

    This function computes the envelope of the signal, which is the magnitude of the analytic
    signal obtained by applying the Hilbert transform. The envelope provides a smooth
    representation of the amplitude modulation of the signal.

    Parameters
    ----------
    data : Tsd, TsdFrame
        The time-series data to compute the Hilbert envelope for.

    Returns
    -------
    Tsd, TsdFrame
        The Hilbert envelope.

    Examples
    --------
    >>> import numpy as np
    >>> import pynapple as nap
    >>> times = np.arange(0, 20, 0.1)
    >>> data = nap.Tsd(d=np.sin(times), t=times)
    >>> envelope = nap.compute_hilbert_envelope(data)
    >>> envelope
    Time (s)
    ----------  --------
    0.0         0.168638
    0.1         0.435858
    0.2         0.444004
    0.3         0.632753
    0.4         0.64262
    0.5         0.7477
    0.6         0.759316
    ...
    19.3        0.953938
    19.4        0.985048
    19.5        0.922744
    19.6        0.958683
    19.7        0.87651
    19.8        0.919777
    19.9        0.885858
    dtype: float64, shape: (200,)

    Can be used for multiple signals in a `TsdFrame`:

    >>> data = nap.TsdFrame(d=np.stack([np.sin(times), np.cos(times)], axis=1), t=times)
    >>> envelopes = nap.compute_hilbert_envelope(data)
    >>> envelopes
    Time (s)           0         1
    ----------  --------  --------
    0.0         0.168638  1.00596
    0.1         0.435858  1.02622
    0.2         0.444004  1.02272
    0.3         0.632753  1.05857
    0.4         0.64262   1.05261
    0.5         0.7477    1.0822
    0.6         0.759316  1.07408
    ...
    19.3        0.953938  0.905652
    19.4        0.985048  0.871714
    19.5        0.922744  0.818226
    19.6        0.958683  0.775808
    19.7        0.87651   0.689073
    19.8        0.919777  0.630159
    19.9        0.885858  0.505618
    dtype: float64, shape: (200, 2)
    """
    analytic_signal = apply_hilbert_transform(data)
    return np.abs(analytic_signal)


def compute_hilbert_phase(data):
    """
    Compute the Hilbert phase of a time-series.

    This function computes the instantaneous phase of the signal using the Hilbert transform.
    The phase is unwrapped to provide a continuous representation, and it is then wrapped to
    ensure it stays within the range [0, 2π].

    Parameters
    ----------
    data : Tsd, TsdFrame
        The time-series data to compute the Hilbert phase for.

    Returns
    -------
    Tsd, TsdFrame
        The instantaneous phase of the signal, with values wrapped between [0, 2π].

    Examples
    --------
    >>> import numpy as np
    >>> import pynapple as nap
    >>> times = np.arange(0, 20, 0.1)
    >>> data = nap.Tsd(d=np.sin(times), t=times)
    >>> phase = nap.compute_hilbert_envelope(data)
    >>> phase
    Time (s)
    ----------  --------
    0.0         0.168638
    0.1         0.435858
    0.2         0.444004
    0.3         0.632753
    0.4         0.64262
    0.5         0.7477
    0.6         0.759316
    ...
    19.3        0.953938
    19.4        0.985048
    19.5        0.922744
    19.6        0.958683
    19.7        0.87651
    19.8        0.919777
    19.9        0.885858
    dtype: float64, shape: (200,)

    Can be used for multiple signals in a `TsdFrame`:

    >>> data = nap.TsdFrame(d=np.stack([np.sin(times), np.cos(times)], axis=1), t=times)
    >>> phases = nap.compute_hilbert_envelope(data)
    >>> phases
    Time (s)           0         1
    ----------  --------  --------
    0.0         0.168638  1.00596
    0.1         0.435858  1.02622
    0.2         0.444004  1.02272
    0.3         0.632753  1.05857
    0.4         0.64262   1.05261
    0.5         0.7477    1.0822
    0.6         0.759316  1.07408
    ...
    19.3        0.953938  0.905652
    19.4        0.985048  0.871714
    19.5        0.922744  0.818226
    19.6        0.958683  0.775808
    19.7        0.87651   0.689073
    19.8        0.919777  0.630159
    19.9        0.885858  0.505618
    dtype: float64, shape: (200, 2)
    """
    analytic_signal = apply_hilbert_transform(data)
    phase = np.angle(analytic_signal)
    phase = np.mod(np.unwrap(phase), 2 * np.pi)
    return phase


def detect_oscillatory_events(
    data,
    epoch,
    freq_band,
    thresh_band,
    duration_band,
    min_inter_duration,
    fs=None,
    wsize=51,
):
    """
    Simple helper for detecting oscillatory events (e.g. ripples, spindles)

    Parameters
    ----------
    data : Tsd
        1-dimensional time series
    epoch : IntervalSet
        The epoch for restricting the detection
    freq_band : tuple
        The (low, high) frequency to bandpass the signal
    thresh_band : tuple
        The (min, max) value for thresholding the normalized squared signal after filtering
    duration_band : tuple
        The (min, max) duration of an event in second
    min_inter_duration : float
        The minimum duration between two events otherwise they are merged (in seconds)
    fs : float, optional
        The sampling frequency of the signal in Hz. If not provided, it will be inferred from the time axis of the data.
    wsize : int, optional
        The size of the window for digital filtering

    Returns
    -------
    IntervalSet
        The interval set of detected events with metadata containing
        the power, amplitude, and peak_time
    """
    import warnings

    from scipy.signal import filtfilt

    data = data.restrict(epoch)

    if fs is None:
        fs = data.rate

    signal = nap.apply_bandpass_filter(data, freq_band, fs)
    squared_signal = np.square(signal.values)
    window = np.ones(wsize) / wsize

    nSS = filtfilt(window, 1, squared_signal)
    nSS = (nSS - np.mean(nSS)) / np.std(nSS)
    nSS = nap.Tsd(t=signal.index.values, d=nSS, time_support=epoch)

    # Detect oscillation periods by thresholding normalized signal
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Some epochs have no duration",
            category=UserWarning,
        )
        nSS2 = nSS.threshold(thresh_band[0], method="above")
        nSS3 = nSS2.threshold(thresh_band[1], method="below")

    # Exclude oscillation where min_duration < length < max_duration
    osc_ep = nSS3.time_support
    osc_ep = osc_ep.drop_short_intervals(duration_band[0], time_units="s")
    osc_ep = osc_ep.drop_long_intervals(duration_band[1], time_units="s")

    # Merge if inter-oscillation period is too short
    osc_ep = osc_ep.merge_close_intervals(min_inter_duration, time_units="s")

    # Compute power, amplitude, and peak_time for each interval
    powers = []
    amplitudes = []
    peak_times = []

    for s, e in osc_ep.values:
        seg = signal.get(s, e)
        if len(seg) == 0:
            powers.append(np.nan)
            amplitudes.append(np.nan)
            peak_times.append(np.nan)
            continue
        power = np.mean(np.square(seg))
        power_db = 10 * np.log10(power)
        amplitude = np.max(np.abs(seg.values))
        peak_idx = np.argmax(np.abs(seg.values))
        peak_time = seg.index.values[peak_idx]
        powers.append(power_db)
        amplitudes.append(amplitude)
        peak_times.append(peak_time)

    metadata = {
        "power": powers,
        "amplitude": amplitudes,
        "peak_time": peak_times,
    }

    return nap.IntervalSet(start=osc_ep.start, end=osc_ep.end, metadata=metadata)
