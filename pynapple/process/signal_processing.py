import numpy as np
from itertools import repeat
import pynapple as nap
from tqdm import tqdm
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------

def compute_fft(sig, fs):
    """  
    Performs numpy fft on sig, returns output  
    ..todo: Make sig handle TsdFrame, TsdTensor  

    :param sig:    :param fs:    :return:  
    """
    if not isinstance(sig, nap.Tsd):
        raise TypeError("Currently compute_fft is only implemented for Tsd")
    fft_result = np.fft.fft(sig.values)
    fft_freq = np.fft.fftfreq(len(sig.values), 1 / fs)
    return fft_result, fft_freq


def morlet(M, ncycles=5.0, scaling=1.0):
    """  
    Defines the complex Morelet wavelet  
    :param M: Length of the wavelet.    :param ncycles: number of wavelet cycles to use. Default is 5    :param scaling: Scaling factor. Default is 1.    :return: (M,) ndarray Morelet wavelet  
    """
    x = np.linspace(-scaling * 2 * np.pi, scaling * 2 * np.pi, M)
    return np.exp(1j * ncycles * x) * (np.exp(-0.5 * (x ** 2)) * np.pi ** (-0.25))


"""  
The following code has been adapted from functions in the neurodsp package:  
https://github.com/neurodsp-tools/neurodsp  

..todo: reference licence in LICENCE directory  
"""


def check_n_cycles(n_cycles, len_cycles=None):
    """Check an input as a number of cycles definition, and make it iterable.

    Parameters    ----------    n_cycles : float or list        Definition of number of cycles.        If a single value, the same number of cycles is used for each frequency value.        If a list or list_like, then should be a n_cycles corresponding to each frequency.    len_cycles : int, optional        What the length of `n_cycles` should, if it's a list.
    Returns    -------    n_cycles : iterable        An iterable version of the number of cycles.    """
    if isinstance(n_cycles, (int, float, np.number)):

        if n_cycles <= 0:
            raise ValueError('Number of cycles must be a positive number.')

        n_cycles = repeat(n_cycles)

    elif isinstance(n_cycles, (tuple, list, np.ndarray)):

        for cycle in n_cycles:
            if cycle <= 0:
                raise ValueError('Each number of cycles must be a positive number.')

        if len_cycles and len(n_cycles) != len_cycles:
            raise ValueError('The length of number of cycles does not match other inputs.')

        n_cycles = iter(n_cycles)

    return n_cycles


def create_freqs(freq_start, freq_stop, freq_step=1):
    """Create an array of frequencies.

    Parameters    ----------    freq_start : float        Starting value for the frequency definition.    freq_stop : float        Stopping value for the frequency definition, inclusive.    freq_step : float, optional, default: 1        Step value, for linearly spaced values between start and stop.
    Returns    -------    freqs : 1d array        Frequency indices.    """
    return np.arange(freq_start, freq_stop + freq_step, freq_step)


def compute_wavelet_transform(sig, fs, freqs, n_cycles=7, scaling=0.5, norm='amp'):
    """Compute the time-frequency representation of a signal using morlet wavelets.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    freqs : 1d array or list of float
        If array, frequency values to estimate with morlet wavelets.
        If list, define the frequency range, as [freq_start, freq_stop, freq_step].
        The `freq_step` is optional, and defaults to 1. Range is inclusive of `freq_stop` value.
    n_cycles : float or 1d array
        Length of the filter, as the number of cycles for each frequency.
        If 1d array, this defines n_cycles for each frequency.
    scaling : float
        Scaling factor.
    norm : {'sss', 'amp'}, optional
        Normalization method:

        * 'sss' - divide by the square root of the sum of squares
        * 'amp' - divide by the sum of amplitudes

    Returns
    -------
    mwt : 2d array
        Time frequency representation of the input signal.

    Notes
    -----
    This computes the continuous wavelet transform at specified frequencies across time.

    Examples
    --------
    Compute a Morlet wavelet time-frequency representation of a signal:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> mwt = compute_wavelet_transform(sig, fs=500, freqs=[1, 30])
    """
    if not isinstance(sig, nap.Tsd) and not isinstance(sig, nap.TsdFrame):
        raise TypeError("`sig` must be instance of Tsd or TsdFrame")

    if isinstance(freqs, (tuple, list)):
        freqs = create_freqs(*freqs)
    n_cycles = check_n_cycles(n_cycles, len(freqs))
    if isinstance(sig, nap.Tsd):
        mwt = np.zeros([len(freqs), len(sig)], dtype=complex)
        for ind, (freq, n_cycle) in enumerate(zip(freqs, n_cycles)):
            wav = convolve_wavelet(sig, fs, freq, n_cycle, scaling, norm=norm)
            mwt[ind, :] = wav
        return nap.TsdFrame(t=sig.index, d=np.transpose(mwt))
    else:
        mwt = np.zeros([sig.values.shape[0], len(freqs), sig.values.shape[1]], dtype=complex)
        for channel_i in tqdm(range(sig.values.shape[1])):
            for ind, (freq, n_cycle) in enumerate(zip(freqs, n_cycles)):
                wav = convolve_wavelet(sig[:, channel_i], fs, freq, n_cycle, scaling, norm=norm)
                mwt[:, ind, channel_i] = wav
        return nap.TsdTensor(t=sig.index, d=mwt)


def convolve_wavelet(sig, fs, freq, n_cycles=7, scaling=0.5, wavelet_len=None, norm='sss'):
    """Convolve a signal with a complex wavelet.

    Parameters
    ----------
    sig : 1d array
        Time series to filter.
    fs : float
        Sampling rate, in Hz.
    freq : float
        Center frequency of bandpass filter.
    n_cycles : float, optional, default: 7
        Length of the filter, as the number of cycles of the oscillation with specified frequency.
    scaling : float, optional, default: 0.5
        Scaling factor for the morlet wavelet.
    wavelet_len : int, optional
        Length of the wavelet. If defined, this overrides the freq and n_cycles inputs.
    norm : {'sss', 'amp'}, optional
        Normalization method:

        * 'sss' - divide by the square root of the sum of squares
        * 'amp' - divide by the sum of amplitudes

    Returns
    -------
    array
        Complex time series.

    Notes
    -----

    * The real part of the returned array is the filtered signal.
    * Taking np.abs() of output gives the analytic amplitude.
    * Taking np.angle() of output gives the analytic phase.

    Examples
    --------
    Convolve a complex wavelet with a simulated signal:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> cts = convolve_wavelet(sig, fs=500, freq=10)
    """
    if norm not in ['sss', 'amp']:
        raise ValueError('Given `norm` must be `sss` or `amp`')

    if wavelet_len is None:
        wavelet_len = int(n_cycles * fs / freq)

    if wavelet_len > sig.shape[-1]:
        raise ValueError('The length of the wavelet is greater than the signal. Can not proceed.')

    morlet_f = morlet(wavelet_len, ncycles=n_cycles, scaling=scaling)

    if norm == 'sss':
        morlet_f = morlet_f / np.sqrt(np.sum(np.abs(morlet_f) ** 2))
    elif norm == 'amp':
        morlet_f = morlet_f / np.sum(np.abs(morlet_f))

    mwt_real = sig.convolve(np.real(morlet_f))
    mwt_imag = sig.convolve(np.imag(morlet_f))

    return mwt_real.values + 1j * mwt_imag.values