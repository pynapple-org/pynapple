import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest
from scipy import signal

import pynapple as nap


############################################################
# Test for power_spectral_density
############################################################

def get_sorted_fft(data,fs):
    fft = np.fft.fft(data, axis=0)
    fft_freq = np.fft.fftfreq(len(data), 1 / fs)
    order = np.argsort(fft_freq)
    if fft.ndim==1:
        fft = fft[:,np.newaxis]
    return fft_freq[order], fft[order]

def test_compute_power_spectral_density():

    t = np.linspace(0, 1, 1000)
    sig = nap.Tsd(d=np.random.random(1000), t=t)
    r = nap.compute_power_spectral_density(sig)
    assert isinstance(r, pd.DataFrame)
    assert r.shape[0] == 500

    a, b = get_sorted_fft(sig.values, sig.rate)
    np.testing.assert_array_almost_equal(r.index.values, a[a>=0])
    np.testing.assert_array_almost_equal(r.values, b[a>=0])
    
    r = nap.compute_power_spectral_density(sig, norm=True)
    np.testing.assert_array_almost_equal(r.values, b[a>=0]/len(sig))

    sig = nap.TsdFrame(d=np.random.random((1000, 4)), t=t)
    r = nap.compute_power_spectral_density(sig)
    assert isinstance(r, pd.DataFrame)
    assert r.shape == (500, 4)

    a, b = get_sorted_fft(sig.values, sig.rate)
    np.testing.assert_array_almost_equal(r.index.values, a[a>=0])
    np.testing.assert_array_almost_equal(r.values, b[a>=0])

    sig = nap.TsdFrame(d=np.random.random((1000, 4)), t=t)
    r = nap.compute_power_spectral_density(sig, full_range=True)
    assert isinstance(r, pd.DataFrame)
    assert r.shape == (1000, 4)

    a, b = get_sorted_fft(sig.values, sig.rate)
    np.testing.assert_array_almost_equal(r.index.values, a)
    np.testing.assert_array_almost_equal(r.values, b)    

    t = np.linspace(0, 1, 1000)
    sig = nap.Tsd(d=np.random.random(1000), t=t)
    r = nap.compute_power_spectral_density(sig, ep=sig.time_support)
    assert isinstance(r, pd.DataFrame)
    assert r.shape[0] == 500

    t = np.linspace(0, 1, 1000)
    sig = nap.Tsd(d=np.random.random(1000), t=t)
    r = nap.compute_power_spectral_density(sig, fs=1000)
    assert isinstance(r, pd.DataFrame)
    assert r.shape[0] == 500


@pytest.mark.parametrize(
    "sig, fs, ep, full_range, norm, expectation",
    [
        (
            nap.Tsd(
                d=np.random.random(1000),
                t=np.linspace(0, 1, 1000),
                time_support=nap.IntervalSet(start=[0.1, 0.6], end=[0.2, 0.81]),
            ),
            1000,
            None,
            False,
            False,
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Given epoch (or signal time_support) must have length 1"
                ),
            ),
        ),
        (
            nap.Tsd(d=np.random.random(1000), t=np.linspace(0, 1, 1000)),
            1000,
            "not_ep",
            False,
            False,
            pytest.raises(
                TypeError,
                match="ep param must be a pynapple IntervalSet object, or None",
            ),
        ),
        (
            nap.Tsd(d=np.random.random(1000), t=np.linspace(0, 1, 1000)),
            "a",
            None,
            False,
            False,
            pytest.raises(
                TypeError,
                match="fs must be of type float or int",
            ),
        ),
        (
            "not_a_tsd",
            1000,
            None,
            False,
            False,
            pytest.raises(
                TypeError,
                match="sig must be either a Tsd or a TsdFrame object.",
            ),
        ),
        (
            nap.Tsd(d=np.random.random(1000), t=np.linspace(0, 1, 1000)),
            1000,
            None,
            "a",
            False,            
            pytest.raises(
                TypeError,
                match="full_range must be of type bool or None",
            ),
        ),
        (
            nap.Tsd(d=np.random.random(1000), t=np.linspace(0, 1, 1000)),
            1000,
            None,
            False,
            "a",
            pytest.raises(
                TypeError,
                match="norm must be of type bool",
            ),
        ),             
    ],
)
def test_compute_power_spectral_density_raise_errors(
    sig, fs, ep, full_range, norm, expectation
):
    with expectation:
        psd = nap.compute_power_spectral_density(sig, fs, ep, full_range, norm)


############################################################
# Test for mean_power_spectral_density
############################################################


def get_signal_and_output(f=2, fs=1000, duration=100, interval_size=10):
    t = np.arange(0, duration, 1 / fs)
    d = np.cos(2 * np.pi * f * t)
    sig = nap.Tsd(t=t, d=d, time_support=nap.IntervalSet(0, 100))
    tmp = d.reshape((int(duration / interval_size), int(fs * interval_size))).T
    # tmp = tmp[0:-1]
    tmp = tmp*signal.windows.hamming(tmp.shape[0])[:,np.newaxis]
    out = np.sum(np.fft.fft(tmp, axis=0), 1)
    freq = np.fft.fftfreq(out.shape[0], 1 / fs)
    order = np.argsort(freq)
    out = out[order]
    freq = freq[order]
    return (sig, out, freq)


def test_compute_mean_power_spectral_density():
    sig, out, freq = get_signal_and_output()
    psd = nap.compute_mean_power_spectral_density(sig, 10)
    assert isinstance(psd, pd.DataFrame)
    assert psd.shape[0] > 0  # Check that the psd DataFrame is not empty
    np.testing.assert_array_almost_equal(psd.values.flatten(), out[freq >= 0])
    np.testing.assert_array_almost_equal(psd.index.values, freq[freq >= 0])

    # Full range
    psd = nap.compute_mean_power_spectral_density(sig, 10, full_range=True)
    assert isinstance(psd, pd.DataFrame)
    assert psd.shape[0] > 0  # Check that the psd DataFrame is not empty
    np.testing.assert_array_almost_equal(psd.values.flatten(), out)
    np.testing.assert_array_almost_equal(psd.index.values, freq)

    # Norm
    psd = nap.compute_mean_power_spectral_density(sig, 10, norm=True)
    assert isinstance(psd, pd.DataFrame)
    assert psd.shape[0] > 0  # Check that the psd DataFrame is not empty
    np.testing.assert_array_almost_equal(psd.values.flatten(), out[freq >= 0]/(10000.0*10.0))
    np.testing.assert_array_almost_equal(psd.index.values, freq[freq >= 0])


    # TsdFrame
    sig2 = nap.TsdFrame(
        t=sig.t, d=np.repeat(sig.values[:, None], 2, 1), time_support=sig.time_support
    )
    psd = nap.compute_mean_power_spectral_density(sig2, 10, full_range=True)
    assert isinstance(psd, pd.DataFrame)
    assert psd.shape[0] > 0  # Check that the psd DataFrame is not empty
    np.testing.assert_array_almost_equal(psd.values, np.repeat(out[:, None], 2, 1))
    np.testing.assert_array_almost_equal(psd.index.values, freq)

    # TsdFrame
    sig2 = nap.TsdFrame(
        t=sig.t, d=np.repeat(sig.values[:, None], 2, 1), time_support=sig.time_support
    )
    psd = nap.compute_mean_power_spectral_density(sig2, 10, full_range=True, fs=1000)
    assert isinstance(psd, pd.DataFrame)
    assert psd.shape[0] > 0  # Check that the psd DataFrame is not empty
    np.testing.assert_array_almost_equal(psd.values, np.repeat(out[:, None], 2, 1))
    np.testing.assert_array_almost_equal(psd.index.values, freq)


@pytest.mark.parametrize(
    "sig, out, freq, interval_size, fs, ep, full_range, norm, time_units, expectation",
    [
        (*get_signal_and_output(), 10, None, None, False, False, "s", does_not_raise()),
        (
            "a", *get_signal_and_output()[1:],
            10,
            None,
            None,
            False,
            False,
            "s",
            pytest.raises(TypeError, match="sig must be either a Tsd or a TsdFrame object."),
        ),
        (
            *get_signal_and_output(),
            10,
            "a",
            None,
            False,
            False,
            "s",
            pytest.raises(TypeError, match="fs must be of type float or int"),
        ),
        (
            *get_signal_and_output(),
            10,
            None,
            "a",
            False,
            False,
            "s",
            pytest.raises(
                TypeError,
                match="ep param must be a pynapple IntervalSet object, or None",
            ),
        ),
        (
            *get_signal_and_output(),
            10,
            None,
            None,
            "a",
            False,
            "s",
            pytest.raises(TypeError, match="full_range must be of type bool or None"),
        ),
        (
            *get_signal_and_output(),
            10,
            None,   # FS
            None,   # Ep
            "a",    # full_range
            False,  # Norm
            "s",    # Time units
            pytest.raises(TypeError, match="full_range must be of type bool or None"),
        ),
        (
            *get_signal_and_output(),
            10,
            None,   # FS
            None,   # Ep
            False,  # full_range
            "a",    # Norm
            "s",    # Time units
            pytest.raises(TypeError, match="norm must be of type bool"),
        ),        
        (*get_signal_and_output(), 10 * 1e3, None, None, False, False, "ms", does_not_raise()),
        (*get_signal_and_output(), 10 * 1e6, None, None, False, False, "us", does_not_raise()),
        (
            *get_signal_and_output(),
            200,
            None,
            None,
            False,
            False,
            "s",
            pytest.raises(
                RuntimeError,
                match="Splitting epochs with interval_size=200 generated an empty IntervalSet. Try decreasing interval_size",
            ),
        ),
        (
            *get_signal_and_output(),
            10,
            None,
            nap.IntervalSet([0, 200], [100, 300]),
            False,
            False,
            "s",
            pytest.raises(
                RuntimeError,
                match="One interval doesn't have any signal associated. Check the parameter ep or the time support if no epoch is passed.",
            ),
        ),
    ],
)
def test_compute_mean_power_spectral_density_raise_errors(
    sig, out, freq, interval_size, fs, ep, full_range, norm, time_units, expectation
):
    with expectation:
        psd = nap.compute_mean_power_spectral_density(
            sig, interval_size, fs, ep, full_range, norm, time_units
        )
