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


def get_sorted_fft(data, fs):
    fft = np.fft.fft(data, axis=0)
    fft_freq = np.fft.fftfreq(len(data), 1 / fs)
    order = np.argsort(fft_freq)
    if fft.ndim == 1:
        fft = fft[:, np.newaxis]
    return fft_freq[order], fft[order]


def get_periodogram(data, fs, full_range=False):
    return_onesided = not full_range
    f, p = signal.periodogram(
        data, fs, return_onesided=return_onesided, detrend=False, axis=0
    )
    if p.ndim == 1:
        p = p[:, np.newaxis]
    if full_range:
        order = np.argsort(f)
        f = f[order]
        p = p[order]
    else:
        f = f[:-1]
        p = p[:-1]
    return f, p


def test_compute_fft():

    t = np.linspace(0, 1, 1000)
    sig = nap.Tsd(d=np.random.random(1000), t=t)
    r = nap.compute_fft(sig)
    assert isinstance(r, pd.DataFrame)
    assert r.shape[0] == 500

    a, b = get_sorted_fft(sig.values, sig.rate)
    np.testing.assert_array_almost_equal(r.index.values, a[a >= 0])
    np.testing.assert_array_almost_equal(r.values, b[a >= 0])

    r = nap.compute_fft(sig, norm=True)
    np.testing.assert_array_almost_equal(r.values, b[a >= 0] / len(sig))

    sig = nap.TsdFrame(d=np.random.random((1000, 4)), t=t)
    r = nap.compute_fft(sig)
    assert isinstance(r, pd.DataFrame)
    assert r.shape == (500, 4)

    a, b = get_sorted_fft(sig.values, sig.rate)
    np.testing.assert_array_almost_equal(r.index.values, a[a >= 0])
    np.testing.assert_array_almost_equal(r.values, b[a >= 0])

    sig = nap.TsdFrame(d=np.random.random((1000, 4)), t=t)
    r = nap.compute_fft(sig, full_range=True)
    assert isinstance(r, pd.DataFrame)
    assert r.shape == (1000, 4)

    a, b = get_sorted_fft(sig.values, sig.rate)
    np.testing.assert_array_almost_equal(r.index.values, a)
    np.testing.assert_array_almost_equal(r.values, b)

    t = np.linspace(0, 1, 1000)
    sig = nap.Tsd(d=np.random.random(1000), t=t)
    r = nap.compute_fft(sig, ep=sig.time_support)
    assert isinstance(r, pd.DataFrame)
    assert r.shape[0] == 500

    t = np.linspace(0, 1, 1000)
    sig = nap.Tsd(d=np.random.random(1000), t=t)
    r = nap.compute_fft(sig, fs=1000)
    assert isinstance(r, pd.DataFrame)
    assert r.shape[0] == 500

    a, b = get_sorted_fft(np.hstack((sig.values, np.zeros(20))), sig.rate)
    r = nap.compute_fft(sig, fs=1000, n=len(sig) + 20, full_range=True)
    np.testing.assert_array_almost_equal(r.index.values, a)
    np.testing.assert_array_almost_equal(r.values, b)


def test_compute_power_spectral_density():

    t = np.linspace(0, 1, 1000)
    sig = nap.Tsd(d=np.random.random(1000), t=t)
    r = nap.compute_power_spectral_density(sig)
    assert isinstance(r, pd.DataFrame)
    assert r.shape[0] == 500

    a, b = get_periodogram(sig.values, sig.rate)
    np.testing.assert_array_almost_equal(r.index.values, a[a >= 0])
    np.testing.assert_array_almost_equal(r.values, b[a >= 0])

    sig = nap.TsdFrame(d=np.random.random((1000, 4)), t=t)
    r = nap.compute_power_spectral_density(sig)
    assert isinstance(r, pd.DataFrame)
    assert r.shape == (500, 4)

    a, b = get_periodogram(sig.values, sig.rate)
    np.testing.assert_array_almost_equal(r.index.values, a[a >= 0])
    np.testing.assert_array_almost_equal(r.values, b[a >= 0])

    sig = nap.TsdFrame(d=np.random.random((1000, 4)), t=t)
    r = nap.compute_power_spectral_density(sig, full_range=True)
    assert isinstance(r, pd.DataFrame)
    assert r.shape == (1000, 4)

    a, b = get_periodogram(sig.values, sig.rate, full_range=True)
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
    "sig, kwargs, expectation",
    [
        (
            nap.Tsd(
                d=np.random.random(1000),
                t=np.linspace(0, 1, 1000),
                time_support=nap.IntervalSet(start=[0.1, 0.6], end=[0.2, 0.81]),
            ),
            {},
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Given epoch (or signal time_support) must have length 1"
                ),
            ),
        ),
        (
            nap.Tsd(d=np.random.random(1000), t=np.linspace(0, 1, 1000)),
            {"ep": "not_ep"},
            pytest.raises(
                TypeError,
                match=re.escape(
                    "Invalid type. Parameter ep must be of type <class 'pynapple.core.interval_set.IntervalSet'>."
                ),
            ),
        ),
        (
            nap.Tsd(d=np.random.random(1000), t=np.linspace(0, 1, 1000)),
            {"fs": "a"},
            pytest.raises(
                TypeError,
                match="Invalid type. Parameter fs must be of type <class 'numbers.Number'>.",
            ),
        ),
        (
            "not_a_tsd",
            {},
            pytest.raises(
                TypeError,
                match=re.escape(
                    "Invalid type. Parameter sig must be of type (<class 'pynapple.core.time_series.Tsd'>, <class 'pynapple.core.time_series.TsdFrame'>)."
                ),
            ),
        ),
        (
            nap.Tsd(d=np.random.random(1000), t=np.linspace(0, 1, 1000)),
            {"full_range": "a"},
            pytest.raises(
                TypeError,
                match=re.escape(
                    "Invalid type. Parameter full_range must be of type <class 'bool'>."
                ),
            ),
        ),
        (
            nap.Tsd(d=np.random.random(1000), t=np.linspace(0, 1, 1000)),
            {"norm": "a"},
            pytest.raises(
                TypeError,
                match=re.escape(
                    "Invalid type. Parameter norm must be of type <class 'bool'>."
                ),
            ),
        ),
    ],
)
def test_compute_fft_raise_errors(sig, kwargs, expectation):
    with expectation:
        nap.compute_fft(sig, **kwargs)
    if "norm" not in kwargs:
        with expectation:
            nap.compute_power_spectral_density(sig, **kwargs)


############################################################
# Test for mean_power_spectral_density
############################################################


def get_signal_and_output(f=2, fs=1000, duration=100, interval_size=10, overlap=0.25):
    t = np.arange(0, duration, 1 / fs)
    d = np.cos(2 * np.pi * f * t)
    sig = nap.Tsd(t=t, d=d, time_support=nap.IntervalSet(0, 100))
    tmp = []
    slice = (0, int(fs * interval_size) + 1)
    while slice[1] < len(d):
        tmp.append(d[slice[0] : slice[1]])
        new_slice = (
            slice[1] - int(fs * interval_size * overlap) - 1,
            slice[1] + int(fs * interval_size * (1 - overlap)),
        )
        slice = new_slice

    tmp = np.transpose(np.array(tmp))
    print(tmp.shape)
    # tmp = d.reshape((int(duration / interval_size), int(fs * interval_size))).T
    # tmp = tmp[0:-1]
    tmp = tmp * signal.windows.hamming(tmp.shape[0])[:, np.newaxis]
    fft = np.fft.fft(tmp, axis=0)
    psd = (1 / (fs * tmp.shape[0])) * (np.power(np.abs(fft), 2.0))
    out = np.mean(psd, 1)
    freq = np.fft.fftfreq(tmp.shape[0], 1 / fs)
    order = np.argsort(freq)
    out = out[order]
    freq = freq[order]
    return (sig, out, freq)


def test_compute_mean_psd():
    sig, out, freq = get_signal_and_output()
    out2 = np.copy(out[freq >= 0])
    out2[out2 > 0] *= 2.0

    psd = nap.compute_mean_power_spectral_density(sig, 10)

    assert isinstance(psd, pd.DataFrame)
    assert psd.shape[0] > 0  # Check that the psd DataFrame is not empty
    np.testing.assert_array_almost_equal(psd.values.flatten(), out2)
    np.testing.assert_array_almost_equal(psd.index.values, freq[freq >= 0])

    # Full range
    psd = nap.compute_mean_power_spectral_density(sig, 10, full_range=True)
    assert isinstance(psd, pd.DataFrame)
    assert psd.shape[0] > 0  # Check that the psd DataFrame is not empty
    np.testing.assert_array_almost_equal(psd.values.flatten(), out)
    np.testing.assert_array_almost_equal(psd.index.values, freq)

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
    "sig, interval_size, kwargs, expectation",
    [
        (get_signal_and_output()[0], 10, {}, does_not_raise()),
        (
            "a",
            10,
            {},
            pytest.raises(
                TypeError,
                match=re.escape(
                    "Invalid type. Parameter sig must be of type (<class 'pynapple.core.time_series.Tsd'>, <class 'pynapple.core.time_series.TsdFrame'>)."
                ),
            ),
        ),
        (
            get_signal_and_output()[0],
            10,
            {"fs": "a"},
            pytest.raises(
                TypeError,
                match=re.escape(
                    "Invalid type. Parameter fs must be of type <class 'numbers.Number'>."
                ),
            ),
        ),
        (
            get_signal_and_output()[0],
            10,
            {"ep": "a"},
            pytest.raises(
                TypeError,
                match=re.escape(
                    "Invalid type. Parameter ep must be of type <class 'pynapple.core.interval_set.IntervalSet'>."
                ),
            ),
        ),
        (
            get_signal_and_output()[0],
            10,
            {"ep": "a"},
            pytest.raises(
                TypeError,
                match=re.escape(
                    "Invalid type. Parameter ep must be of type <class 'pynapple.core.interval_set.IntervalSet'>."
                ),
            ),
        ),
        (
            get_signal_and_output()[0],
            10,
            {"overlap": "a"},
            pytest.raises(
                TypeError,
                match=re.escape(
                    "Invalid type. Parameter overlap must be of type <class 'float'>."
                ),
            ),
        ),
        (
            get_signal_and_output()[0],
            200,
            {},
            pytest.raises(
                RuntimeError,
                match=re.escape(
                    "Splitting epochs with interval_size=200 generated an empty IntervalSet. Try decreasing interval_size"
                ),
            ),
        ),
        (
            get_signal_and_output()[0],
            10,
            {"ep": nap.IntervalSet([0, 200], [100, 300])},
            pytest.raises(
                RuntimeError,
                match=re.escape(
                    "One interval doesn't have any signal associated. Check the parameter ep or the time support if no epoch is passed."
                ),
            ),
        ),
        (
            get_signal_and_output()[0],
            10,
            {"overlap": -0.1},
            pytest.raises(
                ValueError,
                match=re.escape("Overlap should be in intervals [0.0, 1.0)."),
            ),
        ),
        (
            get_signal_and_output()[0],
            10,
            {"overlap": 1.1},
            pytest.raises(
                ValueError,
                match=re.escape("Overlap should be in intervals [0.0, 1.0)."),
            ),
        ),
    ],
)
def test_compute_mean_power_spectral_density_raise_errors(
    sig, interval_size, kwargs, expectation
):
    with expectation:
        nap.compute_mean_power_spectral_density(sig, interval_size, **kwargs)
