"""Tests of `signal_processing` for pynapple"""

import numpy as np
import pandas as pd
import pytest

import pynapple as nap


def test_compute_spectogram():
    with pytest.raises(ValueError) as e_info:
        t = np.linspace(0, 1, 1000)
        sig = nap.Tsd(
            d=np.random.random(1000),
            t=t,
            time_support=nap.IntervalSet(start=[0.1, 0.6], end=[0.2, 0.81]),
        )
        r = nap.compute_spectogram(sig)
    assert (
        str(e_info.value) == "Given epoch (or signal time_support) must have length 1"
    )

    t = np.linspace(0, 1, 1000)
    sig = nap.Tsd(d=np.random.random(1000), t=t)
    r = nap.compute_spectogram(sig)
    assert isinstance(r, pd.DataFrame)
    assert r.shape[0] == 500

    sig = nap.TsdFrame(d=np.random.random((1000, 4)), t=t)
    r = nap.compute_spectogram(sig)
    assert isinstance(r, pd.DataFrame)
    assert r.shape == (500, 4)

    sig = nap.TsdFrame(d=np.random.random((1000, 4)), t=t)
    r = nap.compute_spectogram(sig, full_range=True)
    assert isinstance(r, pd.DataFrame)
    assert r.shape == (1000, 4)

    with pytest.raises(TypeError) as e_info:
        nap.compute_spectogram("a_string")
    assert (
        str(e_info.value)
        == "Currently compute_spectogram is only implemented for Tsd or TsdFrame"
    )


def test_compute_welch_spectogram():
    t = np.linspace(0, 1, 1024)
    sig = nap.Tsd(d=np.random.random(1024), t=t)
    r = nap.compute_welch_spectogram(sig)
    assert isinstance(r, pd.DataFrame)

    sig = nap.TsdFrame(d=np.random.random((1024, 4)), t=t)
    r = nap.compute_welch_spectogram(sig)
    assert isinstance(r, pd.DataFrame)
    assert r.shape[1] == 4

    with pytest.raises(TypeError) as e_info:
        nap.compute_welch_spectogram("a_string")
    assert (
        str(e_info.value)
        == "Currently compute_welch_spectogram is only implemented for Tsd or TsdFrame"
    )


def test_compute_wavelet_transform():

    t = np.linspace(0, 1, 1024)
    sig = nap.Tsd(d=np.random.random(1024), t=t)
    freqs = np.linspace(1, 600, 10)
    mwt = nap.compute_wavelet_transform(sig, fs=None, freqs=freqs)
    assert mwt.shape == (1024, 10)

    t = np.linspace(0, 1, 1024)
    sig = nap.Tsd(d=np.random.random(1024), t=t)
    freqs = (1, 51, 10)
    mwt = nap.compute_wavelet_transform(sig, fs=None, freqs=freqs)
    assert mwt.shape == (1024, 6)

    sig = nap.TsdFrame(d=np.random.random((1024, 4)), t=t)
    freqs = np.linspace(1, 600, 10)
    mwt = nap.compute_wavelet_transform(sig, fs=None, freqs=freqs)
    assert mwt.shape == (1024, 10, 4)

    t = np.linspace(0, 1, 1024)  # can remove this when we move it
    sig = nap.TsdTensor(d=np.random.random((1024, 4, 2)), t=t)
    freqs = np.linspace(1, 600, 10)
    mwt = nap.compute_wavelet_transform(sig, fs=None, freqs=freqs)
    assert mwt.shape == (1024, 10, 4, 2)

    with pytest.raises(ValueError) as e_info:
        nap.compute_wavelet_transform(sig, fs=None, freqs=freqs, n_cycles=-1.5)
    assert str(e_info.value) == "Number of cycles must be a positive number."


if __name__ == "__main__":
    test_compute_wavelet_transform()
