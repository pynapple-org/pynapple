"""Tests of `signal_processing` for pynapple"""

import numpy as np
import pandas as pd
import pytest

import pynapple as nap


def test_compute_spectogram():
    t = np.linspace(0, 1, 1024)
    sig = nap.Tsd(d=np.random.random(1024), t=t)
    r = nap.compute_spectogram(sig)
    assert isinstance(r, pd.DataFrame)
    assert r.shape[0] == 1024

    sig = nap.TsdFrame(d=np.random.random((1024, 4)), t=t)
    r = nap.compute_spectogram(sig)
    assert isinstance(r, pd.DataFrame)
    assert r.shape == (1024, 4)

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

    sig = nap.Tsd(d=np.random.random(1024), t=t)
    freqs = np.linspace(1, 600, 10)
    mwt = nap.compute_wavelet_transform(
        sig, fs=None, freqs=freqs, n_cycles=tuple(np.repeat((1.5, 2.5), 5))
    )
    assert mwt.shape == (1024, 10)

    sig = nap.TsdFrame(d=np.random.random((1024, 4)), t=t)
    freqs = np.linspace(1, 600, 10)
    mwt = nap.compute_wavelet_transform(sig, fs=None, freqs=freqs)
    assert mwt.shape == (1024, 10, 4)

    with pytest.raises(ValueError) as e_info:
        nap.compute_wavelet_transform(sig, fs=None, freqs=freqs, n_cycles=-1.5)
    assert str(e_info.value) == "Number of cycles must be a positive number."

    with pytest.raises(ValueError) as e_info:
        nap.compute_wavelet_transform(
            sig, fs=None, freqs=freqs, n_cycles=tuple(np.repeat((1.5, -2.5), 5))
        )
    assert str(e_info.value) == "Each number of cycles must be a positive number."

    with pytest.raises(ValueError) as e_info:
        nap.compute_wavelet_transform(
            sig, fs=None, freqs=freqs, n_cycles=tuple(np.repeat((1.5, 2.5), 2))
        )
    assert (
        str(e_info.value)
        == "The length of number of cycles does not match other inputs."
    )
