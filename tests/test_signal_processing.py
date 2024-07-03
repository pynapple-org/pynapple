"""Tests of `signal_processing` for pynapple"""

import numpy as np
import pytest

import pynapple as nap


def test_compute_spectogram():
    t = np.linspace(0, 1, 1024)
    sig = nap.Tsd(d=np.random.random(1024), t=t)
    r = nap.compute_spectogram(sig)
    assert len(r[1]) == 1024
    assert len(r[0]) == 1024
    assert r[0].dtype == np.complex128
    assert r[1].dtype == np.float64


def test_ccompute_welch_spectogram():
    t = np.linspace(0, 1, 1024)
    sig = nap.Tsd(d=np.random.random(1024), t=t)
    r = nap.compute_welch_spectogram(sig)
    assert r[0].dtype == np.float64
    assert r[1].dtype == np.float64


def test_compute_wavelet_transform():
    t = np.linspace(0, 1, 1024)
    sig = nap.Tsd(d=np.random.random(1024), t=t)
    freqs = np.linspace(1, 600, 10)
    mwt = nap.compute_wavelet_transform(sig, fs=None, freqs=freqs)
    assert mwt.shape == (1024, 10)

    sig = nap.TsdFrame(d=np.random.random((1024, 4)), t=t)
    freqs = np.linspace(1, 600, 10)
    mwt = nap.compute_wavelet_transform(sig, fs=None, freqs=freqs)
    assert mwt.shape == (1024, 10, 4)
