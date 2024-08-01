"""Tests of `signal_processing` for pynapple"""

import numpy as np
import pandas as pd
import pytest

import pynapple as nap



def test_compute_wavelet_transform():
    t = np.linspace(0, 1, 1001)
    sig = nap.Tsd(
        d=np.sin(t * 50 * np.pi * 2)
        * np.interp(np.linspace(0, 1, 1001), [0, 0.5, 1], [0, 1, 0]),
        t=t,
    )
    freqs = np.linspace(10, 100, 10)
    mwt = nap.compute_wavelet_transform(sig, fs=None, freqs=freqs)
    mpf = freqs[np.argmax(np.sum(np.abs(mwt), axis=0))]
    assert mpf == 50
    assert (
        np.unravel_index(np.abs(mwt.values).argmax(), np.abs(mwt.values).shape)[0]
        == 500
    )

    t = np.linspace(0, 1, 1001)
    sig = nap.Tsd(
        d=np.sin(t * 10 * np.pi * 2)
        * np.interp(np.linspace(0, 1, 1001), [0, 0.5, 1], [0, 1, 0]),
        t=t,
    )
    freqs = np.linspace(10, 100, 10)
    mwt = nap.compute_wavelet_transform(sig, fs=None, freqs=freqs)
    mpf = freqs[np.argmax(np.sum(np.abs(mwt), axis=0))]
    assert mpf == 10
    assert (
        np.unravel_index(np.abs(mwt.values).argmax(), np.abs(mwt.values).shape)[0]
        == 500
    )

    t = np.linspace(0, 1, 1000)
    sig = nap.Tsd(d=np.sin(t * 50 * np.pi * 2), t=t)
    freqs = np.linspace(10, 100, 10)
    mwt = nap.compute_wavelet_transform(sig, fs=None, freqs=freqs)
    mpf = freqs[np.argmax(np.sum(np.abs(mwt), axis=0))]
    assert mpf == 50
    assert mwt.shape == (1000, 10)

    t = np.linspace(0, 1, 1000)
    sig = nap.Tsd(d=np.sin(t * 20 * np.pi * 2), t=t)
    freqs = np.linspace(10, 100, 10)
    mwt = nap.compute_wavelet_transform(sig, fs=None, freqs=freqs)
    mpf = freqs[np.argmax(np.sum(np.abs(mwt), axis=0))]
    assert mpf == 20
    assert mwt.shape == (1000, 10)

    t = np.linspace(0, 1, 1000)
    sig = nap.Tsd(d=np.sin(t * 20 * np.pi * 2), t=t)
    freqs = np.linspace(10, 100, 10)
    mwt = nap.compute_wavelet_transform(sig, fs=None, freqs=freqs, norm="l1")
    mpf = freqs[np.argmax(np.sum(np.abs(mwt), axis=0))]
    assert mpf == 20
    assert mwt.shape == (1000, 10)

    t = np.linspace(0, 1, 1000)
    sig = nap.Tsd(d=np.sin(t * 20 * np.pi * 2), t=t)
    freqs = np.linspace(10, 100, 10)
    mwt = nap.compute_wavelet_transform(sig, fs=None, freqs=freqs, norm="l2")
    mpf = freqs[np.argmax(np.sum(np.abs(mwt), axis=0))]
    assert mpf == 20
    assert mwt.shape == (1000, 10)

    t = np.linspace(0, 1, 1000)
    sig = nap.Tsd(d=np.sin(t * 20 * np.pi * 2), t=t)
    freqs = np.linspace(10, 100, 10)
    mwt = nap.compute_wavelet_transform(sig, fs=None, freqs=freqs, norm=None)
    mpf = freqs[np.argmax(np.sum(np.abs(mwt), axis=0))]
    assert mpf == 20
    assert mwt.shape == (1000, 10)

    t = np.linspace(0, 1, 1000)
    sig = nap.Tsd(d=np.sin(t * 70 * np.pi * 2), t=t)
    freqs = np.linspace(10, 100, 10)
    mwt = nap.compute_wavelet_transform(sig, fs=None, freqs=freqs)
    mpf = freqs[np.argmax(np.sum(np.abs(mwt), axis=0))]
    assert mpf == 70
    assert mwt.shape == (1000, 10)

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
        nap.compute_wavelet_transform(sig, fs=None, freqs=freqs, gaussian_width=-1.5)
    assert str(e_info.value) == "gaussian_width must be a positive number."


if __name__ == "__main__":
    test_compute_wavelet_transform()
