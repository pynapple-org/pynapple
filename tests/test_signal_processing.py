"""Tests of `signal_processing` for pynapple"""

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

import pynapple as nap


def test_generate_morlet_filterbank():
    fs = 1000
    freqs = np.linspace(10, 100, 10)
    fb = nap.generate_morlet_filterbank(
        freqs, fs, gaussian_width=1.5, window_length=1.0, precision=16
    )
    power = np.abs(nap.compute_fft(fb))
    for i, f in enumerate(freqs):
        # Check that peak freq matched expectation
        assert power.iloc[:, i].argmax() == np.abs(power.index - f).argmin()

    fs = 10000
    freqs = np.linspace(100, 1000, 10)
    fb = nap.generate_morlet_filterbank(
        freqs, fs, gaussian_width=1.5, window_length=1.0, precision=16
    )
    power = np.abs(nap.compute_fft(fb))
    for i, f in enumerate(freqs):
        # Check that peak freq matched expectation
        assert power.iloc[:, i].argmax() == np.abs(power.index - f).argmin()

    fs = 1000
    freqs = np.linspace(10, 100, 10)
    fb = nap.generate_morlet_filterbank(
        freqs, fs, gaussian_width=3.5, window_length=1.0, precision=16
    )
    power = np.abs(nap.compute_fft(fb))
    for i, f in enumerate(freqs):
        # Check that peak freq matched expectation
        assert power.iloc[:, i].argmax() == np.abs(power.index - f).argmin()

    fs = 10000
    freqs = np.linspace(100, 1000, 10)
    fb = nap.generate_morlet_filterbank(
        freqs, fs, gaussian_width=3.5, window_length=1.0, precision=16
    )
    power = np.abs(nap.compute_fft(fb))
    for i, f in enumerate(freqs):
        # Check that peak freq matched expectation
        assert power.iloc[:, i].argmax() == np.abs(power.index - f).argmin()

    fs = 1000
    freqs = np.linspace(10, 100, 10)
    fb = nap.generate_morlet_filterbank(
        freqs, fs, gaussian_width=1.5, window_length=3.0, precision=16
    )
    power = np.abs(nap.compute_fft(fb))
    for i, f in enumerate(freqs):
        # Check that peak freq matched expectation
        assert power.iloc[:, i].argmax() == np.abs(power.index - f).argmin()

    fs = 10000
    freqs = np.linspace(100, 1000, 10)
    fb = nap.generate_morlet_filterbank(
        freqs, fs, gaussian_width=1.5, window_length=3.0, precision=16
    )
    power = np.abs(nap.compute_fft(fb))
    for i, f in enumerate(freqs):
        # Check that peak freq matched expectation
        assert power.iloc[:, i].argmax() == np.abs(power.index - f).argmin()

    gaussian_atol = 1e-4
    # Checking that the power spectra of the wavelets resemble correct Gaussians
    fs = 2000
    freqs = np.linspace(100, 1000, 10)
    fb = nap.generate_morlet_filterbank(
        freqs, fs, gaussian_width=1.0, window_length=1.0, precision=24
    )
    power = np.abs(nap.compute_fft(fb))
    for i, f in enumerate(freqs):
        gaussian_width = 1.0
        window_length = 1.0
        fz = power.index
        factor = np.pi**0.25 * gaussian_width**0.25
        morlet_ft = factor * np.exp(
            -np.pi**2 * gaussian_width * (window_length * (fz - f) / f) ** 2
        )
        assert np.isclose(
            power.iloc[:, i] / np.max(power.iloc[:, i]),
            morlet_ft / np.max(morlet_ft),
            atol=gaussian_atol,
        ).all()

    fs = 100
    freqs = np.linspace(1, 10, 10)
    fb = nap.generate_morlet_filterbank(
        freqs, fs, gaussian_width=1.0, window_length=1.0, precision=24
    )
    power = np.abs(nap.compute_fft(fb))
    for i, f in enumerate(freqs):
        gaussian_width = 1.0
        window_length = 1.0
        fz = power.index
        factor = np.pi**0.25 * gaussian_width**0.25
        morlet_ft = factor * np.exp(
            -np.pi**2 * gaussian_width * (window_length * (fz - f) / f) ** 2
        )
        assert np.isclose(
            power.iloc[:, i] / np.max(power.iloc[:, i]),
            morlet_ft / np.max(morlet_ft),
            atol=gaussian_atol,
        ).all()

    fs = 100
    freqs = np.linspace(1, 10, 10)
    fb = nap.generate_morlet_filterbank(
        freqs, fs, gaussian_width=4.0, window_length=1.0, precision=24
    )
    power = np.abs(nap.compute_fft(fb))
    for i, f in enumerate(freqs):
        gaussian_width = 4.0
        window_length = 1.0
        fz = power.index
        factor = np.pi**0.25 * gaussian_width**0.25
        morlet_ft = factor * np.exp(
            -np.pi**2 * gaussian_width * (window_length * (fz - f) / f) ** 2
        )
        assert np.isclose(
            power.iloc[:, i] / np.max(power.iloc[:, i]),
            morlet_ft / np.max(morlet_ft),
            atol=gaussian_atol,
        ).all()

    fs = 100
    freqs = np.linspace(1, 10, 10)
    fb = nap.generate_morlet_filterbank(
        freqs, fs, gaussian_width=4.0, window_length=3.0, precision=24
    )
    power = np.abs(nap.compute_fft(fb))
    for i, f in enumerate(freqs):
        gaussian_width = 4.0
        window_length = 3.0
        fz = power.index
        factor = np.pi**0.25 * gaussian_width**0.25
        morlet_ft = factor * np.exp(
            -np.pi**2 * gaussian_width * (window_length * (fz - f) / f) ** 2
        )
        assert np.isclose(
            power.iloc[:, i] / np.max(power.iloc[:, i]),
            morlet_ft / np.max(morlet_ft),
            atol=gaussian_atol,
        ).all()

    fs = 1000
    freqs = np.linspace(1, 10, 10)
    fb = nap.generate_morlet_filterbank(
        freqs, fs, gaussian_width=3.5, window_length=1.25, precision=24
    )
    power = np.abs(nap.compute_fft(fb))
    for i, f in enumerate(freqs):
        gaussian_width = 3.5
        window_length = 1.25
        fz = power.index
        factor = np.pi**0.25 * gaussian_width**0.25
        morlet_ft = factor * np.exp(
            -np.pi**2 * gaussian_width * (window_length * (fz - f) / f) ** 2
        )
        assert np.isclose(
            power.iloc[:, i] / np.max(power.iloc[:, i]),
            morlet_ft / np.max(morlet_ft),
            atol=gaussian_atol,
        ).all()


@pytest.mark.parametrize(
    "freqs, fs, gaussian_width, window_length, precision, expectation",
    [
        (
            np.linspace(0, 100, 11),
            1000,
            1.5,
            1.0,
            16,
            pytest.raises(
                ValueError, match="All frequencies in freqs must be strictly positive"
            ),
        ),
        (
            "a",
            1000,
            1.5,
            1.0,
            16,
            pytest.raises(TypeError, match="`freqs` must be a ndarray"),
        ),
        (
            np.array([]),
            1000,
            1.5,
            1.0,
            16,
            pytest.raises(ValueError, match="Given list of freqs cannot be empty."),
        ),
        (
            np.linspace(1, 10, 1),
            "a",
            1.5,
            1.0,
            16,
            pytest.raises(TypeError, match="`fs` must be of type float or int ndarray"),
        ),
        (
            np.linspace(1, 10, 1),
            1000,
            -1.5,
            1.0,
            16,
            pytest.raises(
                ValueError, match="gaussian_width must be a positive number."
            ),
        ),
        (
            np.linspace(1, 10, 1),
            1000,
            "a",
            1.0,
            16,
            pytest.raises(
                TypeError, match="gaussian_width must be a float or int instance."
            ),
        ),
        (
            np.linspace(1, 10, 1),
            1000,
            1.5,
            -1.0,
            16,
            pytest.raises(ValueError, match="window_length must be a positive number."),
        ),
        (
            np.linspace(1, 10, 1),
            1000,
            1.5,
            "a",
            16,
            pytest.raises(
                TypeError, match="window_length must be a float or int instance."
            ),
        ),
        (
            np.linspace(1, 10, 1),
            1000,
            1.5,
            1.0,
            -16,
            pytest.raises(ValueError, match="precision must be a positive number."),
        ),
        (
            np.linspace(1, 10, 1),
            1000,
            1.5,
            1.0,
            "a",
            pytest.raises(
                TypeError, match="precision must be a float or int instance."
            ),
        ),
    ],
)
def test_generate_morlet_filterbank_raise_errors(
    freqs, fs, gaussian_width, window_length, precision, expectation
):
    with expectation:
        _ = nap.generate_morlet_filterbank(
            freqs, fs, gaussian_width, window_length, precision
        )


############################################################
# Test for compute_wavelet_transform
############################################################


def get_1d_signal(fs=1000, fc=50):
    t = np.arange(0, 2, 1 / fs)
    d = np.sin(t * fc * np.pi * 2) * np.interp(t, [0, 1, 2], [0, 1, 0])
    return nap.Tsd(t, d, time_support=nap.IntervalSet(0, 2))


def get_2d_signal(fs=1000, fc=50):
    t = np.arange(0, 2, 1 / fs)
    d = np.sin(t * fc * np.pi * 2) * np.interp(t, [0, 1, 2], [0, 1, 0])
    return nap.TsdFrame(
        t, np.repeat(d[:, np.newaxis], 2, axis=1), time_support=nap.IntervalSet(0, 2)
    )


def get_3d_signal(fs=1000, fc=50):
    t = np.arange(0, 2, 1 / fs)
    d = np.sin(t * fc * np.pi * 2) * np.interp(t, [0, 1, 2], [0, 1, 0])
    d = d[:, np.newaxis, np.newaxis]
    d = np.repeat(np.repeat(d, 2, axis=1), 3, axis=2)
    return nap.TsdTensor(t, d, time_support=nap.IntervalSet(0, 2))


def get_output_1d(sig, wavelets):
    T = sig.shape[0]
    M, N = wavelets.shape
    out = []
    for n in range(N):
        out.append(np.convolve(sig, wavelets[:, n], mode="full"))
    out = np.array(out).T
    cut = ((M - 1) // 2, T + M - 1 - ((M - 1) // 2) - (1 - M % 2))
    return out[cut[0] : cut[1]]


def get_output_2d(sig, wavelets):
    T, K = sig.shape
    M, N = wavelets.shape
    cut = ((M - 1) // 2, T + M - 1 - ((M - 1) // 2) - (1 - M % 2))
    out = np.zeros((T, K, N), dtype=np.complex128)
    for n in range(N):  # wavelet
        for k in range(K):
            out[:, k, n] = np.convolve(sig[:, k], wavelets[:, n], mode="full")[
                cut[0] : cut[1]
            ]

    return out


def get_output_3d(sig, wavelets):
    T, K, L = sig.shape
    M, N = wavelets.shape
    cut = ((M - 1) // 2, T + M - 1 - ((M - 1) // 2) - (1 - M % 2))
    out = np.zeros((T, K, L, N), dtype=np.complex128)
    for n in range(N):  # wavelet
        for k in range(K):
            for l in range(L):
                out[:, k, l, n] = np.convolve(
                    sig[:, k, l], wavelets[:, n], mode="full"
                )[cut[0] : cut[1]]

    return out


@pytest.mark.parametrize(
    "func, freqs, fs, gaussian_width, window_length, precision, norm, fc, maxt",
    [
        (get_1d_signal, np.linspace(10, 100, 10), 1000, 1.5, 1.0, 16, None, 50, 1000),
        (get_1d_signal, np.linspace(10, 100, 10), None, 1.5, 1.0, 16, None, 50, 1000),
        (get_1d_signal, np.linspace(10, 100, 10), 1000, 3.0, 1.0, 16, None, 50, 1000),
        (get_1d_signal, np.linspace(10, 100, 10), 1000, 1.5, 2.0, 16, None, 50, 1000),
        (get_1d_signal, np.linspace(10, 100, 10), 1000, 1.5, 1.0, 20, None, 50, 1000),
        (get_1d_signal, np.linspace(10, 100, 10), 1000, 1.5, 1.0, 16, "l1", 50, 1000),
        (get_1d_signal, np.linspace(10, 100, 10), 1000, 1.5, 1.0, 16, "l2", 50, 1000),
        (get_1d_signal, np.linspace(10, 100, 10), 1000, 1.5, 1.0, 16, None, 20, 1000),
        (get_2d_signal, np.linspace(10, 100, 10), 1000, 1.5, 1.0, 16, None, 20, 1000),
        (get_3d_signal, np.linspace(10, 100, 10), 1000, 1.5, 1.0, 16, None, 20, 1000),
    ],
)
def test_compute_wavelet_transform(
    func, freqs, fs, gaussian_width, window_length, precision, norm, fc, maxt
):
    sig = func(1000, fc)
    wavelets = nap.generate_morlet_filterbank(
        freqs, 1000, gaussian_width, window_length, precision
    )
    if sig.ndim == 1:
        output = get_output_1d(sig.d, wavelets.values)
    if sig.ndim == 2:
        output = get_output_2d(sig.d, wavelets.values)
    if sig.ndim == 3:
        output = get_output_3d(sig.d, wavelets.values)

    if norm == "l1":
        output = output / (1000 / freqs)
    if norm == "l2":
        output = output / (1000 / np.sqrt(freqs))

    mwt = nap.compute_wavelet_transform(
        sig,
        freqs,
        fs=fs,
        gaussian_width=gaussian_width,
        window_length=window_length,
        precision=precision,
        norm=norm,
    )

    np.testing.assert_array_almost_equal(output, mwt.values)
    assert freqs[np.argmax(np.sum(np.abs(mwt), axis=0))] == fc
    assert (
        np.unravel_index(np.abs(mwt.values).argmax(), np.abs(mwt.values).shape)[0]
        == maxt
    )
    np.testing.assert_array_almost_equal(
        mwt.time_support.values, sig.time_support.values
    )
    if isinstance(mwt, nap.TsdFrame):
        # test column names if TsdFrame
        np.testing.assert_array_almost_equal(mwt.columns, freqs)


@pytest.mark.parametrize(
    "sig, freqs, fs, gaussian_width, window_length, precision, norm, expectation",
    [
        (
            get_1d_signal(),
            np.linspace(1, 10, 2),
            1000,
            1.5,
            1,
            16,
            None,
            does_not_raise(),
        ),
        (
            "a",
            np.linspace(1, 10, 2),
            1000,
            1.5,
            1,
            16,
            None,
            pytest.raises(
                TypeError,
                match=re.escape(
                    "`sig` must be instance of Tsd, TsdFrame, or TsdTensor"
                ),
            ),
        ),
        (
            get_1d_signal(),
            np.linspace(1, 10, 2),
            "a",
            1.5,
            1,
            16,
            None,
            pytest.raises(
                TypeError,
                match=re.escape("`fs` must be of type float or int or None"),
            ),
        ),
        (
            get_1d_signal(),
            np.linspace(1, 10, 2),
            1000,
            -1.5,
            1,
            16,
            None,
            pytest.raises(
                ValueError,
                match=re.escape("gaussian_width must be a positive number."),
            ),
        ),
        (
            get_1d_signal(),
            np.linspace(1, 10, 2),
            1000,
            "a",
            1,
            16,
            None,
            pytest.raises(
                TypeError,
                match=re.escape("gaussian_width must be a float or int instance."),
            ),
        ),
        (
            get_1d_signal(),
            np.linspace(1, 10, 2),
            1000,
            1.5,
            -1,
            16,
            None,
            pytest.raises(
                ValueError,
                match=re.escape("window_length must be a positive number."),
            ),
        ),
        (
            get_1d_signal(),
            np.linspace(1, 10, 2),
            1000,
            1.5,
            "a",
            16,
            None,
            pytest.raises(
                TypeError,
                match=re.escape("window_length must be a float or int instance."),
            ),
        ),
        (
            get_1d_signal(),
            np.linspace(1, 10, 2),
            1000,
            1.5,
            1,
            16,
            "a",
            pytest.raises(
                ValueError,
                match=re.escape("norm parameter must be 'l1', 'l2', or None."),
            ),
        ),
        (
            get_1d_signal(),
            "a",
            1000,
            1.5,
            1,
            16,
            None,
            pytest.raises(
                TypeError,
                match=re.escape("`freqs` must be a ndarray"),
            ),
        ),
        (
            get_1d_signal(),
            np.array([]),
            1000,
            1.5,
            1,
            16,
            None,
            pytest.raises(
                ValueError,
                match=re.escape("Given list of freqs cannot be empty."),
            ),
        ),
        (
            get_1d_signal(),
            np.array([-1]),
            1000,
            1.5,
            1,
            16,
            None,
            pytest.raises(
                ValueError,
                match=re.escape("All frequencies in freqs must be strictly positive"),
            ),
        ),
        (
            get_1d_signal(),
            np.linspace(1, 10, 2),
            1000,
            1.5,
            1,
            16,
            1,
            pytest.raises(
                ValueError,
                match=re.escape("norm parameter must be 'l1', 'l2', or None."),
            ),
        ),
    ],
)
def test_compute_wavelet_transform_raise_errors(
    sig, freqs, fs, gaussian_width, window_length, precision, norm, expectation
):
    with expectation:
        _ = nap.compute_wavelet_transform(
            sig, freqs, fs, gaussian_width, window_length, precision, norm
        )
