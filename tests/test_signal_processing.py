"""Tests of `signal_processing` for pynapple"""

import numpy as np
import pytest

import pynapple as nap


def test_generate_morlet_filterbank():
    fs = 1000
    freqs = np.linspace(10, 100, 10)
    fb = nap.generate_morlet_filterbank(
        freqs, fs, gaussian_width=1.5, window_length=1.0, precision=16
    )
    power = np.abs(nap.compute_power_spectral_density(fb))
    for i, f in enumerate(freqs):
        assert power.iloc[:, i].argmax() == np.abs(power.index - f).argmin()

    fs = 10000
    freqs = np.linspace(100, 1000, 10)
    fb = nap.generate_morlet_filterbank(
        freqs, fs, gaussian_width=1.5, window_length=1.0, precision=16
    )
    power = np.abs(nap.compute_power_spectral_density(fb))
    for i, f in enumerate(freqs):
        assert power.iloc[:, i].argmax() == np.abs(power.index - f).argmin()

    fs = 1000
    freqs = np.linspace(10, 100, 10)
    fb = nap.generate_morlet_filterbank(
        freqs, fs, gaussian_width=3.5, window_length=1.0, precision=16
    )
    power = np.abs(nap.compute_power_spectral_density(fb))
    for i, f in enumerate(freqs):
        assert power.iloc[:, i].argmax() == np.abs(power.index - f).argmin()

    fs = 10000
    freqs = np.linspace(100, 1000, 10)
    fb = nap.generate_morlet_filterbank(
        freqs, fs, gaussian_width=3.5, window_length=1.0, precision=16
    )
    power = np.abs(nap.compute_power_spectral_density(fb))
    for i, f in enumerate(freqs):
        assert power.iloc[:, i].argmax() == np.abs(power.index - f).argmin()

    fs = 1000
    freqs = np.linspace(10, 100, 10)
    fb = nap.generate_morlet_filterbank(
        freqs, fs, gaussian_width=1.5, window_length=3.0, precision=16
    )
    power = np.abs(nap.compute_power_spectral_density(fb))
    for i, f in enumerate(freqs):
        assert power.iloc[:, i].argmax() == np.abs(power.index - f).argmin()

    fs = 10000
    freqs = np.linspace(100, 1000, 10)
    fb = nap.generate_morlet_filterbank(
        freqs, fs, gaussian_width=1.5, window_length=3.0, precision=16
    )
    power = np.abs(nap.compute_power_spectral_density(fb))
    for i, f in enumerate(freqs):
        assert power.iloc[:, i].argmax() == np.abs(power.index - f).argmin()


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
            [],
            1000,
            1.5,
            1.0,
            16,
            pytest.raises(ValueError, match="Given list of freqs cannot be empty."),
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
        d=np.sin(t * 50 * np.pi * 2)
        * np.interp(np.linspace(0, 1, 1001), [0, 0.5, 1], [0, 1, 0]),
        t=t,
    )
    freqs = (10, 100, 10)
    mwt = nap.compute_wavelet_transform(sig, fs=None, freqs=freqs)
    mwt2 = nap.compute_wavelet_transform(sig, fs=None, freqs=np.linspace(10, 100, 10))
    assert np.array_equal(mwt, mwt2)

    t = np.linspace(0, 1, 1001)
    sig = nap.Tsd(
        d=np.sin(t * 50 * np.pi * 2)
        * np.interp(np.linspace(0, 1, 1001), [0, 0.5, 1], [0, 1, 0]),
        t=t,
    )
    freqs = (10, 100, 10, True)
    mwt = nap.compute_wavelet_transform(sig, fs=None, freqs=freqs)
    mwt2 = nap.compute_wavelet_transform(sig, fs=None, freqs=np.geomspace(10, 100, 10))
    assert np.array_equal(mwt, mwt2)

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
    freqs = (1, 51, 6)
    mwt = nap.compute_wavelet_transform(sig, fs=None, freqs=freqs)
    assert mwt.shape == (1024, 6)

    t = np.linspace(0, 1, 1024)
    sig = nap.TsdFrame(d=np.random.random((1024, 4)), t=t)
    freqs = np.linspace(1, 600, 10)
    mwt = nap.compute_wavelet_transform(sig, fs=None, freqs=freqs)
    assert mwt.shape == (1024, 10, 4)

    t = np.linspace(0, 1, 1024)
    sig = nap.TsdTensor(d=np.random.random((1024, 4, 2)), t=t)
    freqs = np.linspace(1, 600, 10)
    mwt = nap.compute_wavelet_transform(sig, fs=None, freqs=freqs)
    assert mwt.shape == (1024, 10, 4, 2)

    # Testing against manual convolution for l1 norm
    t = np.linspace(0, 1, 1024)
    sig = nap.TsdTensor(d=np.random.random((1024, 4, 2)), t=t)
    freqs = np.linspace(1, 600, 10)
    mwt = nap.compute_wavelet_transform(
        sig, fs=None, freqs=freqs, gaussian_width=1.5, window_length=1.0, norm="l1"
    )
    output_shape = (sig.shape[0], len(freqs), *sig.shape[1:])
    sig = sig.reshape((sig.shape[0], np.prod(sig.shape[1:])))
    filter_bank = nap.generate_morlet_filterbank(freqs, 1024, 1.5, 1.0, precision=16)
    convolved_real = sig.convolve(filter_bank.real().values)
    convolved_imag = sig.convolve(filter_bank.imag().values)
    convolved = convolved_real.values + convolved_imag.values * 1j
    coef = convolved / (1024 / freqs)
    cwt = np.expand_dims(coef, -1) if len(coef.shape) == 2 else coef
    mwt2 = nap.TsdTensor(
        t=sig.index, d=cwt.reshape(output_shape), time_support=sig.time_support
    )
    assert np.array_equal(mwt, mwt2)

    # Testing against manual convolution for l2 norm
    t = np.linspace(0, 1, 1024)
    sig = nap.TsdTensor(d=np.random.random((1024, 4, 2)), t=t)
    freqs = np.linspace(1, 600, 10)
    mwt = nap.compute_wavelet_transform(
        sig, fs=None, freqs=freqs, gaussian_width=1.5, window_length=1.0, norm="l2"
    )
    output_shape = (sig.shape[0], len(freqs), *sig.shape[1:])
    sig = sig.reshape((sig.shape[0], np.prod(sig.shape[1:])))
    filter_bank = nap.generate_morlet_filterbank(freqs, 1024, 1.5, 1.0, precision=16)
    convolved_real = sig.convolve(filter_bank.real().values)
    convolved_imag = sig.convolve(filter_bank.imag().values)
    convolved = convolved_real.values + convolved_imag.values * 1j
    coef = convolved / (1024 / np.sqrt(freqs))
    cwt = np.expand_dims(coef, -1) if len(coef.shape) == 2 else coef
    mwt2 = nap.TsdTensor(
        t=sig.index, d=cwt.reshape(output_shape), time_support=sig.time_support
    )
    assert np.array_equal(mwt, mwt2)

    # Testing against manual convolution for no normalization
    t = np.linspace(0, 1, 1024)
    sig = nap.TsdTensor(d=np.random.random((1024, 4, 2)), t=t)
    freqs = np.linspace(1, 600, 10)
    mwt = nap.compute_wavelet_transform(
        sig, fs=None, freqs=freqs, gaussian_width=1.5, window_length=1.0, norm=None
    )
    output_shape = (sig.shape[0], len(freqs), *sig.shape[1:])
    sig = sig.reshape((sig.shape[0], np.prod(sig.shape[1:])))
    filter_bank = nap.generate_morlet_filterbank(freqs, 1024, 1.5, 1.0, precision=16)
    convolved_real = sig.convolve(filter_bank.real().values)
    convolved_imag = sig.convolve(filter_bank.imag().values)
    convolved = convolved_real.values + convolved_imag.values * 1j
    coef = convolved
    cwt = np.expand_dims(coef, -1) if len(coef.shape) == 2 else coef
    mwt2 = nap.TsdTensor(
        t=sig.index, d=cwt.reshape(output_shape), time_support=sig.time_support
    )
    assert np.array_equal(mwt, mwt2)


@pytest.mark.parametrize(
    "sig, fs, freqs, gaussian_width, window_length, precision, norm, expectation",
    [
        (
            nap.TsdTensor(d=np.random.random((1024, 4, 2)), t=np.linspace(0, 1, 1024)),
            None,
            np.linspace(0, 600, 10),
            1.5,
            1.0,
            16,
            "l1",
            pytest.raises(
                ValueError, match="All frequencies in freqs must be strictly positive"
            ),
        ),
        (
            nap.TsdTensor(d=np.random.random((1024, 4, 2)), t=np.linspace(0, 1, 1024)),
            None,
            np.linspace(1, 600, 10),
            -1.5,
            1.0,
            16,
            "l1",
            pytest.raises(
                ValueError, match="gaussian_width must be a positive number."
            ),
        ),
        (
            nap.TsdTensor(d=np.random.random((1024, 4, 2)), t=np.linspace(0, 1, 1024)),
            None,
            np.linspace(1, 600, 10),
            1.5,
            1.0,
            16,
            "l3",
            pytest.raises(
                ValueError, match="norm parameter must be 'l1', 'l2', or None."
            ),
        ),
        (
            nap.TsdTensor(d=np.random.random((1024, 4, 2)), t=np.linspace(0, 1, 1024)),
            None,
            None,
            1.5,
            1.0,
            16,
            "l1",
            pytest.raises(
                TypeError, match="`freqs` must be a ndarray or tuple instance."
            ),
        ),
        (
            "not_a_signal",
            None,
            np.linspace(10, 100, 10),
            1.5,
            1.0,
            16,
            "l1",
            pytest.raises(
                TypeError, match="`sig` must be instance of Tsd, TsdFrame, or TsdTensor"
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
