import warnings
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest
from scipy import signal

import pynapple as nap


# @pytest.fixture
def sample_data():
    # Create a sample Tsd data object
    t = np.linspace(0, 1, 500)
    d = np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.5, t.shape)
    time_support = nap.IntervalSet(start=[0], end=[1])
    return nap.Tsd(t=t, d=d, time_support=time_support)


def sample_data_with_nan():
    # Create a sample Tsd data object
    t = np.linspace(0, 1, 500)
    d = np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.5, t.shape)
    d[10] = np.nan
    time_support = nap.IntervalSet(start=[0], end=[1])
    return nap.Tsd(t=t, d=d, time_support=time_support)


def compare_scipy(tsd, ep, order, freq, fs, btype):
    sos = signal.butter(order, freq, btype=btype, fs=fs, output="sos")
    out_sci = []
    for iset in ep:
        out_sci.append(signal.sosfiltfilt(sos, tsd.restrict(iset).d, axis=0))
    out_sci = np.concatenate(out_sci, axis=0)
    return out_sci


def compare_sinc(tsd, ep, transition_bandwidth, freq, fs, ftype):

    kernel = nap.process.filtering._get_windowed_sinc_kernel(
        freq, ftype, fs, transition_bandwidth
    )
    return tsd.convolve(kernel, ep).d


@pytest.mark.parametrize("freq", [10])
@pytest.mark.parametrize("mode", ["butter", "sinc"])
@pytest.mark.parametrize("order", [4])
@pytest.mark.parametrize("transition_bandwidth", [0.02])
@pytest.mark.parametrize("shape", [(5000,), (5000, 2), (5000, 2, 3)])
@pytest.mark.parametrize("sampling_frequency", [None, 5000.0])
@pytest.mark.parametrize(
    "ep",
    [
        nap.IntervalSet(start=[0], end=[1]),
        nap.IntervalSet(start=[0, 0.5], end=[0.4, 1]),
    ],
)
def test_low_pass(
    freq, mode, order, transition_bandwidth, shape, sampling_frequency, ep
):
    t = np.linspace(0, 1, shape[0])
    y = np.squeeze(
        np.cos(np.pi * 2 * 80 * t).reshape(-1, *[1] * (len(shape) - 1))
        + np.random.normal(size=shape)
    )

    if len(shape) == 1:
        tsd = nap.Tsd(t, y, time_support=ep)
    elif len(shape) == 2:
        tsd = nap.TsdFrame(t, y, time_support=ep)
    else:
        tsd = nap.TsdTensor(t, y, time_support=ep)
    if sampling_frequency is not None and sampling_frequency != tsd.rate:
        sampling_frequency = tsd.rate

    out = nap.apply_lowpass_filter(
        tsd,
        freq,
        fs=sampling_frequency,
        mode=mode,
        order=order,
        transition_bandwidth=transition_bandwidth,
    )
    if mode == "butter":
        out_sci = compare_scipy(tsd, ep, order, freq, tsd.rate, "lowpass")
        np.testing.assert_array_almost_equal(out.d, out_sci)

    if mode == "sinc":
        out_sinc = compare_sinc(
            tsd, ep, transition_bandwidth, freq, tsd.rate, "lowpass"
        )
        np.testing.assert_array_almost_equal(out.d, out_sinc)

    assert isinstance(out, type(tsd))
    assert np.all(out.t == tsd.t)
    assert np.all(out.time_support == tsd.time_support)
    if isinstance(tsd, nap.TsdFrame):
        assert np.all(tsd.columns == out.columns)


@pytest.mark.parametrize("freq", [10])
@pytest.mark.parametrize("mode", ["butter", "sinc"])
@pytest.mark.parametrize("order", [4])
@pytest.mark.parametrize("transition_bandwidth", [0.02])
@pytest.mark.parametrize("shape", [(5000,), (5000, 2), (5000, 2, 3)])
@pytest.mark.parametrize("sampling_frequency", [None, 5000.0])
@pytest.mark.parametrize(
    "ep",
    [
        nap.IntervalSet(start=[0], end=[1]),
        nap.IntervalSet(start=[0, 0.5], end=[0.4, 1]),
    ],
)
def test_high_pass(
    freq, mode, order, transition_bandwidth, shape, sampling_frequency, ep
):
    t = np.linspace(0, 1, shape[0])
    y = np.squeeze(
        np.cos(np.pi * 2 * 80 * t).reshape(-1, *[1] * (len(shape) - 1))
        + np.random.normal(size=shape)
    )

    if len(shape) == 1:
        tsd = nap.Tsd(t, y, time_support=ep)
    elif len(shape) == 2:
        tsd = nap.TsdFrame(t, y, time_support=ep)
    else:
        tsd = nap.TsdTensor(t, y, time_support=ep)
    if sampling_frequency is not None and sampling_frequency != tsd.rate:
        sampling_frequency = tsd.rate

    out = nap.apply_highpass_filter(
        tsd,
        freq,
        fs=sampling_frequency,
        mode=mode,
        order=order,
        transition_bandwidth=transition_bandwidth,
    )

    if mode == "sinc":
        out_sinc = compare_sinc(
            tsd, ep, transition_bandwidth, freq, tsd.rate, "highpass"
        )
        np.testing.assert_array_almost_equal(out.d, out_sinc)

    if mode == "butter":
        out_sci = compare_scipy(tsd, ep, order, freq, tsd.rate, "highpass")
        np.testing.assert_array_almost_equal(out.d, out_sci)

    assert isinstance(out, type(tsd))
    assert np.all(out.t == tsd.t)
    assert np.all(out.time_support == tsd.time_support)
    if isinstance(tsd, nap.TsdFrame):
        assert np.all(tsd.columns == out.columns)


@pytest.mark.parametrize("freq", [[10, 30]])
@pytest.mark.parametrize("mode", ["butter", "sinc"])
@pytest.mark.parametrize("order", [4])
@pytest.mark.parametrize("transition_bandwidth", [0.02])
@pytest.mark.parametrize("shape", [(5000,), (5000, 2), (5000, 2, 3)])
@pytest.mark.parametrize("sampling_frequency", [None, 5000.0])
@pytest.mark.parametrize(
    "ep",
    [
        nap.IntervalSet(start=[0], end=[1]),
        nap.IntervalSet(start=[0, 0.5], end=[0.4, 1]),
    ],
)
def test_bandpass(
    freq, mode, order, transition_bandwidth, shape, sampling_frequency, ep
):
    t = np.linspace(0, 1, shape[0])
    y = np.squeeze(
        np.cos(np.pi * 2 * 80 * t).reshape(-1, *[1] * (len(shape) - 1))
        + np.random.normal(size=shape)
    )

    if len(shape) == 1:
        tsd = nap.Tsd(t, y, time_support=ep)
    elif len(shape) == 2:
        tsd = nap.TsdFrame(t, y, time_support=ep)
    else:
        tsd = nap.TsdTensor(t, y, time_support=ep)
    if sampling_frequency is not None and sampling_frequency != tsd.rate:
        sampling_frequency = tsd.rate

    out = nap.apply_bandpass_filter(
        tsd,
        freq,
        fs=sampling_frequency,
        mode=mode,
        order=order,
        transition_bandwidth=transition_bandwidth,
    )

    if mode == "sinc":
        out_sinc = compare_sinc(
            tsd, ep, transition_bandwidth, freq, tsd.rate, "bandpass"
        )
        np.testing.assert_array_almost_equal(out.d, out_sinc)

    if mode == "butter":
        out_sci = compare_scipy(tsd, ep, order, freq, tsd.rate, "bandpass")
        np.testing.assert_array_almost_equal(out.d, out_sci)

    assert isinstance(out, type(tsd))
    assert np.all(out.t == tsd.t)
    assert np.all(out.time_support == tsd.time_support)
    if isinstance(tsd, nap.TsdFrame):
        assert np.all(tsd.columns == out.columns)


@pytest.mark.parametrize("freq", [[10, 30]])
@pytest.mark.parametrize("mode", ["butter", "sinc"])
@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("transition_bandwidth", [0.02])
@pytest.mark.parametrize("shape", [(5000,), (5000, 2), (5000, 2, 3)])
@pytest.mark.parametrize("sampling_frequency", [None, 5000.0])
@pytest.mark.parametrize(
    "ep",
    [
        nap.IntervalSet(start=[0], end=[1]),
        nap.IntervalSet(start=[0, 0.5], end=[0.4, 1]),
    ],
)
def test_bandstop(
    freq, mode, order, transition_bandwidth, shape, sampling_frequency, ep
):
    t = np.linspace(0, 1, shape[0])
    y = np.squeeze(
        np.cos(np.pi * 2 * 80 * t).reshape(-1, *[1] * (len(shape) - 1))
        + np.random.normal(size=shape)
    )

    if len(shape) == 1:
        tsd = nap.Tsd(t, y, time_support=ep)
    elif len(shape) == 2:
        tsd = nap.TsdFrame(t, y, time_support=ep)
    else:
        tsd = nap.TsdTensor(t, y, time_support=ep)
    if sampling_frequency is not None and sampling_frequency != tsd.rate:
        sampling_frequency = tsd.rate

    out = nap.apply_bandstop_filter(
        tsd,
        freq,
        fs=sampling_frequency,
        mode=mode,
        order=order,
        transition_bandwidth=transition_bandwidth,
    )

    if mode == "sinc":
        out_sinc = compare_sinc(
            tsd, ep, transition_bandwidth, freq, tsd.rate, "bandstop"
        )
        np.testing.assert_array_almost_equal(out.d, out_sinc)

    if mode == "butter":
        out_sci = compare_scipy(tsd, ep, order, freq, tsd.rate, "bandstop")
        np.testing.assert_array_almost_equal(out.d, out_sci)

    assert isinstance(out, type(tsd))
    assert np.all(out.t == tsd.t)
    assert np.all(out.time_support == tsd.time_support)
    if isinstance(tsd, nap.TsdFrame):
        assert np.all(tsd.columns == out.columns)


########################################################################
# Errors
########################################################################
@pytest.mark.parametrize(
    "func, freq",
    [
        (nap.apply_lowpass_filter, 10),
        (nap.apply_highpass_filter, 10),
        (nap.apply_bandpass_filter, [10, 20]),
        (nap.apply_bandstop_filter, [10, 20]),
    ],
)
@pytest.mark.parametrize(
    "data, fs, mode, order, transition_bandwidth, expected_exception",
    [
        (
            sample_data(),
            None,
            "butter",
            "a",
            0.02,
            pytest.raises(
                ValueError,
                match="Invalid value for 'order': Parameter 'order' should be of type int",
            ),
        ),
        (
            "invalid_data",
            None,
            "butter",
            4,
            0.02,
            pytest.raises(
                ValueError,
                match="Invalid value: invalid_data. First argument should be of type Tsd, TsdFrame or TsdTensor",
            ),
        ),
        (
            sample_data(),
            None,
            "invalid_mode",
            4,
            0.02,
            pytest.raises(
                ValueError,
                match="Unrecognized filter mode. Choose either 'butter' or 'sinc'",
            ),
        ),
        (
            sample_data(),
            "invalid_fs",
            "butter",
            4,
            0.02,
            pytest.raises(
                ValueError,
                match="Invalid value for 'fs'. Parameter 'fs' should be of type float or int",
            ),
        ),
        (
            sample_data(),
            None,
            "sinc",
            4,
            "a",
            pytest.raises(
                ValueError,
                match="Invalid value for 'transition_bandwidth'. 'transition_bandwidth' should be of type float",
            ),
        ),
        (
            sample_data_with_nan(),
            None,
            "sinc",
            4,
            0.02,
            pytest.raises(
                ValueError,
                match="The input signal contains NaN values, which are not supported for filtering",
            ),
        ),
        (
            sample_data_with_nan(),
            None,
            "butter",
            4,
            0.02,
            pytest.raises(
                ValueError,
                match="The input signal contains NaN values, which are not supported for filtering",
            ),
        ),
    ],
)
def test_compute_filtered_signal_raise_errors(
    func, freq, data, fs, mode, order, transition_bandwidth, expected_exception
):
    with expected_exception:
        func(
            data,
            freq,
            fs=fs,
            mode=mode,
            order=order,
            transition_bandwidth=transition_bandwidth,
        )


@pytest.mark.parametrize(
    "func, freq, expected_exception",
    [
        (
            nap.apply_lowpass_filter,
            "a",
            pytest.raises(
                ValueError,
                match=r"lowpass filter require a single number. a provided instead.",
            ),
        ),
        (
            nap.apply_highpass_filter,
            "b",
            pytest.raises(
                ValueError,
                match=r"highpass filter require a single number. b provided instead.",
            ),
        ),
        (
            nap.apply_bandpass_filter,
            [10, "b"],
            pytest.raises(
                ValueError,
                match="bandpass filter require a tuple of two numbers. \[10, 'b'\] provided instead.",
            ),
        ),
        (
            nap.apply_bandstop_filter,
            [10, 20, 30],
            pytest.raises(
                ValueError,
                match=r"bandstop filter require a tuple of two numbers. \[10, 20, 30\] provided instead.",
            ),
        ),
    ],
)
def test_compute_filtered_signal_bad_freq(func, freq, expected_exception):
    with expected_exception:
        func(sample_data(), freq)


#################################################################
# Test with edge-case frequencies close to Nyquist frequency
@pytest.mark.parametrize("nyquist_fraction", [0.99, 0.999])
@pytest.mark.parametrize("order", [2, 4])
def test_filtering_nyquist_edge_case(nyquist_fraction, order):
    data = sample_data()
    nyquist_freq = 0.5 * data.rate
    freq = nyquist_freq * nyquist_fraction

    out = nap.filtering.apply_lowpass_filter(data, freq, order=order)
    assert isinstance(out, type(data))
    np.testing.assert_allclose(out.t, data.t)
    np.testing.assert_allclose(out.time_support, data.time_support)


#################################################################
# Test windowedsinc kernel


@pytest.mark.parametrize("tb", [0.2, 0.3])
def test_get_odd_kernel(tb):
    kernel = nap.process.filtering._get_windowed_sinc_kernel(
        1, "lowpass", 4, transition_bandwidth=tb
    )
    assert len(kernel) % 2 != 0


@pytest.mark.parametrize(
    "filter_type, expected_exception",
    [
        ("a", pytest.raises(ValueError)),
    ],
)
def test_get_kernel_error(filter_type, expected_exception):
    with expected_exception:
        nap.process.filtering._get_windowed_sinc_kernel(1, filter_type, 4)


def test_get__error():
    with pytest.raises(
        TypeError,
        match=r"apply_lowpass_filter\(\) missing 1 required positional argument: 'data'",
    ):
        nap.apply_lowpass_filter(cutoff=0.25)


def test_compare_sinc_kernel():
    kernel = nap.process.filtering._get_windowed_sinc_kernel(1, "lowpass", 4)
    x = np.arange(-(len(kernel) // 2), 1 + len(kernel) // 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kernel2 = np.sin(2 * np.pi * x * 0.25) / x  # (2*np.pi*x*0.25)
    kernel2[len(kernel) // 2] = 0.25 * 2 * np.pi
    kernel2 = kernel2
    kernel2 = kernel2 * np.blackman(len(kernel2))
    kernel2 /= kernel2.sum()
    np.testing.assert_allclose(kernel, kernel2)

    ikernel = nap.process.filtering._compute_spectral_inversion(kernel)
    ikernel2 = kernel2 * -1.0
    ikernel2[len(ikernel2) // 2] = 1.0 + ikernel2[len(kernel2) // 2]
    np.testing.assert_allclose(ikernel, ikernel2)


@pytest.mark.parametrize(
    "cutoff, fs, filter_type, mode, order, tb",
    [
        (250, 1000, "lowpass", "butter", 4, 0.02),
        (250, 1000, "lowpass", "sinc", 4, 0.02),
    ],
)
def test_get_filter_frequency_response(cutoff, fs, filter_type, mode, order, tb):
    output = nap.get_filter_frequency_response(cutoff, fs, filter_type, mode, order, tb)
    assert isinstance(output, pd.Series)
    if mode == "butter":
        sos = nap.process.filtering._get_butter_coefficients(
            cutoff, filter_type, fs, order
        )
        w, h = signal.sosfreqz(sos, worN=1024, fs=fs)
        np.testing.assert_array_almost_equal(w, output.index.values)
        np.testing.assert_array_almost_equal(np.abs(h), output.values)
    if mode == "sinc":
        kernel = nap.process.filtering._get_windowed_sinc_kernel(
            cutoff, filter_type, fs, tb
        )
        fft_result = np.fft.fft(kernel)
        fft_result = np.fft.fftshift(fft_result)
        fft_freq = np.fft.fftfreq(n=len(kernel), d=1 / fs)
        fft_freq = np.fft.fftshift(fft_freq)
        np.testing.assert_array_almost_equal(
            fft_freq[fft_freq >= 0], output.index.values
        )
        np.testing.assert_array_almost_equal(
            np.abs(fft_result[fft_freq >= 0]), output.values
        )


def test_get_filter_frequency_response_error():
    with pytest.raises(
        ValueError, match="Unrecognized filter mode. Choose either 'butter' or 'sinc'"
    ):
        nap.get_filter_frequency_response(250, 1000, "lowpass", "a", 4, 0.02)
