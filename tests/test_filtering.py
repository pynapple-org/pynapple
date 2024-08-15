import pytest
import pynapple as nap
import numpy as np
from scipy import signal
from contextlib import nullcontext as does_not_raise


@pytest.fixture
def sample_data():
    # Create a sample Tsd data object
    t = np.linspace(0, 1, 500)
    d = np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.5, t.shape)
    time_support = nap.IntervalSet(start=[0], end=[1])
    return nap.Tsd(t=t, d=d, time_support=time_support)


@pytest.mark.parametrize("freq", [10, 100])
@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("btype", ["lowpass", "highpass"])
@pytest.mark.parametrize("shape", [(5000,), (5000, 2), (5000, 2, 3)])
@pytest.mark.parametrize(
    "ep",
    [
        nap.IntervalSet(start=[0], end=[1]),
        nap.IntervalSet(start=[0, 0.5], end=[0.4, 1]),
        nap.IntervalSet(start=[0, 0.5, 0.95], end=[0.4, 0.9, 1])
    ]
)
def test_filtering_single_freq_match_sci(freq, order, btype, shape: tuple, ep):

    t = np.linspace(0, 1, shape[0])
    y = np.squeeze(np.cos(np.pi * 2 * 80 * t).reshape(-1, *[1]*(len(shape) - 1)) + np.random.normal(size=shape))

    if len(shape) == 1:
        tsd = nap.Tsd(t, y, time_support=ep)
    elif len(shape) == 2:
        tsd = nap.TsdFrame(t, y, time_support=ep)
    else:
        tsd = nap.TsdTensor(t, y, time_support=ep)
    out = nap.compute_filtered_signal(tsd, freq_band=freq, filter_type=btype, order=order)
    b, a = signal.butter(order, freq, fs=tsd.rate, btype=btype)
    out_sci = []
    for iset in ep:
        out_sci.append(signal.filtfilt(b, a, tsd.restrict(iset).d, axis=0))
    out_sci = np.concatenate(out_sci, axis=0)
    np.testing.assert_array_equal(out.d, out_sci)


@pytest.mark.parametrize("freq", [[10, 30], [100,150]])
@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("btype", ["bandpass", "bandstop"])
@pytest.mark.parametrize("shape", [(5000,), (5000, 2), (5000, 2, 3)])
@pytest.mark.parametrize(
    "ep",
    [
        nap.IntervalSet(start=[0], end=[1]),
        nap.IntervalSet(start=[0, 0.5], end=[0.4, 1]),
        nap.IntervalSet(start=[0, 0.5, 0.95], end=[0.4, 0.9, 1])
    ]
)
def test_filtering_freq_band_match_sci(freq, order, btype, shape: tuple, ep):

    t = np.linspace(0, 1, shape[0])
    y = np.squeeze(np.cos(np.pi * 2 * 80 * t).reshape(-1, *[1]*(len(shape) - 1)) + np.random.normal(size=shape))

    if len(shape) == 1:
        tsd = nap.Tsd(t, y, time_support=ep)
    elif len(shape) == 2:
        tsd = nap.TsdFrame(t, y, time_support=ep)
    else:
        tsd = nap.TsdTensor(t, y, time_support=ep)
    out = nap.compute_filtered_signal(tsd, freq_band=freq, filter_type=btype, order=order)
    b, a = signal.butter(order, freq, fs=tsd.rate, btype=btype)
    out_sci = []
    for iset in ep:
        out_sci.append(signal.filtfilt(b, a, tsd.restrict(iset).d, axis=0))
    out_sci = np.concatenate(out_sci, axis=0)
    np.testing.assert_array_equal(out.d, out_sci)


@pytest.mark.parametrize("freq", [10, 100])
@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("btype", ["lowpass", "highpass"])
@pytest.mark.parametrize("shape", [(5000,), (5000, 2), (5000, 2, 3)])
@pytest.mark.parametrize(
    "ep",
    [
        nap.IntervalSet(start=[0], end=[1]),
        nap.IntervalSet(start=[0, 0.5], end=[0.4, 1]),
        nap.IntervalSet(start=[0, 0.5, 0.95], end=[0.4, 0.9, 1])
    ]
)
def test_filtering_single_freq_dtype(freq, order, btype, shape: tuple, ep):
    t = np.linspace(0, 1, shape[0])
    y = np.squeeze(np.cos(np.pi * 2 * 80 * t).reshape(-1, *[1]*(len(shape) - 1)) + np.random.normal(size=shape))

    if len(shape) == 1:
        tsd = nap.Tsd(t, y, time_support=ep)
    elif len(shape) == 2:
        tsd = nap.TsdFrame(t, y, time_support=ep, columns=np.arange(10, 10 + y.shape[1]))
    else:
        tsd = nap.TsdTensor(t, y, time_support=ep)
    out = nap.compute_filtered_signal(tsd, freq_band=freq, filter_type=btype, order=order)
    assert isinstance(out, type(tsd))
    assert np.all(out.t == tsd.t)
    assert np.all(out.time_support == tsd.time_support)
    if isinstance(tsd, nap.TsdFrame):
        assert np.all(tsd.columns == out.columns)


@pytest.mark.parametrize("freq", [[10, 30], [100, 150]])
@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("btype", ["bandpass", "bandstop"])
@pytest.mark.parametrize("shape", [(5000,), (5000, 2), (5000, 2, 3)])
@pytest.mark.parametrize(
    "ep",
    [
        nap.IntervalSet(start=[0], end=[1]),
        nap.IntervalSet(start=[0, 0.5], end=[0.4, 1]),
        nap.IntervalSet(start=[0, 0.5, 0.95], end=[0.4, 0.9, 1])
    ]
)
def test_filtering_freq_band_dtype(freq, order, btype, shape: tuple, ep):
    t = np.linspace(0, 1, shape[0])
    y = np.squeeze(np.cos(np.pi * 2 * 80 * t).reshape(-1, *[1]*(len(shape) - 1)) + np.random.normal(size=shape))

    if len(shape) == 1:
        tsd = nap.Tsd(t, y, time_support=ep)
    elif len(shape) == 2:
        tsd = nap.TsdFrame(t, y, time_support=ep, columns=np.arange(10, 10 + y.shape[1]))
    else:
        tsd = nap.TsdTensor(t, y, time_support=ep)
    out = nap.compute_filtered_signal(tsd, freq_band=freq, filter_type=btype, order=order)
    assert isinstance(out, type(tsd))
    assert np.all(out.t == tsd.t)
    assert np.all(out.time_support == tsd.time_support)
    if isinstance(tsd, nap.TsdFrame):
        assert np.all(tsd.columns == out.columns)


@pytest.mark.parametrize("freq_band, filter_type, order, expected_exception", [
    ((5, 15), "bandpass", 4, does_not_raise()),
    ((5, 15), "bandstop", 4, does_not_raise()),
    (10, "highpass", 4, does_not_raise()),
    (10, "lowpass", 4, does_not_raise()),
    ((5, 15), "invalid_filter", 4, pytest.raises(ValueError, match="Unrecognized filter type")),
    (10, "bandpass", 4, pytest.raises(ValueError,  match="Band-pass/stop filter specification requires two frequencies")),
    ((5, 15), "highpass", 4, pytest.raises(ValueError, match="Low/high-pass filter specification requires a single frequency")),
    (None, "bandpass", 4, pytest.raises(ValueError,  match="Band-pass/stop filter specification requires two frequencies")),
    ((None, 1), "highpass", 4, pytest.raises(ValueError, match="Low/high-pass filter specification requires a single frequency"))
])
def test_compute_filtered_signal(sample_data, freq_band, filter_type, order, expected_exception):
    with expected_exception:
        filtered_data = nap.filtering.compute_filtered_signal(sample_data, freq_band, filter_type, order)
        if not expected_exception:
            assert isinstance(filtered_data, type(sample_data))
            assert filtered_data.d.shape == sample_data.d.shape


# Test with edge-case frequencies close to Nyquist frequency
@pytest.mark.parametrize("nyquist_fraction", [0.99, 0.999])
@pytest.mark.parametrize("order", [2, 4])
def test_filtering_nyquist_edge_case(nyquist_fraction, order, sample_data):
    nyquist_freq = 0.5 * sample_data.rate
    freq = nyquist_freq * nyquist_fraction

    out = nap.filtering.compute_filtered_signal(
        sample_data, freq_band=freq, filter_type="lowpass", order=order
    )
    assert isinstance(out, type(sample_data))
    np.testing.assert_allclose(out.t, sample_data.t)
    np.testing.assert_allclose(out.time_support, sample_data.time_support)
