from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from scipy.signal import hilbert

import pynapple as nap


@pytest.mark.parametrize(
    "freq_band, thresh_band, num_events, start, end",
    [
        ((10, 30), (1, 10), 1, 0, 2),
        ((40, 60), (1, 10), 1, 3, 5),
        ((100, 150), (1, 10), 0, None, None),
    ],
)
def test_detect_oscillatory_events(freq_band, thresh_band, num_events, start, end):
    fs = 1000
    duration = 5
    min_dur = 0.1
    max_dur = 2
    min_inter = 0.02

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.zeros_like(t)

    # 25 Hz oscillation from 0-2s
    freq_1 = 25
    mask1 = (t >= 0) & (t < 2)
    signal[mask1] = np.sin(2 * np.pi * freq_1 * t[mask1])

    # 50 Hz oscillation from 3-5s
    freq_2 = 50
    mask2 = (t >= 3) & (t < 5)
    signal[mask2] = np.sin(2 * np.pi * freq_2 * t[mask2])

    ts = nap.Tsd(t=t, d=signal)
    epoch = nap.IntervalSet(start=0, end=duration)
    osc_ep = nap.detect_oscillatory_events(
        ts, epoch, freq_band, thresh_band, (min_dur, max_dur), min_inter
    )

    assert len(osc_ep) == num_events  # Only one event in given freq_band

    if num_events > 0:
        # Start and end should be close to actuals +/- a small amount
        detected_start = osc_ep.start[0]
        detected_end = osc_ep.end[0]
        assert np.isclose(start, detected_start, atol=0.05)
        assert np.isclose(end, detected_end, atol=0.05)

        # Check we store power, amplitude, and peak_time
        for key in ["power", "amplitude", "peak_time"]:
            assert key in osc_ep._metadata

        # Check peak_time is within the interval
        peak_time = osc_ep._metadata["peak_time"][0]
        assert start <= peak_time <= end


@pytest.mark.parametrize(
    "func",
    [
        nap.apply_hilbert_transform,
        nap.compute_hilbert_envelope,
        nap.compute_hilbert_phase,
    ],
)
@pytest.mark.parametrize(
    "input, expectation",
    [
        (nap.Tsd(d=np.ones(3), t=[1, 2, 3]), does_not_raise()),
        (nap.TsdFrame(d=np.ones((3, 3)), t=[1, 2, 3]), does_not_raise()),
        (
            np.ones((3, 3)),
            pytest.raises(TypeError, match="data should be a Tsd or TsdFrame."),
        ),
        (
            nap.TsdTensor(d=np.ones((3, 3, 3)), t=[1, 2, 3]),
            pytest.raises(TypeError, match="data should be a Tsd or TsdFrame."),
        ),
        (
            [],
            pytest.raises(TypeError, match="data should be a Tsd or TsdFrame."),
        ),
        (
            nap.IntervalSet(1, 2),
            pytest.raises(TypeError, match="data should be a Tsd or TsdFrame."),
        ),
        (
            None,
            pytest.raises(TypeError, match="data should be a Tsd or TsdFrame."),
        ),
    ],
)
def test_hilbert_type_errors(input, func, expectation):
    with expectation:
        func(input)


@pytest.mark.parametrize(
    "data",
    [
        nap.Tsd(t=np.linspace(0, 1, 500), d=np.sin(np.linspace(0, 1, 500))),
        nap.TsdFrame(
            t=np.linspace(0, 1, 500),
            d=np.stack(
                [np.sin(np.linspace(0, 1, 500)), np.cos(np.linspace(0, 1, 500))],
                axis=1,
            ),
        ),
    ],
)
def test_apply_hilbert_transform(data):
    result = nap.apply_hilbert_transform(data)
    expected = hilbert(data.values, axis=0)

    assert isinstance(result, type(data))
    np.testing.assert_array_equal(data.time_support, result.time_support)
    np.testing.assert_array_equal(data.times(), result.times())

    np.testing.assert_array_equal(result.values, expected)


@pytest.mark.parametrize(
    "data",
    [
        nap.Tsd(t=np.linspace(0, 1, 500), d=np.sin(np.linspace(0, 1, 500))),
        nap.TsdFrame(
            t=np.linspace(0, 1, 500),
            d=np.stack(
                [np.sin(np.linspace(0, 1, 500)), np.cos(np.linspace(0, 1, 500))],
                axis=1,
            ),
        ),
    ],
)
def test_compute_hilbert_phase(data):
    result = nap.compute_hilbert_phase(data)
    analytic_signal = hilbert(data.values, axis=0)
    phase = np.angle(analytic_signal)
    expected = np.mod(np.unwrap(phase), 2 * np.pi)

    assert isinstance(result, type(data))
    np.testing.assert_array_equal(data.time_support, result.time_support)
    np.testing.assert_array_equal(data.times(), result.times())

    np.testing.assert_array_equal(result.values, expected)


@pytest.mark.parametrize(
    "data",
    [
        nap.Tsd(t=np.linspace(0, 1, 500), d=np.sin(np.linspace(0, 1, 500))),
        nap.TsdFrame(
            t=np.linspace(0, 1, 500),
            d=np.stack(
                [np.sin(np.linspace(0, 1, 500)), np.cos(np.linspace(0, 1, 500))],
                axis=1,
            ),
        ),
    ],
)
def test_compute_hilbert_envelope(data):
    result = nap.compute_hilbert_envelope(data)
    analytic_signal = hilbert(data.values, axis=0)
    expected = np.abs(analytic_signal)

    assert isinstance(result, type(data))
    np.testing.assert_array_equal(data.time_support, result.time_support)
    np.testing.assert_array_equal(data.times(), result.times())

    np.testing.assert_array_equal(result.values, expected)
