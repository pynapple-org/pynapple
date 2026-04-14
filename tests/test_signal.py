import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from scipy.signal import hilbert

import pynapple as nap


@pytest.mark.parametrize(
    "param_name, invalid_value, exception",
    [
        (
            "data",
            None,
            pytest.raises(
                TypeError, match="`data` must be `Tsd`, got <class 'NoneType'>"
            ),
        ),
        (
            "data",
            "invalid_data",
            pytest.raises(TypeError, match="`data` must be `Tsd`, got <class 'str'>"),
        ),
        (
            "epoch",
            None,
            pytest.raises(
                TypeError,
                match="`epoch` must be `IntervalSet`, got <class 'NoneType'>",
            ),
        ),
        (
            "epoch",
            "invalid_epoch",
            pytest.raises(
                TypeError,
                match="`epoch` must be `IntervalSet`, got <class 'str'>",
            ),
        ),
        (
            "freq_band",
            (10, "not_a_number"),
            pytest.raises(TypeError, match="`freq_band` must contain numeric values"),
        ),
        (
            "freq_band",
            (10, 5),
            pytest.raises(
                ValueError,
                match=re.escape("`freq_band` must be (min, max) with min < max"),
            ),
        ),
        (
            "thresh_band",
            (1, "not_a_number"),
            pytest.raises(TypeError, match="`thresh_band` must contain numeric values"),
        ),
        (
            "thresh_band",
            (10, 5),
            pytest.raises(
                ValueError,
                match=re.escape("`thresh_band` must be (min, max) with min < max"),
            ),
        ),
        (
            "duration_band",
            (1, "not_a_number"),
            pytest.raises(
                TypeError, match="`duration_band` must contain numeric values"
            ),
        ),
        (
            "duration_band",
            (5,),
            pytest.raises(
                ValueError, match="`duration_band` must have length 2, got 1"
            ),
        ),
        (
            "duration_band",
            (5, 1),
            pytest.raises(
                ValueError,
                match=re.escape("`duration_band` must be (min, max) with min < max"),
            ),
        ),
        (
            "min_inter_duration",
            "string",
            pytest.raises(TypeError, match="`min_inter_duration` must be a number"),
        ),
        (
            "min_inter_duration",
            -0.1,
            pytest.raises(ValueError, match="`min_inter_duration` must be >= 0"),
        ),
        (
            "fs",
            "string",
            pytest.raises(TypeError, match="`fs` must be a number or None"),
        ),
        (
            "fs",
            -1,
            pytest.raises(ValueError, match="`fs` must be > 0"),
        ),
        (
            "wsize",
            "string",
            pytest.raises(TypeError, match="`wsize` must be an integer"),
        ),
        (
            "wsize",
            -5,
            pytest.raises(ValueError, match="`wsize` must be > 0"),
        ),
        (
            "wsize",
            0,
            pytest.raises(ValueError, match="`wsize` must be > 0"),
        ),
        (
            "wsize",
            2,
            pytest.raises(
                ValueError,
                match="`wsize` should be odd for symmetric smoothing",
            ),
        ),
        # Valid cases (does not raise exceptions)
        (
            "data",
            nap.Tsd(t=np.linspace(0, 5, 100), d=np.sin(np.linspace(0, 5, 100))),
            does_not_raise(),
        ),
        ("epoch", nap.IntervalSet(start=0, end=5), does_not_raise()),
        ("freq_band", (10, 30), does_not_raise()),
        ("thresh_band", (1, 10), does_not_raise()),
        ("duration_band", (0.1, 2), does_not_raise()),
        ("min_inter_duration", 0.02, does_not_raise()),
        ("fs", 1000, does_not_raise()),
        ("fs", None, does_not_raise()),
        ("wsize", 51, does_not_raise()),
    ],
)
def test_detect_oscillatory_events_input_types(param_name, invalid_value, exception):
    # Create some valid input values for other parameters
    duration = 5
    fs = 1000
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 25 * t)
    ts = nap.Tsd(t=t, d=signal)
    epoch = nap.IntervalSet(start=0, end=duration)
    min_dur = 0.1
    max_dur = 2
    min_inter = 0.02
    freq_band = (10, 30)
    thresh_band = (1, 10)

    # Modify the parameter based on the test case
    kwargs = {
        "data": ts,
        "epoch": epoch,
        "freq_band": freq_band,
        "thresh_band": thresh_band,
        "duration_band": (min_dur, max_dur),
        "min_inter_duration": min_inter,
        "fs": fs,
        "wsize": 51,
    }
    kwargs[param_name] = invalid_value

    with exception:
        nap.detect_oscillatory_events(**kwargs)


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
                [
                    np.sin(np.linspace(0, 1, 500)),
                    np.cos(np.linspace(0, 1, 500)),
                ],
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
                [
                    np.sin(np.linspace(0, 1, 500)),
                    np.cos(np.linspace(0, 1, 500)),
                ],
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
                [
                    np.sin(np.linspace(0, 1, 500)),
                    np.cos(np.linspace(0, 1, 500)),
                ],
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
