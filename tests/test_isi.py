"""Tests of burst detection in pynapple.process.isi"""

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from scipy import special, stats

import pynapple as nap
from pynapple.process.isi import _jitpoisson_surprise


def _regular_bursts(n_bursts=200, spikes_per_burst=5, intra=0.01, period=1.5):
    starts = np.arange(n_bursts) * period
    return nap.Ts(
        t=np.concatenate([s + np.arange(spikes_per_burst) * intra for s in starts])
    )


def _assert_bursts(bursts, starts, ends, n_spikes):
    assert isinstance(bursts, nap.IntervalSet)
    np.testing.assert_allclose(bursts.start, starts)
    np.testing.assert_allclose(bursts.end, ends)
    np.testing.assert_array_equal(np.asarray(bursts.n_spikes), n_spikes)


# ---------------------------------------------------------------------------
# detect_spike_bursts: dispatch and errors
# ---------------------------------------------------------------------------


def test_default_method_is_poisson_surprise():
    spikes = _regular_bursts()
    default = nap.detect_spike_bursts(spikes)
    poisson_surprise = nap.detect_spike_bursts(spikes, method="poisson_surprise")
    np.testing.assert_array_equal(default.start, poisson_surprise.start)
    np.testing.assert_array_equal(default.end, poisson_surprise.end)
    assert len(default) == 200
    assert np.all(np.asarray(default.n_spikes) == 5)


@pytest.mark.parametrize(
    "spikes, method, params, expectation",
    [
        (
            None,
            "poisson_surprise",
            {},
            pytest.raises(
                TypeError,
                match="`spikes` must be `Ts` or `TsGroup`, got <class 'NoneType'>",
            ),
        ),
        (
            nap.Tsd(t=[0, 1], d=[0, 1]),
            "poisson_surprise",
            {},
            pytest.raises(TypeError, match="`spikes` must be `Ts` or `TsGroup`"),
        ),
        (
            nap.Ts(t=[0.0, 1.0]),
            "rank_surprise",
            {},
            pytest.raises(ValueError, match="`method` must be one of"),
        ),
        (
            nap.Ts(t=[0.0, 1.0]),
            1,
            {},
            pytest.raises(ValueError, match="`method` must be one of"),
        ),
        (
            nap.Ts(t=[0.0, 1.0]),
            "poisson_surprise",
            {"max_isi_start": 0.1},
            pytest.raises(
                TypeError,
                match=re.escape(
                    "Unknown parameter(s) ['max_isi_start'] "
                    "for method 'poisson_surprise'"
                ),
            ),
        ),
        (
            nap.Ts(t=[0.0, 1.0]),
            "instantaneous_rate",
            {"max_isi_burst": 0.1, "min_surprise": 5},
            pytest.raises(
                TypeError,
                match=re.escape(
                    "Unknown parameter(s) ['max_isi_burst', 'min_surprise']"
                ),
            ),
        ),
        (
            nap.Ts(t=[0.0, 1.0]),
            "log_isi",
            {},
            pytest.raises(ValueError, match="`method` must be one of"),
        ),
        # max_interval parameters
        (
            nap.Ts(t=[0.0, 1.0]),
            "max_interval",
            {"min_spikes_in_burst": 2.5},
            pytest.raises(TypeError, match="`min_spikes_in_burst` must be an integer"),
        ),
        (
            nap.Ts(t=[0.0, 1.0]),
            "max_interval",
            {"max_isi_start": -1},
            pytest.raises(ValueError, match="`max_isi_start` must be > 0"),
        ),
        (
            nap.Ts(t=[0.0, 1.0]),
            "max_interval",
            {"max_isi_burst": None},
            pytest.raises(TypeError, match="`max_isi_burst` must be a number"),
        ),
        (
            nap.Ts(t=[0.0, 1.0]),
            "max_interval",
            {"min_inter_burst_interval": -0.1},
            pytest.raises(ValueError, match="`min_inter_burst_interval` must be >= 0"),
        ),
        (
            nap.Ts(t=[0.0, 1.0]),
            "max_interval",
            {"min_burst_duration": -0.1},
            pytest.raises(ValueError, match="`min_burst_duration` must be >= 0"),
        ),
        (
            nap.Ts(t=[0.0, 1.0]),
            "max_interval",
            {"min_inter_burst_interval": 0, "min_burst_duration": 0},
            does_not_raise(),
        ),
        # instantaneous_rate parameters
        (
            nap.Ts(t=[0.0, 1.0]),
            "instantaneous_rate",
            {"rate_threshold": 0},
            pytest.raises(ValueError, match="`rate_threshold` must be > 0"),
        ),
        (
            nap.Ts(t=[0.0, 1.0]),
            "instantaneous_rate",
            {"min_spikes_in_burst": 1},
            pytest.raises(ValueError, match="`min_spikes_in_burst` must be >= 2"),
        ),
    ],
)
def test_detect_spike_bursts_errors(spikes, method, params, expectation):
    with expectation:
        nap.detect_spike_bursts(spikes, method=method, **params)


# ---------------------------------------------------------------------------
# Shared behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method", ["poisson_surprise", "max_interval", "instantaneous_rate"]
)
def test_tsgroup_returns_dict_per_unit(method):
    t = np.concatenate([[0.0, 0.1, 0.2, 0.3], [2.0, 2.05, 2.1], [5.0]])
    group = nap.TsGroup({0: nap.Ts(t=t), 3: nap.Ts(t=t + 0.5)})
    bursts = nap.detect_spike_bursts(group, method=method)
    assert isinstance(bursts, dict)
    assert list(bursts.keys()) == [0, 3]
    for k, iset in bursts.items():
        assert isinstance(iset, nap.IntervalSet)
        single = nap.detect_spike_bursts(group[k], method=method)
        np.testing.assert_array_equal(iset.start, single.start)
        np.testing.assert_array_equal(iset.end, single.end)
        np.testing.assert_array_equal(
            np.asarray(iset.n_spikes), np.asarray(single.n_spikes)
        )


@pytest.mark.parametrize(
    "method", ["poisson_surprise", "max_interval", "instantaneous_rate"]
)
def test_bursts_do_not_span_time_support_gaps(method):
    spikes = _regular_bursts(n_bursts=100, spikes_per_burst=6, intra=0.005)
    # cut the 50th burst in two: 3 spikes before the gap, 3 after
    cut = 49 * 1.5 + 0.0125
    epochs = nap.IntervalSet(start=[0.0, cut], end=[cut - 1e-6, 200.0])
    bursts = nap.detect_spike_bursts(spikes.restrict(epochs), method=method)
    assert np.all(bursts.end <= epochs.end.max())
    in_gap = (bursts.start < cut) & (bursts.end > cut)
    assert not np.any(in_gap)
    # the 3 spikes before the gap and the 3 spikes after it each form a burst
    assert len(bursts) == 101
    assert np.sum(np.asarray(bursts.n_spikes) == 3) == 2


@pytest.mark.parametrize(
    "method", ["poisson_surprise", "max_interval", "instantaneous_rate"]
)
@pytest.mark.parametrize("t", [[], [1.0], [1.0, 2.0, 3.0]])
def test_no_bursts(method, t):
    bursts = nap.detect_spike_bursts(nap.Ts(t=np.array(t)), method=method)
    assert isinstance(bursts, nap.IntervalSet)
    assert len(bursts) == 0
    assert "n_spikes" in bursts.metadata_columns


def test_identical_spike_times_are_skipped():
    spikes = nap.Ts(t=[0.0, 0.0, 0.0, 1.0, 1.001, 1.002])
    bursts = nap.detect_spike_bursts(spikes, method="instantaneous_rate")
    _assert_bursts(bursts, [1.0], [1.002], [3])


# ---------------------------------------------------------------------------
# MaxInterval
# ---------------------------------------------------------------------------


def test_max_interval_defaults():
    t = np.concatenate([[0.0, 0.1, 0.2, 0.3], [2.0, 2.05, 2.1], [5.0]])
    bursts = nap.detect_spike_bursts(nap.Ts(t=t), method="max_interval")
    _assert_bursts(bursts, [0.0, 2.0], [0.3, 2.1], [4, 3])


def test_max_interval_start_vs_within_threshold():
    # 0.2 is above max_isi_start but below max_isi_burst: cannot start a burst,
    # but continues one.
    t = np.array([0.0, 0.2, 0.4, 1.0, 1.1, 1.3, 1.5])
    bursts = nap.detect_spike_bursts(
        nap.Ts(t=t), method="max_interval", max_isi_start=0.15, max_isi_burst=0.25
    )
    _assert_bursts(bursts, [1.0], [1.5], [4])


def test_max_interval_merge():
    t = np.array([0.0, 0.05, 0.1, 0.5, 0.55, 0.6])
    kwargs = dict(max_isi_start=0.1, max_isi_burst=0.1, min_burst_duration=0.0)
    separate = nap.detect_spike_bursts(
        nap.Ts(t=t), method="max_interval", min_inter_burst_interval=0.3, **kwargs
    )
    _assert_bursts(separate, [0.0, 0.5], [0.1, 0.6], [3, 3])
    merged = nap.detect_spike_bursts(
        nap.Ts(t=t), method="max_interval", min_inter_burst_interval=0.5, **kwargs
    )
    _assert_bursts(merged, [0.0], [0.6], [6])


def test_max_interval_min_duration_and_spikes():
    t = np.array([0.0, 0.001, 0.002, 1.0, 1.1, 1.2, 1.3, 3.0, 3.1])
    bursts = nap.detect_spike_bursts(
        nap.Ts(t=t),
        method="max_interval",
        min_burst_duration=0.01,
        min_spikes_in_burst=3,
    )
    _assert_bursts(bursts, [1.0], [1.3], [4])
    bursts = nap.detect_spike_bursts(
        nap.Ts(t=t),
        method="max_interval",
        min_burst_duration=0.0,
        min_spikes_in_burst=2,
    )
    _assert_bursts(bursts, [0.0, 1.0, 3.0], [0.002, 1.3, 3.1], [3, 4, 2])


# ---------------------------------------------------------------------------
# Instantaneous rate
# ---------------------------------------------------------------------------


def test_instantaneous_rate():
    spikes = nap.Ts(t=[0.0, 0.005, 0.01, 0.1, 0.2, 0.202, 0.3])
    bursts = nap.detect_spike_bursts(spikes, method="instantaneous_rate")
    _assert_bursts(bursts, [0.0, 0.2], [0.01, 0.202], [3, 2])
    bursts = nap.detect_spike_bursts(
        spikes, method="instantaneous_rate", min_spikes_in_burst=3
    )
    _assert_bursts(bursts, [0.0], [0.01], [3])
    bursts = nap.detect_spike_bursts(
        spikes, method="instantaneous_rate", rate_threshold=9.0
    )
    _assert_bursts(bursts, [0.0], [0.3], [7])


# ---------------------------------------------------------------------------
# Poisson surprise
# ---------------------------------------------------------------------------

# fmt: off
_PS_TRAIN = np.array([
    0.268, 1.236, 2.159, 2.511, 3.531, 3.724, 4.034, 4.119, 4.243, 5, 5.01, 5.02,
    5.03, 5.04, 5.05, 5.06, 5.07, 5.31, 5.344, 6.807, 7.442, 7.601, 7.648, 7.682,
    7.722, 8.225, 9.642, 9.871, 9.954, 11.457, 11.991, 12, 12.02, 12.04, 12.06,
    12.08, 12.1, 12.582, 13.033, 13.216, 13.369, 13.74, 14.352, 14.474, 15.397,
    15.549, 15.885, 16.547, 17.394, 17.968, 18.164, 18.694, 18.894, 19.838,
])
# fmt: on


def test_poisson_surprise_matches_r():
    # PS.method() of Cotterill & Eglen (2018) on _PS_TRAIN gives spikes
    # (1-based) 10-17, 22-25 and 31-37, with surprise 20.33, 5.38 and 14.15.
    bursts = nap.detect_spike_bursts(nap.Ts(t=_PS_TRAIN), method="poisson_surprise")
    _assert_bursts(
        bursts,
        _PS_TRAIN[[9, 21, 30]],
        _PS_TRAIN[[16, 24, 36]],
        [8, 4, 7],
    )


def test_poisson_surprise_min_surprise():
    spikes = nap.Ts(t=_PS_TRAIN)
    bursts = nap.detect_spike_bursts(
        spikes, method="poisson_surprise", min_surprise=10
    )
    np.testing.assert_array_equal(np.asarray(bursts.n_spikes), [8, 7])
    bursts = nap.detect_spike_bursts(spikes, method="poisson_surprise", min_surprise=0)
    assert len(bursts) >= 3


# The next two tests cover the two places where the method departs from the R
# code. _PS_TRAIN does not exercise either, so test_poisson_surprise_matches_r
# passes with or without them.


def test_poisson_surprise_long_isi_elsewhere_does_not_stop_burst():
    # The burst (spikes 5 to 11) has a longer ISI in the middle, so the surprise
    # drops before it rises again. The R code stops extending it there, because
    # it reads the 3 s ISI between spikes 2 and 3 of the train.
    t = np.array(
        [0.3, 0.9, 1.5, 4.5, 5.2, 6.0, 6.05, 6.1, 6.45, 6.5, 6.55, 6.6, 7.8, 8.9, 10.0]
    )
    bursts = nap.detect_spike_bursts(nap.Ts(t=t), method="poisson_surprise")
    _assert_bursts(bursts, [6.0], [6.6], [7])


def test_poisson_surprise_max_isi_burst_is_a_hard_limit():
    # The spike at 4.05 increases the surprise, but is 30 ms after the previous
    # one. `max_isi_burst` takes precedence, so it is left out.
    t = np.array([0, 1, 2, 3, 4.0, 4.005, 4.01, 4.015, 4.02, 4.05, 5, 6, 7, 8])
    bursts = nap.detect_spike_bursts(nap.Ts(t=t), method="poisson_surprise")
    _assert_bursts(bursts, [4.0], [4.05], [6])
    bursts = nap.detect_spike_bursts(
        nap.Ts(t=t), method="poisson_surprise", max_isi_burst=0.02
    )
    _assert_bursts(bursts, [4.0], [4.02], [5])


@pytest.mark.parametrize(
    "n_spikes, duration, mean_isi",
    [
        (3, 0.02, 1.0),
        (8, 0.07, 0.37),
        (30, 0.5, 0.1),
        # more spikes expected than observed: the sum runs over the lower side
        (5, 3.0, 1.0),
        (4, 10.0, 1.0),
    ],
)
def test_poisson_surprise_value(n_spikes, duration, mean_isi):
    times = np.linspace(0.0, duration, n_spikes)
    surprise = _jitpoisson_surprise(times, 0, n_spikes, mean_isi)
    expected = -stats.poisson.logsf(n_spikes - 2, duration / mean_isi)
    np.testing.assert_allclose(surprise, expected, rtol=1e-10)


@pytest.mark.parametrize(
    "n_spikes, duration, mean_isi",
    [(50, 1e-7, 10.0), (200, 1e-9, 100.0), (1000, 1e-6, 1e3)],
)
def test_poisson_surprise_does_not_underflow(n_spikes, duration, mean_isi):
    # The probability is below the smallest float, so its log cannot be taken
    # directly. It is dominated by its first term.
    times = np.linspace(0.0, duration, n_spikes)
    surprise = _jitpoisson_surprise(times, 0, n_spikes, mean_isi)
    k, expected = n_spikes - 1, duration / mean_isi
    first_term = k * np.log(expected) - expected - special.gammaln(k + 1)
    assert first_term < np.log(np.finfo(float).tiny)
    assert np.isfinite(surprise)
    np.testing.assert_allclose(surprise, -first_term, rtol=1e-6)


def test_poisson_surprise_extreme_burst():
    # 100 spikes 1 ns apart in a train firing once every 100 s
    burst = 5050.0 + np.arange(100) * 1e-9
    t = np.sort(np.concatenate([np.arange(100) * 100.0, burst]))
    bursts = nap.detect_spike_bursts(nap.Ts(t=t), method="poisson_surprise")
    _assert_bursts(bursts, [burst[0]], [burst[-1]], [100])


@pytest.mark.parametrize(
    "params, expectation",
    [
        ({"min_surprise": -1}, pytest.raises(ValueError, match="`min_surprise`")),
        ({"min_surprise": "a"}, pytest.raises(TypeError, match="`min_surprise`")),
        ({"max_isi_burst": 0}, pytest.raises(ValueError, match="`max_isi_burst`")),
        ({"max_isi_burst": 0.5}, does_not_raise()),
        ({"max_isi_start": 0.1}, pytest.raises(TypeError, match="Unknown parameter")),
    ],
)
def test_poisson_surprise_errors(params, expectation):
    with expectation:
        nap.detect_spike_bursts(
            nap.Ts(t=_PS_TRAIN), method="poisson_surprise", **params
        )
