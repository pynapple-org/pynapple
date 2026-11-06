"""Tests of burst detection in pynapple.process.isi"""

import re
import warnings
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

import pynapple as nap


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


def test_default_method_is_log_isi():
    spikes = _regular_bursts()
    default = nap.detect_spike_bursts(spikes)
    log_isi = nap.detect_spike_bursts(spikes, method="log_isi")
    np.testing.assert_array_equal(default.start, log_isi.start)
    np.testing.assert_array_equal(default.end, log_isi.end)
    assert len(default) == 200
    assert np.all(np.asarray(default.n_spikes) == 5)


@pytest.mark.parametrize(
    "spikes, method, params, expectation",
    [
        (
            None,
            "log_isi",
            {},
            pytest.raises(
                TypeError,
                match="`spikes` must be `Ts` or `TsGroup`, got <class 'NoneType'>",
            ),
        ),
        (
            nap.Tsd(t=[0, 1], d=[0, 1]),
            "log_isi",
            {},
            pytest.raises(TypeError, match="`spikes` must be `Ts` or `TsGroup`"),
        ),
        (
            nap.Ts(t=[0.0, 1.0]),
            "poisson_surprise",
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
            "log_isi",
            {"max_isi_start": 0.1},
            pytest.raises(
                TypeError,
                match=re.escape(
                    "Unknown parameter(s) ['max_isi_start'] for method 'log_isi'"
                ),
            ),
        ),
        (
            nap.Ts(t=[0.0, 1.0]),
            "instantaneous_rate",
            {"fallback": False, "max_cutoff": 0.1},
            pytest.raises(
                TypeError,
                match=re.escape("Unknown parameter(s) ['fallback', 'max_cutoff']"),
            ),
        ),
        # log_isi parameters
        (
            nap.Ts(t=[0.0, 1.0]),
            "log_isi",
            {"max_cutoff": "a"},
            pytest.raises(TypeError, match="`max_cutoff` must be a number"),
        ),
        (
            nap.Ts(t=[0.0, 1.0]),
            "log_isi",
            {"max_cutoff": 0},
            pytest.raises(ValueError, match="`max_cutoff` must be > 0"),
        ),
        (
            nap.Ts(t=[0.0, 1.0]),
            "log_isi",
            {"min_spikes_in_burst": 2.5},
            pytest.raises(TypeError, match="`min_spikes_in_burst` must be an integer"),
        ),
        (
            nap.Ts(t=[0.0, 1.0]),
            "log_isi",
            {"min_spikes_in_burst": 1},
            pytest.raises(ValueError, match="`min_spikes_in_burst` must be >= 2"),
        ),
        (
            nap.Ts(t=[0.0, 1.0]),
            "log_isi",
            {"fallback": 1},
            pytest.raises(TypeError, match="`fallback` must be a bool"),
        ),
        # max_interval parameters
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


@pytest.mark.parametrize(
    "spikes, expectation",
    [
        (None, pytest.raises(TypeError, match="`spikes` must be `Ts` or `TsGroup`")),
        (nap.Ts(t=[0.0, 1.0]), does_not_raise()),
    ],
)
def test_compute_log_isi_threshold_spikes_type(spikes, expectation):
    with expectation:
        nap.compute_log_isi_threshold(spikes)


@pytest.mark.parametrize(
    "max_cutoff, expectation",
    [
        ("a", pytest.raises(TypeError, match="`max_cutoff` must be a number")),
        (0, pytest.raises(ValueError, match="`max_cutoff` must be > 0")),
        (0.1, does_not_raise()),
    ],
)
def test_compute_log_isi_threshold_max_cutoff(max_cutoff, expectation):
    with expectation:
        nap.compute_log_isi_threshold(nap.Ts(t=[0.0, 1.0]), max_cutoff=max_cutoff)


# ---------------------------------------------------------------------------
# Shared behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["log_isi", "max_interval", "instantaneous_rate"])
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


@pytest.mark.parametrize("method", ["log_isi", "max_interval", "instantaneous_rate"])
def test_bursts_do_not_span_time_support_gaps(method):
    spikes = _regular_bursts(n_bursts=100, spikes_per_burst=6, intra=0.005)
    # cut the 50th burst in two: 3 spikes before the gap, 3 after
    cut = 49 * 1.5 + 0.0125
    epochs = nap.IntervalSet(start=[0.0, cut], end=[cut - 1e-6, 200.0])
    bursts = nap.detect_spike_bursts(spikes.restrict(epochs), method=method)
    assert len(bursts) == 101
    assert np.all(bursts.end <= epochs.end.max())
    in_gap = (bursts.start < cut) & (bursts.end > cut)
    assert not np.any(in_gap)
    assert np.sum(np.asarray(bursts.n_spikes) == 3) == 2


@pytest.mark.parametrize("method", ["log_isi", "max_interval", "instantaneous_rate"])
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
# logISI
# ---------------------------------------------------------------------------


def test_log_isi_threshold_regular_bursts():
    spikes = _regular_bursts()
    threshold = nap.compute_log_isi_threshold(spikes)
    # valley is the first bin after the 10 ms peak on a 10 bins / decade axis
    assert threshold == pytest.approx(0.0106, abs=1e-4)
    group = nap.TsGroup({0: spikes, 1: spikes})
    thresholds = nap.compute_log_isi_threshold(group)
    assert isinstance(thresholds, dict)
    assert thresholds[0] == thresholds[1] == threshold


@pytest.mark.parametrize(
    "spikes",
    [
        nap.Ts(t=np.array([])),
        nap.Ts(t=np.array([1.0])),
        # all intervals under 1 ms
        nap.Ts(t=np.arange(100) * 0.0005),
        # no intra-burst peak: all intervals at 1 s
        nap.Ts(t=np.arange(100) * 1.0),
    ],
)
def test_log_isi_threshold_none_without_intraburst_peak(spikes):
    assert nap.compute_log_isi_threshold(spikes) is None
    # no intra-burst peak at all: no bursts, and nothing to warn about
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert len(nap.detect_spike_bursts(spikes)) == 0
        assert len(nap.detect_spike_bursts(spikes, fallback=True)) == 0


def _train_without_clear_minimum():
    # Intra-burst peak at 10 ms, then a ramp of inter-burst intervals whose
    # counts increase up to the last bin: no later peak, so no clear minimum.
    isis = [0.01] * 300 + [
        v for k, v in enumerate(np.logspace(-1.6, -0.1, 15)) for _ in range(k + 1)
    ]
    return nap.Ts(t=np.cumsum(np.random.default_rng(0).permutation(isis)))


def test_log_isi_no_minimum_returns_no_bursts_and_warns():
    spikes = _train_without_clear_minimum()
    assert nap.compute_log_isi_threshold(spikes) is None
    with pytest.warns(UserWarning, match="No clear minimum.*Set `fallback=True`"):
        bursts = nap.detect_spike_bursts(spikes)
    assert len(bursts) == 0
    # the published fallback is opt-in and does not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with_fallback = nap.detect_spike_bursts(spikes, fallback=True)
    assert len(with_fallback) > 0
    assert np.all(np.asarray(with_fallback.n_spikes) >= 3)


def test_log_isi_no_minimum_tsgroup_warns_once_with_units():
    group = nap.TsGroup(
        {
            2: _train_without_clear_minimum(),
            5: _regular_bursts(),
            7: _train_without_clear_minimum(),
        }
    )
    with pytest.warns(UserWarning, match=re.escape("unit(s) [2, 7]")) as record:
        bursts = nap.detect_spike_bursts(group)
    assert len(record) == 1
    assert len(bursts[2]) == 0 and len(bursts[7]) == 0 and len(bursts[5]) == 200


def test_log_isi_threshold_above_max_cutoff_extends_cores():
    # Each burst has a core of 5 spikes at 50 ms intervals followed by 8
    # "burst-related" spikes at intervals growing by one histogram bin each,
    # up to 330 ms, then a 5 s gap. The valley lies above max_cutoff (100 ms),
    # so the cores found with max_cutoff are extended to the tail spikes.
    tails = 0.05 * 1.266 ** np.arange(1, 9)
    burst = np.concatenate([[0.0], np.cumsum([0.05] * 4 + list(tails))])
    starts = np.arange(100) * 5.0
    spikes = nap.Ts(t=np.concatenate([s + burst for s in starts]))
    threshold = nap.compute_log_isi_threshold(spikes)
    assert threshold is not None and threshold > 0.1
    bursts = nap.detect_spike_bursts(spikes)
    assert len(bursts) == 100
    assert np.all(np.asarray(bursts.n_spikes) == 13)
    np.testing.assert_allclose(bursts.end - bursts.start, burst[-1])
    # with fallback=False nothing changes, since a valley was found
    bursts = nap.detect_spike_bursts(spikes, fallback=False)
    assert len(bursts) == 100


def test_log_isi_min_spikes():
    spikes = _regular_bursts(spikes_per_burst=3)
    assert len(nap.detect_spike_bursts(spikes, min_spikes_in_burst=3)) == 200
    assert len(nap.detect_spike_bursts(spikes, min_spikes_in_burst=4)) == 0


def test_log_isi_histogram_uses_compute_isi_distribution():
    # The log-ISI histogram is built with compute_isi_distribution on a
    # 10-bins-per-decade axis in seconds.
    spikes = _regular_bursts()
    isi_ms = spikes.time_diff().values * 1000
    decades = int(np.ceil(np.log10(isi_ms.max())))
    edges = np.logspace(0, decades, 10 * decades) / 1000
    hist = nap.compute_isi_distribution(spikes, bins=edges)
    assert hist[0].sum() == len(isi_ms)
    # two populated bins: 10 ms and 1.46 s
    assert np.sum(hist[0].values > 0) == 2
