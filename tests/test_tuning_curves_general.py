"""Tests of tuning curves for `pynapple` package."""

import itertools
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import scipy

import pynapple as nap


def get_group(
    n_units: int, duration: float = 100.0, mean_rate_hz: float = 5.0
) -> nap.TsGroup:
    units = {}
    for k in range(n_units):
        n_spikes = np.random.poisson(mean_rate_hz * duration)
        spike_times = np.random.uniform(0.0, duration, size=n_spikes)
        spike_times.sort()
        units[k] = nap.Ts(t=spike_times)

    return nap.TsGroup(units)


def get_features(num_dims: int, duration: float = 100.0, dt: float = 0.1):
    t = np.arange(0.0, duration, dt)
    # Saw‑tooth features, each phase‑shifted so they differ
    data = np.column_stack([(t + i / num_dims) % 1.0 for i in range(num_dims)])
    # Wrap in a TsdFrame with a matching time_support
    return nap.TsdFrame(t=t, d=data, time_support=nap.IntervalSet(0.0, duration))


@pytest.mark.parametrize(
    "group",
    [
        group.count(0.1) if continuous else group
        for continuous in [False, True]
        for n_units in range(1, 4)
        if (group := get_group(n_units))
    ],
)
@pytest.mark.parametrize(
    "features, bins",
    [
        (get_features(D), bins)
        for D in range(1, 4)
        for bins in (
            [2, 5, 10]
            + [list(tup) for tup in itertools.product([2, 5, 10], repeat=D) if D > 1]
        )
    ]
    + [
        (
            nap.Tsd(
                t=tsdframe.times(),
                d=tsdframe.values.flatten(),
                time_support=tsdframe.time_support,
            ),
            num_bins,
        )
        for num_bins in [2, 5, 10]
        if (tsdframe := get_features(num_dims=1))
    ],
)
@pytest.mark.parametrize("range_alpha", [None, 0.0, 0.5])
@pytest.mark.parametrize(
    "epochs",
    [
        None,
        nap.IntervalSet(0.0, 50.0),
        nap.IntervalSet(0.0, 100.0),
        nap.IntervalSet(0.0, 200.0),
        nap.IntervalSet([0.0, 40.0], [10.0, 90.0]),
    ],
)
def test_compute_tuning_curves(group, features, bins, range_alpha, epochs):
    if range_alpha is None:
        range = None
    else:
        full_min = np.nanmin(features.values, axis=0)
        full_max = np.nanmax(features.values, axis=0)
        span = full_max - full_min
        range = np.c_[full_min + range_alpha * span, full_max - range_alpha * span]

    # ------------------------------------------------------------------
    # compute actual
    # ------------------------------------------------------------------
    tcs, tc_bins = nap.compute_tuning_curves(
        group=group,
        features=features,
        bins=bins,
        range=range,
        epochs=epochs,
    )

    # ------------------------------------------------------------------
    # compute expected
    # ------------------------------------------------------------------
    if epochs is None:
        epochs = features.time_support
        group = group.restrict(epochs)
    else:
        features = features.restrict(epochs)
        group = group.restrict(epochs)

    if isinstance(features, nap.Tsd):
        features = nap.TsdFrame(
            d=features.values,
            t=features.times(),
            time_support=features.time_support,
        )

    # Occupancy
    occupancy, bin_edges = np.histogramdd(features.values, bins=bins, range=range)
    occupancy[occupancy == 0] = np.nan

    # Tuning curves
    expected_tcs = {}
    if isinstance(group, nap.TsGroup):
        for n in group.keys():
            count, _ = np.histogramdd(
                group[n].value_from(features, epochs).values,
                bins=bin_edges,
            )
            expected_tcs[n] = (count / occupancy) * features.rate
    else:
        _expected_tcs = scipy.stats.binned_statistic_dd(
            group.value_from(features, epochs).values,
            values=group.values.T,
            bins=bin_edges,
        ).statistic
        _expected_tcs[:, np.isnan(occupancy)] = np.nan
        for k, tc in zip(group.columns, _expected_tcs):
            expected_tcs[k] = tc

    # expected bin centres
    expected_tc_bins = [e[:-1] + np.diff(e) / 2 for e in bin_edges]

    # ------------------------------------------------------------------
    # test
    # ------------------------------------------------------------------
    assert isinstance(tcs, dict)
    assert len(tcs) == len(expected_tcs)
    for (key, tc), (expected_key, expected_tc) in zip(
        tcs.items(), expected_tcs.items()
    ):
        assert key == expected_key
        assert isinstance(tc, np.ndarray)
        assert tc.shape == expected_tc.shape
        np.testing.assert_allclose(tc, expected_tc)

    assert isinstance(tc_bins, list)
    assert len(tc_bins) == len(expected_tc_bins)
    for bins, expected_bins in zip(tc_bins, expected_tc_bins):
        assert isinstance(bins, np.ndarray)
        assert bins.shape == expected_bins.shape
        np.testing.assert_allclose(bins, expected_bins)
