"""Tests of tuning curves for `pynapple` package."""

import itertools
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

import pynapple as nap


def get_group(
    n_units: int = 2, duration: float = 100.0, mean_rate_hz: float = 5.0
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
    "num_dims, num_bins",
    [
        (num_dims, num_bins)
        for num_dims in range(1, 4)
        for num_bins in (
            [1, 5, 10]
            + [
                list(tup)
                for tup in itertools.product([1, 5, 10], repeat=num_dims)
                if num_dims > 1
            ]
        )
    ],
)
@pytest.mark.parametrize("bounds_alpha", [None, 0.0, 0.2])
@pytest.mark.parametrize(
    "epoch",
    [
        None,
        # nap.IntervalSet(0.0, 50.0),
        # nap.IntervalSet(0.0, 100.0),
        # nap.IntervalSet(0.0, 200.0),
        # nap.IntervalSet([0.0, 40.0], [10.0, 90.0]),
    ],
)
@pytest.mark.parametrize("continuous", [True])
def test_compute_tuning_curves(continuous, num_dims, num_bins, bounds_alpha, epoch):
    _group = get_group()
    group = _group.count(0.1) if continuous else _group
    features = get_features(num_dims)

    if bounds_alpha is None:
        bounds = None
    else:
        full_min = np.nanmin(features.values, axis=0)
        full_max = np.nanmax(features.values, axis=0)
        span = full_max - full_min
        bounds = np.vstack(
            [full_min + bounds_alpha * span, full_max - bounds_alpha * span]
        )  # shape (2, num_dims)

    # ------------------------------------------------------------------
    # compute actual
    # ------------------------------------------------------------------
    tcs, tc_bins = nap.compute_tuning_curves(
        group=group,
        features=features,
        num_bins=num_bins,
        bounds=bounds,
        epoch=epoch,
    )

    # ------------------------------------------------------------------
    # compute expected
    # ------------------------------------------------------------------
    _features = features if epoch is None else features.restrict(epoch)
    _num_bins = [num_bins] * num_dims if isinstance(num_bins, int) else num_bins

    # build edges identical to what the function *should* have used
    if bounds is None:
        occupancy, bin_edges = np.histogramdd(_features.values, bins=_num_bins)
    else:
        bin_edges = [
            np.linspace(low, high, n + 1)
            for low, high, n in zip(bounds[0], bounds[1], _num_bins, strict=True)
        ]
        occupancy, _ = np.histogramdd(_features.values, bins=bin_edges)
    occupancy[occupancy == 0] = np.nan  # avoid /0

    # tuning curves
    expected_tcs = {}
    group_vals = {d: _group.value_from(_features[:, d], epoch) for d in range(num_dims)}
    for k in _group.keys():
        spike_feat = np.column_stack(
            [group_vals[d][k].values.flatten() for d in range(num_dims)]
        )
        counts, _ = np.histogramdd(spike_feat, bins=bin_edges)
        expected_tcs[k] = (counts / occupancy) * _features.rate

    # expected bin centres
    expected_tc_bins = [e[:-1] + np.diff(e) / 2 for e in bin_edges]

    # ------------------------------------------------------------------
    # test
    # ------------------------------------------------------------------
    assert isinstance(tcs, dict)
    assert len(tcs) == len(expected_tcs) == len(_group)
    for (key, tc), (expected_key, expected_tc) in zip(
        tcs.items(), expected_tcs.items()
    ):
        assert key == expected_key
        assert isinstance(tc, np.ndarray)
        assert tc.ndim == num_dims
        assert tc.shape == tuple(_num_bins)
        np.testing.assert_almost_equal(tc, expected_tc)

    assert isinstance(tc_bins, list)
    assert len(tc_bins) == len(expected_tc_bins) == num_dims
    for bins, expected_bins, expected_size in zip(tc_bins, expected_tc_bins, _num_bins):
        assert isinstance(bins, np.ndarray)
        assert bins.ndim == 1
        assert bins.size == expected_size
        np.testing.assert_allclose(bins, expected_bins)
