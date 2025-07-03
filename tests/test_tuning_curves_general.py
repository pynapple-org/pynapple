"""Tests of tuning curves for `pynapple` package."""

import itertools
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import scipy
import xarray as xr

import pynapple as nap


def get_group(
    num_units: int, duration: float = 100.0, mean_rate_hz: float = 5.0
) -> nap.TsGroup:
    units = {}
    for k in range(num_units):
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
    return nap.TsdFrame(
        t=t,
        d=data,
        time_support=nap.IntervalSet(0.0, duration),
        columns=[f"col{i}" for i in range(num_dims)],
    )


@pytest.mark.parametrize(
    "group, features, kwargs, expectation",
    [
        # group
        (
            [1],
            get_features(1),
            {},
            pytest.raises(
                TypeError, match="group should be a Tsd, TsdFrame, TsGroup, or dict."
            ),
        ),
        (
            None,
            get_features(1),
            {},
            pytest.raises(
                TypeError, match="group should be a Tsd, TsdFrame, TsGroup, or dict."
            ),
        ),
        (get_group(1), get_features(1), {}, does_not_raise()),
        (get_group(3), get_features(1), {}, does_not_raise()),
        (get_group(1).count(0.1), get_features(1), {}, does_not_raise()),
        (get_group(3).count(0.1), get_features(1), {}, does_not_raise()),
        (nap.Tsd(t=[1, 2, 3], d=[1, 1, 1]), get_features(1), {}, does_not_raise()),
        ({1: nap.Ts([1, 2, 3])}, get_features(1), {}, does_not_raise()),
        (
            {1: nap.Ts([1, 2, 3]), 2: nap.Ts([1, 2, 3])},
            get_features(1),
            {},
            does_not_raise(),
        ),
        # features
        (
            get_group(1),
            [1],
            {},
            pytest.raises(TypeError, match="features should be a Tsd or TsdFrame"),
        ),
        (
            get_group(1),
            None,
            {},
            pytest.raises(TypeError, match="features should be a Tsd or TsdFrame"),
        ),
        (
            get_group(1),
            nap.Tsd(d=[1, 1, 1], t=[1, 2, 3]),
            {},
            does_not_raise(),
        ),
        (
            get_group(1),
            get_features(3),
            {},
            does_not_raise(),
        ),
        # epochs
        (
            get_group(1),
            get_features(1),
            {"epochs": 1},
            pytest.raises(TypeError, match="epochs should be an IntervalSet."),
        ),
        (
            get_group(1),
            get_features(1),
            {"epochs": [1, 2]},
            pytest.raises(TypeError, match="epochs should be an IntervalSet."),
        ),
        (
            get_group(1),
            get_features(1),
            {"epochs": None},
            does_not_raise(),
        ),
        (
            get_group(1),
            get_features(1),
            {"epochs": nap.IntervalSet(0.0, 50.0)},
            does_not_raise(),
        ),
        (
            get_group(1),
            get_features(1),
            {"epochs": nap.IntervalSet([0.0, 30.0], [10.0, 50.0])},
            does_not_raise(),
        ),
        (
            get_group(1),
            get_features(1),
            {"epochs": nap.IntervalSet([0.0, 1000.0])},
            pytest.warns(
                UserWarning,
                match="The passed epochs are larger than the time support of the features,"
                "this will artificially increase the outer bins of the tuning curves.",
            ),
        ),
        # fs
        (
            get_group(1),
            get_features(1),
            {"fs": "1"},
            pytest.raises(TypeError, match="fs should be a number"),
        ),
        (
            get_group(1),
            get_features(1),
            {"fs": []},
            pytest.raises(TypeError, match="fs should be a number"),
        ),
        (
            get_group(1),
            get_features(1),
            {"fs": 1},
            does_not_raise(),
        ),
        (
            get_group(1),
            get_features(1),
            {"fs": 1.0},
            does_not_raise(),
        ),
    ],
)
def test_compute_tuning_curves_type_errors(group, features, kwargs, expectation):
    with expectation:
        nap.compute_tuning_curves(group, features, **kwargs)


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
    tcs = nap.compute_tuning_curves(
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
    else:
        features = features.restrict(epochs)
    group = group.restrict(epochs)

    if isinstance(features, nap.Tsd):
        features = nap.TsdFrame(
            d=features.values,
            t=features.times(),
            time_support=features.time_support,
            columns=["f0"],
        )

    fs = 10.0
    occupancy, bin_edges = np.histogramdd(features, bins=bins, range=range)
    if isinstance(group, nap.TsGroup):
        occupancy[occupancy == 0] = np.nan
        expected_tcs = np.zeros([len(group), *occupancy.shape])
        for i, n in enumerate(group):
            count, _ = np.histogramdd(
                group[n].value_from(features, epochs).values,
                bins=bin_edges,
            )
            expected_tcs[i] = (count / occupancy) * fs
    else:
        expected_tcs = scipy.stats.binned_statistic_dd(
            group.value_from(features, epochs).values,
            values=group.values.T,
            bins=bin_edges,
        ).statistic
        expected_tcs[:, np.isnan(occupancy)] = np.nan

    # expected bin centres
    expected_tc_bins = [e[:-1] + np.diff(e) / 2 for e in bin_edges]

    # ------------------------------------------------------------------
    # test
    # ------------------------------------------------------------------

    # values
    assert isinstance(tcs, xr.DataArray)
    np.testing.assert_allclose(tcs, expected_tcs)

    # labels
    units = group.keys() if isinstance(group, nap.TsGroup) else group.columns
    assert "unit" in tcs.coords
    assert np.all(tcs.coords["unit"] == units)
    for dim, (dim_label, bins) in enumerate(list(tcs.coords.items())[1:]):
        assert dim_label == features.columns[dim]
        np.testing.assert_allclose(bins, expected_tc_bins[dim])
