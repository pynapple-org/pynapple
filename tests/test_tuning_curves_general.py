"""Tests of tuning curves for `pynapple` package."""

import itertools
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr

import pynapple as nap


def get_group(n):
    return nap.TsGroup(
        {i + 1: nap.Ts(t=np.arange(0, 100, 10 ** (i - 1))) for i in range(n)}
    )


def get_features(n, fs=10.0):
    return nap.TsdFrame(
        t=np.arange(0, 100, 1 / fs),
        d=np.stack(
            [np.arange(0, 100, 1 / fs) % 10 * i for i in range(1, n + 1)], axis=1
        ),
        columns=[f"f{i}" for i in range(n)],
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
    "group, features, kwargs, expected",
    [
        # single unit, single feature
        (
            get_group(1),
            get_features(1),
            {},
            xr.DataArray(
                np.full((1, 10), 10.0),
                dims=["unit", "f0"],
                coords={"unit": [1], "f0": np.linspace(0, 9.9, 11)[:-1] + 0.495},
            ),
        ),
        # multiple units, single feature
        (
            get_group(2),
            get_features(1),
            {},
            xr.DataArray(
                np.concatenate([np.full((1, 10), 10.0), np.full((1, 10), 1.0)]),
                dims=["unit", "f0"],
                coords={"unit": [1, 2], "f0": np.linspace(0, 9.9, 11)[:-1] + 0.495},
            ),
        ),
        # multiple units, multiple features
        (
            get_group(2),
            get_features(2),
            {},
            xr.DataArray(
                np.stack(
                    [
                        np.where(np.eye(10), 10.0, np.nan),
                        np.where(np.eye(10), 1.0, np.nan),
                    ]
                ),
                dims=["unit", "f0", "f1"],
                coords={
                    "unit": [1, 2],
                    "f0": np.linspace(0, 9.9, 11)[:-1] + 0.495,
                    "f1": np.linspace(0, 19.8, 11)[:-1] + 0.99,
                },
            ),
        ),
        # single unit, single feature, specified bins
        (
            get_group(1),
            get_features(1),
            {"bins": 5},
            xr.DataArray(
                np.full((1, 5), 10.0),
                dims=["unit", "f0"],
                coords={"unit": [1], "f0": np.linspace(0, 9.9, 6)[:-1] + 0.99},
            ),
        ),
        # single unit, multiple features, specified bins
        (
            get_group(1),
            get_features(2),
            {"bins": (5, 4)},
            xr.DataArray(
                np.array(
                    [
                        [
                            [10.0, np.nan, np.nan, np.nan],
                            [10.0, 10.0, np.nan, np.nan],
                            [np.nan, 10.0, 10.0, np.nan],
                            [np.nan, np.nan, 10.0, 10.0],
                            [np.nan, np.nan, np.nan, 10.0],
                        ]
                    ]
                ),
                dims=["unit", "f0", "f1"],
                coords={
                    "unit": [1],
                    "f0": np.linspace(0, 9.9, 6)[:-1] + 0.99,
                    "f1": np.linspace(0, 19.8, 5)[:-1] + 2.475,
                },
            ),
        ),
        # single unit, single feature, specified range
        (
            get_group(1),
            get_features(1),
            {"range": [(0, 5)]},
            xr.DataArray(
                np.full((1, 10), 10.0),
                dims=["unit", "f0"],
                coords={"unit": [1], "f0": np.linspace(0, 5.0, 11)[:-1] + 0.25},
            ),
        ),
    ],
)
def test_compute_tuning_curves(group, features, kwargs, expected):
    tcs = nap.compute_tuning_curves(group, features, **kwargs)
    assert isinstance(tcs, xr.DataArray)
    for dim in expected.coords.keys():
        assert dim in tcs.coords
        np.testing.assert_allclose(tcs.coords[dim].values, expected.coords[dim].values)
    np.testing.assert_allclose(tcs, expected)
