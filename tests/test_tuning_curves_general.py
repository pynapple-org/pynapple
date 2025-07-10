"""Tests of tuning curves for `pynapple` package."""

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
        columns=[f"feature{i}" for i in range(n)],
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
            does_not_raise(),
        ),
        # range
        (
            get_group(1),
            get_features(2),
            {"range": (0, 1)},
            pytest.raises(
                ValueError,
                match="range should be a sequence of tuples, one for each feature.",
            ),
        ),
        (
            get_group(1),
            get_features(1),
            {"range": (0, 1)},
            does_not_raise(),
        ),
        (
            get_group(1),
            get_features(1),
            {"range": [(0, 1)]},
            does_not_raise(),
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
        # single rate unit, single feature
        (
            get_group(1).count(1.0),
            get_features(1),
            {},
            xr.DataArray(
                np.full((1, 10), 10.0),
                dims=["unit", "feature0"],
                coords={"unit": [1], "feature0": np.linspace(0, 9.9, 11)[:-1] + 0.495},
            ),
        ),
        # multiple rate units, single feature
        (
            get_group(2).count(1.0),
            get_features(1),
            {},
            xr.DataArray(
                np.concatenate([np.full((1, 10), 10.0), np.full((1, 10), 1.0)]),
                dims=["unit", "feature0"],
                coords={
                    "unit": [1, 2],
                    "feature0": np.linspace(0, 9.9, 11)[:-1] + 0.495,
                },
            ),
        ),
        # multiple rate units, multiple features
        (
            get_group(2).count(1.0),
            get_features(2),
            {},
            xr.DataArray(
                np.stack(
                    [
                        np.where(np.eye(10), 10.0, np.nan),
                        np.where(np.eye(10), 1.0, np.nan),
                    ]
                ),
                dims=["unit", "feature0", "feature1"],
                coords={
                    "unit": [1, 2],
                    "feature0": np.linspace(0, 9.9, 11)[:-1] + 0.495,
                    "feature1": np.linspace(0, 19.8, 11)[:-1] + 0.99,
                },
            ),
        ),
        # single unit, single feature
        (
            get_group(1),
            get_features(1),
            {},
            xr.DataArray(
                np.full((1, 10), 10.0),
                dims=["unit", "feature0"],
                coords={"unit": [1], "feature0": np.linspace(0, 9.9, 11)[:-1] + 0.495},
            ),
        ),
        # multiple units, single feature
        (
            get_group(2),
            get_features(1),
            {},
            xr.DataArray(
                np.concatenate([np.full((1, 10), 10.0), np.full((1, 10), 1.0)]),
                dims=["unit", "feature0"],
                coords={
                    "unit": [1, 2],
                    "feature0": np.linspace(0, 9.9, 11)[:-1] + 0.495,
                },
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
                dims=["unit", "feature0", "feature1"],
                coords={
                    "unit": [1, 2],
                    "feature0": np.linspace(0, 9.9, 11)[:-1] + 0.495,
                    "feature1": np.linspace(0, 19.8, 11)[:-1] + 0.99,
                },
            ),
        ),
        # single unit, single feature, specified number of bins
        (
            get_group(1),
            get_features(1),
            {"bins": 5},
            xr.DataArray(
                np.full((1, 5), 10.0),
                dims=["unit", "feature0"],
                coords={"unit": [1], "feature0": np.linspace(0, 9.9, 6)[:-1] + 0.99},
            ),
        ),
        # single unit, multiple features, specified number of bins
        (
            get_group(1),
            get_features(2),
            {"bins": 5},
            xr.DataArray(
                np.where(np.eye(5), 10.0, np.nan)[None, :],
                dims=["unit", "feature0", "feature1"],
                coords={
                    "unit": [1],
                    "feature0": np.linspace(0, 9.9, 6)[:-1] + 0.99,
                    "feature1": np.linspace(0, 19.8, 6)[:-1] + 1.98,
                },
            ),
        ),
        # single unit, multiple features, specified number of bins per feature
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
                dims=["unit", "feature0", "feature1"],
                coords={
                    "unit": [1],
                    "feature0": np.linspace(0, 9.9, 6)[:-1] + 0.99,
                    "feature1": np.linspace(0, 19.8, 5)[:-1] + 2.475,
                },
            ),
        ),
        # single unit, single feature, specified bins
        (
            get_group(1),
            get_features(1),
            {"bins": [np.linspace(0, 10, 6)]},
            xr.DataArray(
                np.full((1, 5), 10.0),
                dims=["unit", "feature0"],
                coords={"unit": [1], "feature0": np.arange(1, 11, 2)},
            ),
        ),
        # single unit, multiple features, specified bins
        (
            get_group(1),
            get_features(2),
            {"bins": [np.linspace(0, 10, 6), np.linspace(0, 20, 6)]},
            xr.DataArray(
                np.where(np.eye(5), 10.0, np.nan)[None, :],
                dims=["unit", "feature0", "feature1"],
                coords={
                    "unit": [1],
                    "feature0": np.arange(1, 11, 2),
                    "feature1": np.arange(2, 22, 4),
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
                dims=["unit", "feature0"],
                coords={"unit": [1], "feature0": np.linspace(0, 5.0, 11)[:-1] + 0.25},
            ),
        ),
        # single unit, multiple features, specified range per feature
        (
            get_group(1),
            get_features(2),
            {"range": [(0, 5), (0, 10)]},
            xr.DataArray(
                np.where(np.eye(10), 10.0, np.nan)[None, :],
                dims=["unit", "feature0", "feature1"],
                coords={
                    "unit": [1],
                    "feature0": np.linspace(0, 5.0, 11)[:-1] + 0.25,
                    "feature1": np.linspace(0, 10.0, 11)[:-1] + 0.5,
                },
            ),
        ),
        # single unit, single feature, specified range and number of bins
        (
            get_group(1),
            get_features(1),
            {"bins": 10, "range": [(0, 5)]},
            xr.DataArray(
                np.full((1, 10), 10.0),
                dims=["unit", "feature0"],
                coords={"unit": [1], "feature0": np.linspace(0, 5.0, 11)[:-1] + 0.25},
            ),
        ),
        # single unit, multiple features, specified range per feature and number of bins
        (
            get_group(1),
            get_features(2),
            {"bins": 10, "range": [(0, 5), (0, 10)]},
            xr.DataArray(
                np.where(np.eye(10), 10.0, np.nan)[None, :],
                dims=["unit", "feature0", "feature1"],
                coords={
                    "unit": [1],
                    "feature0": np.linspace(0, 5.0, 11)[:-1] + 0.25,
                    "feature1": np.linspace(0, 10.0, 11)[:-1] + 0.5,
                },
            ),
        ),
        # single unit, multiple features, specified range and number of bins per feature
        (
            get_group(1),
            get_features(2),
            {"bins": (10, 10), "range": [(0, 5), (0, 10)]},
            xr.DataArray(
                np.where(np.eye(10), 10.0, np.nan)[None, :],
                dims=["unit", "feature0", "feature1"],
                coords={
                    "unit": [1],
                    "feature0": np.linspace(0, 5.0, 11)[:-1] + 0.25,
                    "feature1": np.linspace(0, 10.0, 11)[:-1] + 0.5,
                },
            ),
        ),
        # single unit, single feature, specified epochs (smaller)
        (
            get_group(1),
            get_features(1),
            {"epochs": nap.IntervalSet([0.0, 50.0])},
            xr.DataArray(
                np.full((1, 10), 10.0),
                dims=["unit", "feature0"],
                coords={"unit": [1], "feature0": np.linspace(0, 9.9, 11)[:-1] + 0.495},
            ),
        ),
        # single unit, single feature, specified epochs (larger)
        (
            get_group(1),
            get_features(1),
            {"epochs": nap.IntervalSet([0.0, 200.0])},
            xr.DataArray(
                np.full((1, 10), 10.0),
                dims=["unit", "feature0"],
                coords={"unit": [1], "feature0": np.linspace(0, 9.9, 11)[:-1] + 0.495},
            ),
        ),
        # single unit, single feature, specified epochs (multiple)
        (
            get_group(1),
            get_features(1),
            {"epochs": nap.IntervalSet([0.0, 50.0], [20.0, 70.0])},
            xr.DataArray(
                np.full((1, 10), 10.0),
                dims=["unit", "feature0"],
                coords={"unit": [1], "feature0": np.linspace(0, 9.9, 11)[:-1] + 0.495},
            ),
        ),
    ],
)
def test_compute_tuning_curves(group, features, kwargs, expected):
    xr.testing.assert_allclose(
        nap.compute_tuning_curves(group, features, **kwargs), expected
    )
