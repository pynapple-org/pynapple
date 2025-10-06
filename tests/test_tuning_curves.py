"""Tests of N-dimensional tuning curves for `pynapple` package."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pynapple as nap


def get_group_n(n):
    return nap.TsGroup(
        {i + 1: nap.Ts(t=np.arange(0, 100, 10 ** (i - 1))) for i in range(n)}
    )


def get_features_n(n, fs=10.0):
    return nap.TsdFrame(
        t=np.arange(0, 100, 1 / fs),
        d=np.stack(
            [np.arange(0, 100, 1 / fs) % 10 * i for i in range(1, n + 1)], axis=1
        ),
        columns=[f"feature{i}" for i in range(n)],
    )


@pytest.mark.parametrize(
    "data, features, kwargs, expectation",
    [
        # data
        (
            [1],
            get_features_n(1),
            {},
            pytest.raises(
                TypeError, match="data should be a TsdFrame, TsGroup, Ts, or Tsd."
            ),
        ),
        (
            None,
            get_features_n(1),
            {},
            pytest.raises(
                TypeError, match="data should be a TsdFrame, TsGroup, Ts, or Tsd."
            ),
        ),
        (
            {1: nap.Ts([1, 2, 3])},
            get_features_n(1),
            {},
            pytest.raises(
                TypeError, match="data should be a TsdFrame, TsGroup, Ts, or Tsd."
            ),
        ),
        (get_group_n(1), get_features_n(1), {}, does_not_raise()),
        (get_group_n(3), get_features_n(1), {}, does_not_raise()),
        (get_group_n(1).count(0.1), get_features_n(1), {}, does_not_raise()),
        (get_group_n(3).count(0.1), get_features_n(1), {}, does_not_raise()),
        (nap.Tsd(t=[1, 2, 3], d=[1, 1, 1]), get_features_n(1), {}, does_not_raise()),
        (nap.Ts([1, 2, 3]), get_features_n(1), {}, does_not_raise()),
        # features
        (
            get_group_n(1),
            [1],
            {},
            pytest.raises(TypeError, match="features should be a Tsd or TsdFrame"),
        ),
        (
            get_group_n(1),
            None,
            {},
            pytest.raises(TypeError, match="features should be a Tsd or TsdFrame"),
        ),
        (
            get_group_n(1),
            nap.Tsd(d=[1, 1, 1], t=[1, 2, 3]),
            {},
            does_not_raise(),
        ),
        (
            get_group_n(1),
            get_features_n(3),
            {},
            does_not_raise(),
        ),
        # epochs
        (
            get_group_n(1),
            get_features_n(1),
            {"epochs": 1},
            pytest.raises(TypeError, match="epochs should be an IntervalSet."),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"epochs": [1, 2]},
            pytest.raises(TypeError, match="epochs should be an IntervalSet."),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"epochs": None},
            does_not_raise(),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"epochs": nap.IntervalSet(0.0, 50.0)},
            does_not_raise(),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"epochs": nap.IntervalSet([0.0, 30.0], [10.0, 50.0])},
            does_not_raise(),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"epochs": nap.IntervalSet([0.0, 1000.0])},
            does_not_raise(),
        ),
        # range
        (
            get_group_n(1),
            get_features_n(2),
            {"range": (0, 1)},
            pytest.raises(
                ValueError,
                match="range should be a sequence of tuples, one for each feature.",
            ),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"range": (0, 1)},
            does_not_raise(),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"range": [(0, 1)]},
            does_not_raise(),
        ),
        # fs
        (
            get_group_n(1),
            get_features_n(1),
            {"fs": "1"},
            pytest.raises(TypeError, match="fs should be a number"),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"fs": []},
            pytest.raises(TypeError, match="fs should be a number"),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"fs": 1},
            does_not_raise(),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"fs": 1.0},
            does_not_raise(),
        ),
        # feature names
        (
            get_group_n(1),
            get_features_n(1),
            {"feature_names": "feature0"},
            pytest.raises(
                TypeError,
                match="feature_names should be a list of strings.",
            ),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"feature_names": 0},
            pytest.raises(
                TypeError,
                match="feature_names should be a list of strings.",
            ),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"feature_names": [1]},
            pytest.raises(
                TypeError,
                match="feature_names should be a list of strings.",
            ),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"feature_names": [(1,)]},
            pytest.raises(
                TypeError,
                match="feature_names should be a list of strings.",
            ),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"feature_names": [(1, 1)]},
            pytest.raises(
                TypeError,
                match="feature_names should be a list of strings.",
            ),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"feature_names": ["feature0", "feature1"]},
            pytest.raises(
                ValueError, match="feature_names should match the number of features."
            ),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"feature_names": ["feature0"]},
            does_not_raise(),
        ),
        (
            get_group_n(1),
            get_features_n(2),
            {"feature_names": ["feature0", "feature1"]},
            does_not_raise(),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"feature_names": np.array(["feature0"])},
            does_not_raise(),
        ),
        (
            get_group_n(1),
            get_features_n(2),
            {"feature_names": np.array(["feature0", "feature1"])},
            does_not_raise(),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"feature_names": ("feature0",)},
            does_not_raise(),
        ),
        (
            get_group_n(1),
            get_features_n(2),
            {"feature_names": ("feature0", "feature1")},
            does_not_raise(),
        ),
        # return pandas
        (
            get_group_n(1),
            get_features_n(1),
            {"return_pandas": 1},
            pytest.raises(
                TypeError,
                match="return_pandas should be a boolean.",
            ),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"return_pandas": "1"},
            pytest.raises(
                TypeError,
                match="return_pandas should be a boolean.",
            ),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"return_pandas": True},
            does_not_raise(),
        ),
        (
            get_group_n(1),
            get_features_n(2),
            {"return_pandas": True},
            pytest.raises(
                ValueError,
                match="Cannot convert arrays with 3 dimensions into pandas objects. Requires 2 or fewer dimensions.",
            ),
        ),
    ],
)
def test_compute_tuning_curves_type_errors(data, features, kwargs, expectation):
    with expectation:
        nap.compute_tuning_curves(data, features, **kwargs)


@pytest.mark.parametrize(
    "data, features, kwargs, expectation",
    [
        # single rate unit, single feature
        (
            get_group_n(1).count(1.0),
            get_features_n(1),
            {},
            xr.DataArray(
                np.full((1, 10), 10.0),
                dims=["unit", "feature0"],
                coords={"unit": [1], "feature0": np.linspace(0, 9.9, 11)[:-1] + 0.495},
            ),
        ),
        # multiple rate units, single feature
        (
            get_group_n(2).count(1.0),
            get_features_n(1),
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
            get_group_n(2).count(1.0),
            get_features_n(2),
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
            get_group_n(1),
            get_features_n(1),
            {},
            xr.DataArray(
                np.full((1, 10), 10.0),
                dims=["unit", "feature0"],
                coords={"unit": [1], "feature0": np.linspace(0, 9.9, 11)[:-1] + 0.495},
            ),
        ),
        # multiple units, single feature
        (
            get_group_n(2),
            get_features_n(1),
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
            get_group_n(2),
            get_features_n(2),
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
            get_group_n(1),
            get_features_n(1),
            {"bins": 5},
            xr.DataArray(
                np.full((1, 5), 10.0),
                dims=["unit", "feature0"],
                coords={"unit": [1], "feature0": np.linspace(0, 9.9, 6)[:-1] + 0.99},
            ),
        ),
        # single unit, multiple features, specified number of bins
        (
            get_group_n(1),
            get_features_n(2),
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
            get_group_n(1),
            get_features_n(2),
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
            get_group_n(1),
            get_features_n(1),
            {"bins": [np.linspace(0, 10, 6)]},
            xr.DataArray(
                np.full((1, 5), 10.0),
                dims=["unit", "feature0"],
                coords={"unit": [1], "feature0": np.arange(1, 11, 2)},
            ),
        ),
        # single unit, multiple features, specified bins
        (
            get_group_n(1),
            get_features_n(2),
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
            get_group_n(1),
            get_features_n(1),
            {"range": [(0, 5)]},
            xr.DataArray(
                np.full((1, 10), 10.0),
                dims=["unit", "feature0"],
                coords={"unit": [1], "feature0": np.linspace(0, 5.0, 11)[:-1] + 0.25},
            ),
        ),
        # single unit, multiple features, specified range per feature
        (
            get_group_n(1),
            get_features_n(2),
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
            get_group_n(1),
            get_features_n(1),
            {"bins": 10, "range": [(0, 5)]},
            xr.DataArray(
                np.full((1, 10), 10.0),
                dims=["unit", "feature0"],
                coords={"unit": [1], "feature0": np.linspace(0, 5.0, 11)[:-1] + 0.25},
            ),
        ),
        # single unit, multiple features, specified range per feature and number of bins
        (
            get_group_n(1),
            get_features_n(2),
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
            get_group_n(1),
            get_features_n(2),
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
            get_group_n(1),
            get_features_n(1),
            {"epochs": nap.IntervalSet([0.0, 50.0])},
            xr.DataArray(
                np.full((1, 10), 10.0),
                dims=["unit", "feature0"],
                coords={"unit": [1], "feature0": np.linspace(0, 9.9, 11)[:-1] + 0.495},
            ),
        ),
        # single unit, single feature, specified epochs (larger)
        (
            get_group_n(1),
            get_features_n(1),
            {"epochs": nap.IntervalSet([0.0, 200.0])},
            xr.DataArray(
                np.full((1, 10), 10.0),
                dims=["unit", "feature0"],
                coords={"unit": [1], "feature0": np.linspace(0, 9.9, 11)[:-1] + 0.495},
            ),
        ),
        # single unit, single feature, specified epochs (multiple)
        (
            get_group_n(1),
            get_features_n(1),
            {"epochs": nap.IntervalSet([0.0, 50.0], [20.0, 70.0])},
            xr.DataArray(
                np.full((1, 10), 10.0),
                dims=["unit", "feature0"],
                coords={"unit": [1], "feature0": np.linspace(0, 9.9, 11)[:-1] + 0.495},
            ),
        ),
        # single unit, single feature, specified feature name
        (
            get_group_n(1),
            get_features_n(1),
            {"feature_names": ["f0"]},
            xr.DataArray(
                np.full((1, 10), 10.0),
                dims=["unit", "f0"],
                coords={"unit": [1], "f0": np.linspace(0, 9.9, 11)[:-1] + 0.495},
            ),
        ),
        # single unit, multiple features, specified feature names
        (
            get_group_n(1),
            get_features_n(2),
            {"feature_names": ["f0", "f1"]},
            xr.DataArray(
                np.where(np.eye(10), 10.0, np.nan)[None, :],
                dims=["unit", "f0", "f1"],
                coords={
                    "unit": [1],
                    "f0": np.linspace(0, 9.9, 11)[:-1] + 0.495,
                    "f1": np.linspace(0, 19.8, 11)[:-1] + 0.99,
                },
            ),
        ),
        # single unit, single feature, return_pandas=True
        (
            get_group_n(1),
            get_features_n(1),
            {"return_pandas": True},
            xr.DataArray(
                np.full((1, 10), 10.0),
                dims=["unit", "feature0"],
                coords={"unit": [1], "feature0": np.linspace(0, 9.9, 11)[:-1] + 0.495},
            )
            .to_pandas()
            .T,
        ),
    ],
)
def test_compute_tuning_curves(data, features, kwargs, expectation):
    tcs = nap.compute_tuning_curves(data, features, **kwargs)
    if isinstance(expectation, pd.DataFrame):
        pd.testing.assert_frame_equal(tcs, expectation)
    else:
        xr.testing.assert_allclose(tcs, expectation)


# ------------------------------------------------------------------------------------
# MUTUAL INFORMATION TESTS
# ------------------------------------------------------------------------------------


def get_testing_set(n_units=1, n_features=1, pattern="uniform"):
    dims = ["unit"] + [f"dim_{i}" for i in range(n_features)]
    coords = {"unit": np.arange(n_units)}
    shape = (n_units,) + (2,) * n_features  # 2 bins per feature, for simplicity
    for i in range(n_features):
        coords[f"dim_{i}"] = np.arange(2)

    # Build tuning curves
    data = np.zeros(shape)

    if pattern == "uniform":
        data[:] = 1.0
        expected_mi_per_sec = 0.0
        expected_mi_per_spike = 0.0

    elif pattern == "onehot":
        # Each unit fires in a unique location only
        for u in range(n_units):
            index = [u] + [0] * n_features
            data[tuple(index)] = 1.0

        n_bins = np.prod(shape[1:])
        expected_mi_per_spike = np.log2(n_bins)
        mean_rate = 1.0 / n_bins
        expected_mi_per_sec = mean_rate * expected_mi_per_spike

    else:
        raise ValueError("Unknown firing_pattern. Use 'uniform' or 'onehot'.")

    tuning_curves = xr.DataArray(
        data,
        coords=coords,
        dims=dims,
        attrs={"occupancy": np.ones(shape[1:]) / np.prod(shape[1:])},
    )

    MI = pd.DataFrame(
        data=np.stack(
            [
                np.full(n_units, expected_mi_per_sec),
                np.full(n_units, expected_mi_per_spike),
            ],
            axis=1,
        ),
        index=coords["unit"],
        columns=["bits/sec", "bits/spike"],
    )

    return tuning_curves, MI


@pytest.mark.parametrize(
    "tuning_curves, expectation",
    [
        (
            [],
            pytest.raises(
                TypeError,
                match="tuning_curves should be an xr.DataArray as computed by compute_tuning_curves.",
            ),
        ),
        (
            1,
            pytest.raises(
                TypeError,
                match="tuning_curves should be an xr.DataArray as computed by compute_tuning_curves.",
            ),
        ),
        (
            get_testing_set()[0].to_pandas().T,
            pytest.raises(
                TypeError,
                match="tuning_curves should be an xr.DataArray as computed by compute_tuning_curves.",
            ),
        ),
        (get_testing_set(1, 1)[0], does_not_raise()),
        (get_testing_set(1, 2)[0], does_not_raise()),
        (get_testing_set(1, 3)[0], does_not_raise()),
        (get_testing_set(2, 1)[0], does_not_raise()),
        (get_testing_set(2, 2)[0], does_not_raise()),
        (get_testing_set(2, 3)[0], does_not_raise()),
    ],
)
def test_compute_mutual_information_errors(tuning_curves, expectation):
    with expectation:
        nap.compute_mutual_information(tuning_curves)


@pytest.mark.parametrize(
    "n_units, n_features",
    [(1, 1), (1, 2), (1, 3)],
)
@pytest.mark.parametrize(
    "pattern",
    ["uniform", "onehot"],
)
def test_compute_mutual_information(n_units, n_features, pattern):
    tuning_curves, expectation = get_testing_set(n_units, n_features, pattern)
    actual = nap.compute_mutual_information(tuning_curves)
    pd.testing.assert_frame_equal(actual, expectation)
