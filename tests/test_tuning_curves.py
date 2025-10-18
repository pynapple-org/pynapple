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
            {"return_pandas": 2},
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
            {"return_pandas": 0},
            does_not_raise(),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"return_pandas": 1},
            does_not_raise(),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"return_pandas": True},
            does_not_raise(),
        ),
        (
            get_group_n(1),
            get_features_n(1),
            {"return_pandas": False},
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
# DISCRETE TUNING CURVE TESTS
# ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data, epochs_dict, kwargs, expectation",
    [
        # data
        (
            [1],
            {},
            {},
            pytest.raises(
                TypeError, match="data should be a TsdFrame, TsGroup, Ts, or Tsd."
            ),
        ),
        (
            None,
            {},
            {},
            pytest.raises(
                TypeError, match="data should be a TsdFrame, TsGroup, Ts, or Tsd."
            ),
        ),
        (
            {1: nap.Ts([1, 2, 3])},
            {},
            {},
            pytest.raises(
                TypeError, match="data should be a TsdFrame, TsGroup, Ts, or Tsd."
            ),
        ),
        (get_group_n(1), {}, {}, does_not_raise()),
        (get_group_n(3), {}, {}, does_not_raise()),
        (get_group_n(1).count(0.1), {}, {}, does_not_raise()),
        (get_group_n(3).count(0.1), {}, {}, does_not_raise()),
        (nap.Tsd(t=[1, 2, 3], d=[1, 1, 1]), {}, {}, does_not_raise()),
        (nap.Ts([1, 2, 3]), {}, {}, does_not_raise()),
        # epochs_dict
        (
            get_group_n(1),
            1,
            {},
            pytest.raises(
                TypeError, match="epochs_dict should be a dictionary of IntervalSets."
            ),
        ),
        (
            get_group_n(1),
            None,
            {},
            pytest.raises(
                TypeError, match="epochs_dict should be a dictionary of IntervalSets."
            ),
        ),
        (
            get_group_n(1),
            nap.IntervalSet(0, 100),
            {},
            pytest.raises(
                TypeError, match="epochs_dict should be a dictionary of IntervalSets."
            ),
        ),
        (
            get_group_n(1),
            [nap.IntervalSet(0, 100)],
            {},
            pytest.raises(
                TypeError, match="epochs_dict should be a dictionary of IntervalSets."
            ),
        ),
        (
            get_group_n(1),
            {"0": nap.IntervalSet(0, 100), "1": 0},
            {},
            pytest.raises(
                TypeError, match="epochs_dict should be a dictionary of IntervalSets."
            ),
        ),
        (
            get_group_n(1),
            {"0": nap.IntervalSet(0, 100)},
            {},
            does_not_raise(),
        ),
        (
            get_group_n(1),
            {"0": nap.IntervalSet(0, 100), "1": nap.IntervalSet(0, 50)},
            {},
            does_not_raise(),
        ),
        # return pandas
        (
            get_group_n(1),
            {},
            {"return_pandas": 2},
            pytest.raises(
                TypeError,
                match="return_pandas should be a boolean.",
            ),
        ),
        (
            get_group_n(1),
            {},
            {"return_pandas": "1"},
            pytest.raises(
                TypeError,
                match="return_pandas should be a boolean.",
            ),
        ),
        (
            get_group_n(1),
            {},
            {"return_pandas": True},
            does_not_raise(),
        ),
        (
            get_group_n(1),
            {},
            {"return_pandas": False},
            does_not_raise(),
        ),
        (
            get_group_n(1),
            {},
            {"return_pandas": 0},
            does_not_raise(),
        ),
        (
            get_group_n(1),
            {},
            {"return_pandas": 1},
            does_not_raise(),
        ),
    ],
)
def test_compute_discrete_tuning_curves_type_errors(
    data, epochs_dict, kwargs, expectation
):
    with expectation:
        nap.compute_discrete_tuning_curves(data, epochs_dict, **kwargs)


@pytest.mark.parametrize(
    "data, epochs_dict, kwargs, expectation",
    [
        # single rate unit, single epoch
        (
            get_group_n(1).count(1.0),
            {"0": nap.IntervalSet(0, 50)},
            {},
            xr.DataArray(
                [[10.0]],
                dims=["unit", "epochs"],
                coords={"unit": [1], "epochs": ["0"]},
            ),
        ),
        # two rate units, single epoch
        (
            get_group_n(2).count(1.0),
            {"0": nap.IntervalSet(0, 50)},
            {},
            xr.DataArray(
                [[10.0], [1.0]],
                dims=["unit", "epochs"],
                coords={"unit": [1, 2], "epochs": ["0"]},
            ),
        ),
        # two rate units, multiple epochs
        (
            get_group_n(2).count(1.0),
            {"0": nap.IntervalSet(0, 50), "1": nap.IntervalSet(50, 100)},
            {},
            xr.DataArray(
                [[10.0, 10.0], [1.0, 1.0]],
                dims=["unit", "epochs"],
                coords={"unit": [1, 2], "epochs": ["0", "1"]},
            ),
        ),
        # single unit, single epoch
        (
            get_group_n(1),
            {"0": nap.IntervalSet(50, 100)},
            {},
            xr.DataArray(
                [[10.0]],
                dims=["unit", "epochs"],
                coords={"unit": [1], "epochs": ["0"]},
            ),
        ),
        # two units, single epoch
        (
            get_group_n(2),
            {"0": nap.IntervalSet(0, 100)},
            {},
            xr.DataArray(
                [[10.0], [1.0]],
                dims=["unit", "epochs"],
                coords={"unit": [1, 2], "epochs": ["0"]},
            ),
        ),
        # two units, multiple epochs
        (
            get_group_n(2),
            {"0": nap.IntervalSet(0, 100), "1": nap.IntervalSet(50, 100)},
            {},
            xr.DataArray(
                [[10.0, 10.0], [1.0, 1.0]],
                dims=["unit", "epochs"],
                coords={"unit": [1, 2], "epochs": ["0", "1"]},
            ),
        ),
        # two units, multiple epochs, return_pandas=True
        (
            get_group_n(2),
            {"0": nap.IntervalSet(0, 100), "1": nap.IntervalSet(50, 100)},
            {"return_pandas": True},
            xr.DataArray(
                [[10.0, 10.0], [1.0, 1.0]],
                dims=["unit", "epochs"],
                coords={"unit": [1, 2], "epochs": ["0", "1"]},
            )
            .to_pandas()
            .T,
        ),
    ],
)
def test_compute_discrete_tuning_curves(data, epochs_dict, kwargs, expectation):
    tcs = nap.compute_discrete_tuning_curves(data, epochs_dict, **kwargs)
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
        (
            (lambda x: (x.attrs.clear(), x)[1])(get_testing_set()[0]),
            pytest.raises(
                ValueError,
                match="No occupancy found in tuning curves.",
            ),
        ),
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


# ------------------------------------------------------------------------------------
# OLD MUTUAL INFORMATION TESTS
# ------------------------------------------------------------------------------------


def get_group():
    return nap.TsGroup({0: nap.Ts(t=np.arange(0, 100))})


def get_feature():
    return nap.Tsd(
        t=np.arange(0, 100, 0.1),
        d=np.arange(0, 100, 0.1) % 1.0,
        time_support=nap.IntervalSet(0, 100),
    )


def get_features():
    tmp = np.vstack(
        (np.repeat(np.arange(0, 100), 10), np.tile(np.arange(0, 100), 10))
    ).T
    return nap.TsdFrame(
        t=np.arange(0, 200, 0.1),
        d=np.vstack((tmp, tmp[::-1])),
        time_support=nap.IntervalSet(0, 200),
    )


def get_ep():
    return nap.IntervalSet(start=0, end=50)


def get_tsdframe():
    return nap.TsdFrame(t=np.arange(0, 100), d=np.ones((100, 2)))


@pytest.mark.parametrize(
    "tc, feature, ep, minmax, bitssec, expected_exception",
    [
        (
            "a",
            get_feature(),
            get_ep(),
            (0, 1),
            True,
            "Argument tc should be of type pandas.DataFrame or numpy.ndarray",
        ),
        (
            pd.DataFrame(),
            "a",
            get_ep(),
            (0, 1),
            True,
            r"feature should be a Tsd \(or TsdFrame with 1 column only\)",
        ),
        (
            pd.DataFrame(),
            get_feature(),
            "a",
            (0, 1),
            True,
            r"ep should be an IntervalSet",
        ),
        (
            pd.DataFrame(),
            get_feature(),
            get_ep(),
            1,
            True,
            r"minmax should be a tuple\/list of 2 numbers",
        ),
        (
            pd.DataFrame(),
            get_feature(),
            get_ep(),
            (0, 1),
            "a",
            r"Argument bitssec should be of type bool",
        ),
    ],
)
def test_compute_1d_mutual_info_errors(
    tc, feature, ep, minmax, bitssec, expected_exception
):
    with pytest.raises(TypeError, match=expected_exception):
        nap.compute_1d_mutual_info(tc, feature, ep, minmax, bitssec)


@pytest.mark.parametrize(
    "dict_tc, features, ep, minmax, bitssec, expected_exception",
    [
        (
            "a",
            get_features(),
            get_ep(),
            (0, 1),
            True,
            "Argument dict_tc should be a dictionary of numpy.ndarray or numpy.ndarray",
        ),
        (
            {0: np.zeros((2, 2))},
            "a",
            get_ep(),
            (0, 1),
            True,
            r"features should be a TsdFrame with 2 columns",
        ),
        (
            {0: np.zeros((2, 2))},
            get_features(),
            "a",
            (0, 1),
            True,
            r"ep should be an IntervalSet",
        ),
        (
            {0: np.zeros((2, 2))},
            get_features(),
            get_ep(),
            1,
            True,
            r"minmax should be a tuple\/list of 2 numbers",
        ),
        (
            {0: np.zeros((2, 2))},
            get_features(),
            get_ep(),
            (0, 1),
            "a",
            r"Argument bitssec should be of type bool",
        ),
    ],
)
def test_compute_2d_mutual_info_errors(
    dict_tc, features, ep, minmax, bitssec, expected_exception
):
    with pytest.raises(TypeError, match=expected_exception):
        nap.compute_2d_mutual_info(dict_tc, features, ep, minmax, bitssec)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "args, kwargs, expected",
    [
        (
            (
                pd.DataFrame(index=np.arange(0, 2), data=np.array([0, 10])),
                nap.Tsd(t=np.arange(100), d=np.tile(np.arange(2), 50)),
            ),
            {},
            np.array([[1.0]]),
        ),
        (
            (
                pd.DataFrame(index=np.arange(0, 2), data=np.array([0, 10])),
                nap.Tsd(t=np.arange(100), d=np.tile(np.arange(2), 50)),
            ),
            {"bitssec": True},
            np.array([[5.0]]),
        ),
        (
            (
                pd.DataFrame(index=np.arange(0, 2), data=np.array([0, 10])),
                nap.Tsd(t=np.arange(100), d=np.tile(np.arange(2), 50)),
            ),
            {"ep": nap.IntervalSet(start=0, end=49)},
            np.array([[1.0]]),
        ),
        (
            (
                pd.DataFrame(index=np.arange(0, 2), data=np.array([0, 10])),
                nap.Tsd(t=np.arange(100), d=np.tile(np.arange(2), 50)),
            ),
            {"minmax": (0, 1)},
            np.array([[1.0]]),
        ),
        (
            (
                np.array([[0], [10]]),
                nap.Tsd(t=np.arange(100), d=np.tile(np.arange(2), 50)),
            ),
            {"minmax": (0, 1)},
            np.array([[1.0]]),
        ),
    ],
)
def test_compute_1d_mutual_info(args, kwargs, expected):
    tc = args[0]
    feature = args[1]
    si = nap.compute_1d_mutual_info(tc, feature, **kwargs)
    assert isinstance(si, pd.DataFrame)
    assert list(si.columns) == ["SI"]
    if isinstance(tc, pd.DataFrame):
        assert list(si.index.values) == list(tc.columns)
    np.testing.assert_approx_equal(si.values, expected)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "args, kwargs, expected",
    [
        (
            (
                {0: np.array([[0, 1], [0, 0]])},
                nap.TsdFrame(
                    t=np.arange(100),
                    d=np.tile(np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), 25).T,
                ),
            ),
            {},
            np.array([[2.0]]),
        ),
        (
            (
                np.array([[[0, 1], [0, 0]]]),
                nap.TsdFrame(
                    t=np.arange(100),
                    d=np.tile(np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), 25).T,
                ),
            ),
            {},
            np.array([[2.0]]),
        ),
        (
            (
                {0: np.array([[0, 1], [0, 0]])},
                nap.TsdFrame(
                    t=np.arange(100),
                    d=np.tile(np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), 25).T,
                ),
            ),
            {"bitssec": True},
            np.array([[0.5]]),
        ),
        (
            (
                {0: np.array([[0, 1], [0, 0]])},
                nap.TsdFrame(
                    t=np.arange(100),
                    d=np.tile(np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), 25).T,
                ),
            ),
            {"ep": nap.IntervalSet(start=0, end=7)},
            np.array([[2.0]]),
        ),
        (
            (
                {0: np.array([[0, 1], [0, 0]])},
                nap.TsdFrame(
                    t=np.arange(100),
                    d=np.tile(np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), 25).T,
                ),
            ),
            {"minmax": (0, 1, 0, 1)},
            np.array([[2.0]]),
        ),
    ],
)
def test_compute_2d_mutual_info(args, kwargs, expected):
    dict_tc = args[0]
    features = args[1]
    si = nap.compute_2d_mutual_info(dict_tc, features, **kwargs)
    assert isinstance(si, pd.DataFrame)
    assert list(si.columns) == ["SI"]
    if isinstance(dict_tc, dict):
        assert list(si.index.values) == list(dict_tc.keys())
    np.testing.assert_approx_equal(si.values, expected)


# ------------------------------------------------------------------------------------
# OLD TUNING CURVE TESTS
# ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "group, feature, nb_bins, ep, minmax, expected_exception",
    [
        ("a", get_feature(), 10, get_ep(), (0, 1), "group should be a TsGroup."),
        (
            get_group(),
            "a",
            10,
            get_ep(),
            (0, 1),
            r"feature should be a Tsd \(or TsdFrame with 1 column only\)",
        ),
        (
            get_group(),
            get_feature(),
            "a",
            get_ep(),
            (0, 1),
            r"nb_bins should be of type int \(or tuple with \(int, int\) for 2D tuning curves\).",
        ),
        (get_group(), get_feature(), 10, "a", (0, 1), r"ep should be an IntervalSet"),
        (
            get_group(),
            get_feature(),
            10,
            get_ep(),
            1,
            r"minmax should be a tuple\/list of 2 numbers",
        ),
    ],
)
def test_compute_1d_tuning_curves_errors(
    group, feature, nb_bins, ep, minmax, expected_exception
):
    with pytest.raises(TypeError, match=expected_exception):
        nap.compute_1d_tuning_curves(group, feature, nb_bins, ep, minmax)


@pytest.mark.parametrize(
    "group, features, nb_bins, ep, minmax, expected_exception",
    [
        ("a", get_features(), 10, get_ep(), (0, 1), "group should be a TsGroup."),
        (
            get_group(),
            "a",
            10,
            get_ep(),
            (0, 1),
            r"features should be a TsdFrame with 2 columns",
        ),
        (
            get_group(),
            get_features(),
            "a",
            get_ep(),
            (0, 1),
            r"nb_bins should be of type int \(or tuple with \(int, int\) for 2D tuning curves\).",
        ),
        (get_group(), get_features(), 10, "a", (0, 1), r"ep should be an IntervalSet"),
        (
            get_group(),
            get_features(),
            10,
            get_ep(),
            1,
            r"minmax should be a tuple\/list of 2 numbers",
        ),
    ],
)
def test_compute_2d_tuning_curves_errors(
    group, features, nb_bins, ep, minmax, expected_exception
):
    with pytest.raises(TypeError, match=expected_exception):
        nap.compute_2d_tuning_curves(group, features, nb_bins, ep, minmax)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "args, kwargs, expectation",
    [
        ((get_group(), get_feature(), 10), {}, np.array([10.0] + [0.0] * 9)[:, None]),
        (
            (get_group(), get_feature(), 10),
            {"ep": get_ep()},
            np.array([10.0] + [0.0] * 9)[:, None],
        ),
        (
            (get_group(), get_feature(), 10),
            {"minmax": (0, 0.9)},
            np.array([10.0] + [0.0] * 9)[:, None],
        ),
        (
            (get_group(), get_feature(), 20),
            {"minmax": (0, 1.9)},
            np.array([10.0] + [0.0] * 9 + [np.nan] * 10)[:, None],
        ),
    ],
)
def test_compute_1d_tuning_curves(args, kwargs, expectation):
    tc = nap.compute_1d_tuning_curves(*args, **kwargs)
    # Columns
    assert list(tc.columns) == list(args[0].keys())

    # Index
    assert len(tc) == args[2]
    if "minmax" in kwargs:
        tmp = np.linspace(kwargs["minmax"][0], kwargs["minmax"][1], args[2] + 1)
    else:
        tmp = np.linspace(np.min(args[1]), np.max(args[1]), args[2] + 1)
    np.testing.assert_almost_equal(tmp[0:-1] + np.diff(tmp) / 2, tc.index.values)

    # Array
    np.testing.assert_almost_equal(tc.values, expectation)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "args, kwargs, expectation",
    [
        ((get_group(), get_features(), 10), {}, np.ones((10, 10)) * 0.5),
        ((get_group(), get_features(), (10, 10)), {}, np.ones((10, 10)) * 0.5),
        (
            (get_group(), get_features(), 10),
            {"ep": nap.IntervalSet(0, 400)},
            np.ones((10, 10)) * 0.5,
        ),
        (
            (get_group(), get_features(), 10),
            {"minmax": (0, 100, 0, 100)},
            np.ones((10, 10)) * 0.5,
        ),
        (
            (get_group(), get_features(), 10),
            {"minmax": (0, 200, 0, 100)},
            np.vstack((np.ones((5, 10)) * 0.5, np.ones((5, 10)) * np.nan)),
        ),
    ],
)
def test_compute_2d_tuning_curves(args, kwargs, expectation):
    tc, xy = nap.compute_2d_tuning_curves(*args, **kwargs)
    assert isinstance(tc, dict)

    # Keys
    assert list(tc.keys()) == list(args[0].keys())

    # Index
    assert isinstance(xy, list)
    assert len(xy) == 2
    nb_bins = args[2]
    if isinstance(args[2], int):
        nb_bins = (args[2], args[2])
    if "minmax" in kwargs:
        tmp1 = np.linspace(kwargs["minmax"][0], kwargs["minmax"][1], nb_bins[0] + 1)
        tmp2 = np.linspace(kwargs["minmax"][2], kwargs["minmax"][3], nb_bins[1] + 1)
    else:
        tmp1 = np.linspace(np.min(args[1][:, 0]), np.max(args[1][:, 0]), nb_bins[0] + 1)
        tmp2 = np.linspace(np.min(args[1][:, 1]), np.max(args[1][:, 1]), nb_bins[1] + 1)

    np.testing.assert_almost_equal(tmp1[0:-1] + np.diff(tmp1) / 2, xy[0])
    np.testing.assert_almost_equal(tmp2[0:-1] + np.diff(tmp2) / 2, xy[1])

    # Values
    for i in tc.keys():
        assert tc[i].shape == nb_bins
        np.testing.assert_almost_equal(tc[i], expectation)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "args, kwargs, expectation",
    [
        (
            (get_tsdframe(), get_feature(), 10),
            {},
            np.vstack((np.ones((1, 2)), np.zeros((9, 2)))),
        ),
        (
            (get_tsdframe(), get_feature()[:, np.newaxis], 10),
            {},
            np.vstack((np.ones((1, 2)), np.zeros((9, 2)))),
        ),
        (
            (get_tsdframe()[:, 0], get_feature(), 10),
            {},
            np.vstack((np.ones((1, 1)), np.zeros((9, 1)))),
        ),
        (
            (get_tsdframe(), get_feature(), 10),
            {"ep": get_ep()},
            np.vstack((np.ones((1, 2)), np.zeros((9, 2)))),
        ),
        (
            (get_tsdframe(), get_feature(), 10),
            {"minmax": (0, 0.9)},
            np.vstack((np.ones((1, 2)), np.zeros((9, 2)))),
        ),
        (
            (get_tsdframe(), get_feature(), 20),
            {"minmax": (0, 1.9)},
            np.vstack((np.ones((1, 2)), np.zeros((9, 2)), np.ones((10, 2)) * np.nan)),
        ),
    ],
)
def test_compute_1d_tuning_curves_continuous(args, kwargs, expectation):
    tsdframe, feature, nb_bins = args
    tc = nap.compute_1d_tuning_curves_continuous(tsdframe, feature, nb_bins, **kwargs)
    # Columns
    if hasattr(tsdframe, "columns"):
        assert list(tc.columns) == list(tsdframe.columns)
    # Index
    assert len(tc) == nb_bins
    if "minmax" in kwargs:
        tmp = np.linspace(kwargs["minmax"][0], kwargs["minmax"][1], nb_bins + 1)
    else:
        tmp = np.linspace(np.min(feature), np.max(feature), nb_bins + 1)
    np.testing.assert_almost_equal(tmp[0:-1] + np.diff(tmp) / 2, tc.index.values)
    # Array
    np.testing.assert_almost_equal(tc.values, expectation)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "tsdframe, nb_bins, kwargs, expectation",
    [
        (
            nap.TsdFrame(
                t=np.arange(0, 100),
                d=np.hstack((np.ones((100, 1)), np.ones((100, 1)) * 2)),
            ),
            1,
            {},
            {0: np.array([[1.0]]), 1: np.array([[2.0]])},
        ),
        (
            nap.TsdFrame(
                t=np.arange(0, 100),
                d=np.hstack((np.ones((100, 1)), np.ones((100, 1)) * 2)),
                columns=["x", "y"],
            ),
            2,
            {},
            {"x": np.ones((2, 2)), "y": np.ones((2, 2)) * 2},
        ),
        (
            nap.Tsd(t=np.arange(0, 100), d=np.hstack((np.ones((100,)) * 2))),
            2,
            {},
            {0: np.ones((2, 2)) * 2},
        ),
        (
            nap.TsdFrame(
                t=np.arange(0, 100),
                d=np.hstack((np.ones((100, 1)), np.ones((100, 1)) * 2)),
            ),
            (1, 2),
            {},
            {0: np.array([[1.0, 1.0]]), 1: np.array([[2.0, 2.0]])},
        ),
        (
            nap.TsdFrame(
                t=np.arange(0, 100),
                d=np.hstack((np.ones((100, 1)), np.ones((100, 1)) * 2)),
            ),
            1,
            {"ep": get_ep()},
            {0: np.array([[1.0]]), 1: np.array([[2.0]])},
        ),
        (
            nap.TsdFrame(
                t=np.arange(0, 100),
                d=np.hstack((np.ones((100, 1)), np.ones((100, 1)) * 2)),
            ),
            1,
            {"minmax": (0, 1, 0, 1)},
            {0: np.array([[1.0]]), 1: np.array([[2.0]])},
        ),
        (
            nap.TsdFrame(
                t=np.arange(0, 100),
                d=np.hstack((np.ones((100, 1)), np.ones((100, 1)) * 2)),
            ),
            (1, 3),
            {"minmax": (0, 1, 0, 3)},
            {0: np.array([[1.0, 1.0, np.nan]]), 1: np.array([[2.0, 2.0, np.nan]])},
        ),
    ],
)
def test_compute_2d_tuning_curves_continuous(tsdframe, nb_bins, kwargs, expectation):
    features = nap.TsdFrame(
        t=np.arange(100), d=np.tile(np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), 25).T
    )
    tc, xy = nap.compute_2d_tuning_curves_continuous(
        tsdframe, features, nb_bins, **kwargs
    )

    # Keys
    if hasattr(tsdframe, "columns"):
        assert list(tc.keys()) == list(tsdframe.columns)

    # Index
    assert isinstance(xy, list)
    assert len(xy) == 2
    if isinstance(nb_bins, int):
        nb_bins = (nb_bins, nb_bins)
    if "minmax" in kwargs:
        tmp1 = np.linspace(kwargs["minmax"][0], kwargs["minmax"][1], nb_bins[0] + 1)
        tmp2 = np.linspace(kwargs["minmax"][2], kwargs["minmax"][3], nb_bins[1] + 1)
    else:
        tmp1 = np.linspace(
            np.min(features[:, 0]), np.max(features[:, 0]), nb_bins[0] + 1
        )
        tmp2 = np.linspace(
            np.min(features[:, 1]), np.max(features[:, 1]), nb_bins[1] + 1
        )

    np.testing.assert_almost_equal(tmp1[0:-1] + np.diff(tmp1) / 2, xy[0])
    np.testing.assert_almost_equal(tmp2[0:-1] + np.diff(tmp2) / 2, xy[1])

    # Values
    for i in tc.keys():
        assert tc[i].shape == nb_bins
        np.testing.assert_almost_equal(tc[i], expectation[i])
