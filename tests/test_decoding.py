"""Tests of decoding for `pynapple` package."""

from contextlib import nullcontext as does_not_raise
from itertools import product

import numpy as np
import pytest

import pynapple as nap


def get_testing_set_n(n_features=1, binned=False):
    combos = np.array(list(product([0, 1], repeat=n_features)))  # (2^F, F)
    reps = 5
    feature_data = np.tile(combos, (reps, 1))  # (T, F)
    times = np.arange(len(feature_data))

    features = nap.TsdFrame(t=times, d=feature_data)
    epochs = nap.IntervalSet(start=0, end=len(times))

    data = nap.TsGroup(
        {
            i: nap.Ts(t=times[np.all(feature_data == combo, axis=1)])
            for i, combo in enumerate(combos)
        }
    )

    if binned:
        frame = data.count(bin_size=1, ep=epochs)
        data = nap.TsdFrame(
            frame.times() - 0.5,
            frame.values,
            time_support=epochs,
        )

    tuning_curves = nap.compute_tuning_curves(
        data, features, bins=2, range=[(-0.5, 1.5)] * n_features
    )

    return {
        "features": features,
        "tuning_curves": tuning_curves,
        "data": data,
        "epochs": epochs,
        "bin_size": 1,
    }


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "overwrite_default_args, expectation",
    [
        # tuning_curves
        (
            {"tuning_curves": []},
            pytest.raises(
                TypeError,
                match="tuning_curves should be an xr.DataArray as computed by compute_tuning_curves.",
            ),
        ),
        (
            {"tuning_curves": 1},
            pytest.raises(
                TypeError,
                match="tuning_curves should be an xr.DataArray as computed by compute_tuning_curves.",
            ),
        ),
        (
            {"tuning_curves": get_testing_set_n()["tuning_curves"].to_pandas().T},
            pytest.raises(
                TypeError,
                match="tuning_curves should be an xr.DataArray as computed by compute_tuning_curves.",
            ),
        ),
        (
            {"tuning_curves": get_testing_set_n(2)["tuning_curves"]},
            pytest.raises(
                ValueError,
                match="Different shapes for tuning_curves and data.",
            ),
        ),
        (
            {"tuning_curves": get_testing_set_n(2, binned=True)["tuning_curves"]},
            pytest.raises(
                ValueError,
                match="Different shapes for tuning_curves and data.",
            ),
        ),
        (
            {
                "tuning_curves": get_testing_set_n()["tuning_curves"].assign_coords(
                    unit=[2, 3]
                )
            },
            pytest.raises(
                ValueError,
                match="Different indices for tuning curves and data keys.",
            ),
        ),
        (
            {
                "tuning_curves": get_testing_set_n(binned=True)[
                    "tuning_curves"
                ].assign_coords(unit=[2, 3])
            },
            pytest.raises(
                ValueError,
                match="Different indices for tuning curves and data keys.",
            ),
        ),
        ({}, does_not_raise()),
        (get_testing_set_n(1), does_not_raise()),
        (get_testing_set_n(2), does_not_raise()),
        # data
        (
            {"data": []},
            pytest.raises(
                TypeError,
                match="Unknown format for data.",
            ),
        ),
        (
            {"data": 1},
            pytest.raises(
                TypeError,
                match="Unknown format for data.",
            ),
        ),
        (
            {"data": get_testing_set_n(2)["data"]},
            pytest.raises(
                ValueError,
                match="Different shapes for tuning_curves and data.",
            ),
        ),
        (
            {
                "data": nap.TsGroup(
                    {2: nap.Ts(t=np.arange(0, 50)), 3: nap.Ts(t=np.arange(0, 50))}
                )
            },
            pytest.raises(
                ValueError,
                match="Different indices for tuning curves and data keys.",
            ),
        ),
        (
            {"data": get_testing_set_n(binned=True)["data"]},
            does_not_raise(),
        ),
        (
            get_testing_set_n(2, binned=True),
            does_not_raise(),
        ),
        (
            get_testing_set_n(3, binned=True),
            does_not_raise(),
        ),
    ],
)
def test_decode_input_errors(overwrite_default_args, expectation):
    default_args = get_testing_set_n()
    default_args.update(overwrite_default_args)
    default_args.pop("features")
    with expectation:
        nap.decode_bayes(**default_args)
        nap.decode_template(**default_args)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "overwrite_default_args, expectation",
    [
        # uniform_prior
        (
            {
                "uniform_prior": False,
                "tuning_curves": (lambda x: (x.attrs.clear(), x)[1])(
                    get_testing_set_n()["tuning_curves"]
                ),
            },
            pytest.raises(
                ValueError,
                match="uniform_prior set to False but no occupancy found in tuning curves.",
            ),
        ),
        (
            {"uniform_prior": True},
            does_not_raise(),
        ),
    ],
)
def test_decode_bayes_input_errors(overwrite_default_args, expectation):
    default_args = get_testing_set_n()
    default_args.update(overwrite_default_args)
    default_args.pop("features")
    with expectation:
        nap.decode_bayes(**default_args)


@pytest.mark.parametrize("uniform_prior", [True, False])
@pytest.mark.parametrize("n_features", [1, 2, 3])
@pytest.mark.parametrize("binned", [True, False])
def test_decode_bayes(n_features, binned, uniform_prior):
    features, tuning_curves, data, epochs, bin_size = get_testing_set_n(
        n_features, binned=binned
    ).values()
    decoded, proba = nap.decode_bayes(
        tuning_curves=tuning_curves,
        data=data,
        epochs=epochs,
        bin_size=bin_size,
        time_units="s",
        uniform_prior=uniform_prior,
    )

    assert isinstance(decoded, nap.Tsd if features.shape[1] == 1 else nap.TsdFrame)
    np.testing.assert_array_almost_equal(decoded.values, features.values.squeeze())

    assert isinstance(
        proba,
        nap.TsdFrame if features.shape[1] == 1 else nap.TsdTensor,
    )
    expected_proba = np.zeros((len(features), *tuning_curves.shape[1:]))
    target_indices = [np.arange(len(features))] + [
        features[:, d] for d in range(features.shape[1])
    ]
    expected_proba[tuple(target_indices)] = 1.0
    np.testing.assert_array_almost_equal(proba.values, expected_proba)


@pytest.mark.parametrize("metric", ["correlation", "euclidean", "cosine"])
@pytest.mark.parametrize("n_features", [1, 2, 3])
@pytest.mark.parametrize("binned", [True, False])
def test_decode_template(n_features, binned, metric):
    features, tuning_curves, data, epochs, bin_size = get_testing_set_n(
        n_features, binned=binned
    ).values()
    decoded, dist = nap.decode_template(
        tuning_curves=tuning_curves,
        data=data,
        epochs=epochs,
        metric=metric,
        bin_size=bin_size,
        time_units="s",
    )

    assert isinstance(decoded, nap.Tsd if features.shape[1] == 1 else nap.TsdFrame)
    np.testing.assert_array_almost_equal(decoded.values, features.values.squeeze())

    assert isinstance(
        dist,
        nap.TsdFrame if features.shape[1] == 1 else nap.TsdTensor,
    )
