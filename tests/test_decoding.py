"""Tests of decoding for `pynapple` package."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

import pynapple as nap


def get_testing_set_n(n_units=1, n_features=1, binned=False):
    features = nap.TsdFrame(
        t=np.arange(0, 100, 1),
        d=np.stack([np.repeat(np.arange(0, 2), 50) for _ in range(n_features)], axis=1),
    )
    group = nap.TsGroup(
        {i: nap.Ts(t=np.arange(0, 50) + 50 * i) for i in range(n_units)}
    )
    if binned:
        group = group.count(bin_size=1)
    tc = nap.compute_tuning_curves(
        group=group, features=features, bins=2, range=[(-0.5, 1.5)] * n_features
    )
    epochs = nap.IntervalSet(start=0, end=100)
    return {"tuning_curves": tc, "group": group, "epochs": epochs, "bin_size": 1}


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "overwrite_default_args, expectation",
    [
        # tuning_curves
        (
            {"tuning_curves": []},
            pytest.raises(
                TypeError,
                match="tuning_curves should be an xr.DataArray as outputed by compute_tuning_curves",
            ),
        ),
        (
            {"tuning_curves": 1},
            pytest.raises(
                TypeError,
                match="tuning_curves should be an xr.DataArray as outputed by compute_tuning_curves",
            ),
        ),
        (
            {"tuning_curves": get_testing_set_n()["tuning_curves"].to_pandas().T},
            pytest.raises(
                TypeError,
                match="tuning_curves should be an xr.DataArray as outputed by compute_tuning_curves",
            ),
        ),
        (
            {"tuning_curves": get_testing_set_n(n_units=2)["tuning_curves"]},
            pytest.raises(
                ValueError,
                match="Different shapes for tuning_curves and group",
            ),
        ),
        (
            {
                "tuning_curves": get_testing_set_n(n_units=1)[
                    "tuning_curves"
                ].assign_coords(unit=[3])
            },
            pytest.raises(
                ValueError,
                match="Different indices for tuning curves and group keys",
            ),
        ),
        ({}, does_not_raise()),
        (get_testing_set_n(1, 2), does_not_raise()),
        (get_testing_set_n(1, 3), does_not_raise()),
        (get_testing_set_n(2, 1), does_not_raise()),
        (get_testing_set_n(2, 2), does_not_raise()),
        (get_testing_set_n(2, 3), does_not_raise()),
        (get_testing_set_n(3, 1), does_not_raise()),
        (get_testing_set_n(3, 2), does_not_raise()),
        (get_testing_set_n(3, 3), does_not_raise()),
        # group
        (
            {"group": []},
            pytest.raises(
                TypeError,
                match="Unknown format for group.",
            ),
        ),
        (
            {"group": 1},
            pytest.raises(
                TypeError,
                match="Unknown format for group.",
            ),
        ),
        (
            {"group": get_testing_set_n(2)["group"]},
            pytest.raises(
                ValueError,
                match="Different shapes for tuning_curves and group",
            ),
        ),
        (
            {"group": nap.TsGroup({2: nap.Ts(t=np.arange(0, 50))})},
            pytest.raises(
                ValueError,
                match="Different indices for tuning curves and group keys",
            ),
        ),
        (
            {"group": get_testing_set_n(binned=True)["group"]},
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
        # use_occupancy
        (
            {
                "use_occupancy": True,
                "tuning_curves": (lambda x: (x.attrs.clear(), x)[1])(
                    get_testing_set_n()["tuning_curves"]
                ),
            },
            pytest.raises(
                ValueError,
                match="use_occupancy set to True but no occupancy found in tuning curves",
            ),
        ),
        (
            {
                "use_occupancy": True,
                "tuning_curves": get_testing_set_n(1)["tuning_curves"].assign_attrs(
                    {"occupancy": np.array([1, 2, 3])}
                ),
            },
            pytest.raises(
                ValueError,
                match="Occupancy shape does not match tuning curves shape.",
            ),
        ),
        (
            {"use_occupancy": True},
            does_not_raise(),
        ),
    ],
)
def test_decode_type_errors(overwrite_default_args, expectation):
    default_args = get_testing_set_n(1)
    default_args.update(overwrite_default_args)
    with expectation:
        nap.decode(**default_args)


# def test_decode_1d():
#    feature, group, tc, epochs = get_testing_set_n(1)
#    decoded, proba = nap.decode(tc, group, epochs, bin_size=1)
#    assert isinstance(decoded, nap.Tsd)
#    assert isinstance(proba, nap.TsdFrame)
#    np.testing.assert_array_almost_equal(feature.values, decoded.values)
#    assert len(decoded) == 100
#    assert len(proba) == 100
#    tmp = np.ones((100, 2))
#    tmp[50:, 0] = 0.0
#    tmp[0:50, 1] = 0.0
#    np.testing.assert_array_almost_equal(proba.values, tmp)
#
#
# def test_decode_1d_with_TsdFrame():
#    feature, group, tc, epochs = get_testing_set_n(1)
#    count = group.count(bin_size=1, ep=epochs)
#    decoded, proba = nap.decode(tc, count, epochs, bin_size=1)
#    assert isinstance(decoded, nap.Tsd)
#    assert isinstance(proba, nap.TsdFrame)
#    np.testing.assert_array_almost_equal(feature.values, decoded.values)
#    assert len(decoded) == 100
#    assert len(proba) == 100
#    tmp = np.ones((100, 2))
#    tmp[50:, 0] = 0.0
#    tmp[0:50, 1] = 0.0
#    np.testing.assert_array_almost_equal(proba.values, tmp)
#
#
# def test_decode_1d_with_occupancy():
#    feature, group, tc, epochs = get_testing_set_n(1)
#    decoded, proba = nap.decode(tc, group, epochs, bin_size=1, use_occupancy=True)
#    np.testing.assert_array_almost_equal(feature.values, decoded.values)
#    assert isinstance(decoded, nap.Tsd)
#    assert isinstance(proba, nap.TsdFrame)
#    np.testing.assert_array_almost_equal(feature.values, decoded.values)
#
#
# def test_decode_1d_with_dict():
#    feature, group, tc, epochs = get_testing_set_n(1)
#    group = dict(group)
#    decoded, proba = nap.decode(tc, group, epochs, bin_size=1)
#    assert isinstance(decoded, nap.Tsd)
#    assert isinstance(proba, nap.TsdFrame)
#    np.testing.assert_array_almost_equal(feature.values, decoded.values)
#    assert len(decoded) == 100
#    assert len(proba) == 100
#    tmp = np.ones((100, 2))
#    tmp[50:, 0] = 0.0
#    tmp[0:50, 1] = 0.0
#    np.testing.assert_array_almost_equal(proba.values, tmp)
#
#
# def test_decode_1d_with_time_units():
#    feature, group, tc, epochs = get_testing_set_n(1)
#    for t, tu in zip([1, 1e3, 1e6], ["s", "ms", "us"]):
#        decoded, proba = nap.decode(tc, group, epochs, 1.0 * t, time_units=tu)
#        np.testing.assert_array_almost_equal(feature.values, decoded.values)
#
#
# def test_decoded_1d_raise_errors():
#    feature, group, tc, epochs = get_testing_set_n(1)
#    with pytest.raises(Exception) as e_info:
#        nap.decode(tc, np.random.rand(10), epochs, 1)
#    assert str(e_info.value) == "Unknown format for group"
#
#    feature, group, tc, epochs = get_testing_set_n(1)
#    _tc = xr.DataArray(data=np.random.rand(10, 3), dims=["time", "unit"])
#    with pytest.raises(Exception) as e_info:
#        nap.decode(_tc, group, epochs, 1)
#    assert str(e_info.value) == "Different shapes for tuning_curves and group"
#
#    feature, group, tc, epochs = get_testing_set_n(1)
#    tc.coords["unit"] = [0, 2]
#    with pytest.raises(Exception) as e_info:
#        nap.decode(tc, group, epochs, 1)
#    assert str(e_info.value) == "Different indices for tuning curves and group keys"

# ------------------------------------------------------------------------------------
# OLD DECODING TESTS
# ------------------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore")
def get_testing_set_1d():
    feature = nap.Tsd(t=np.arange(0, 100, 1), d=np.repeat(np.arange(0, 2), 50))
    group = nap.TsGroup({i: nap.Ts(t=np.arange(0, 50) + 50 * i) for i in range(2)})
    tc = nap.compute_1d_tuning_curves(
        group=group, feature=feature, nb_bins=2, minmax=(-0.5, 1.5)
    )
    ep = nap.IntervalSet(start=0, end=100)
    return feature, group, tc, ep


@pytest.mark.filterwarnings("ignore")
def test_decode_1d():
    feature, group, tc, ep = get_testing_set_1d()
    decoded, proba = nap.decode_1d(tc, group, ep, bin_size=1)
    assert isinstance(decoded, nap.Tsd)
    assert isinstance(proba, nap.TsdFrame)
    np.testing.assert_array_almost_equal(feature.values, decoded.values)
    assert len(decoded) == 100
    assert len(proba) == 100
    tmp = np.ones((100, 2))
    tmp[50:, 0] = 0.0
    tmp[0:50, 1] = 0.0
    np.testing.assert_array_almost_equal(proba.values, tmp)


@pytest.mark.filterwarnings("ignore")
def test_decode_1d_with_TsdFrame():
    feature, group, tc, ep = get_testing_set_1d()
    count = group.count(bin_size=1, ep=ep)
    decoded, proba = nap.decode_1d(tc, count, ep, bin_size=1)
    assert isinstance(decoded, nap.Tsd)
    assert isinstance(proba, nap.TsdFrame)
    np.testing.assert_array_almost_equal(feature.values, decoded.values)
    assert len(decoded) == 100
    assert len(proba) == 100
    tmp = np.ones((100, 2))
    tmp[50:, 0] = 0.0
    tmp[0:50, 1] = 0.0
    np.testing.assert_array_almost_equal(proba.values, tmp)


@pytest.mark.filterwarnings("ignore")
def test_decode_1d_with_feature():
    feature, group, tc, ep = get_testing_set_1d()
    decoded, proba = nap.decode_1d(tc, group, ep, bin_size=1, feature=feature)
    np.testing.assert_array_almost_equal(feature.values, decoded.values)
    assert isinstance(decoded, nap.Tsd)
    assert isinstance(proba, nap.TsdFrame)
    np.testing.assert_array_almost_equal(feature.values, decoded.values)
    assert len(decoded) == 100
    assert len(proba) == 100
    tmp = np.ones((100, 2))
    tmp[50:, 0] = 0.0
    tmp[0:50, 1] = 0.0
    np.testing.assert_array_almost_equal(proba.values, tmp)


@pytest.mark.filterwarnings("ignore")
def test_decode_1d_with_dict():
    feature, group, tc, ep = get_testing_set_1d()
    group = dict(group)
    decoded, proba = nap.decode_1d(tc, group, ep, bin_size=1, feature=feature)
    np.testing.assert_array_almost_equal(feature.values, decoded.values)
    assert isinstance(decoded, nap.Tsd)
    assert isinstance(proba, nap.TsdFrame)
    np.testing.assert_array_almost_equal(feature.values, decoded.values)
    assert len(decoded) == 100
    assert len(proba) == 100
    tmp = np.ones((100, 2))
    tmp[50:, 0] = 0.0
    tmp[0:50, 1] = 0.0
    np.testing.assert_array_almost_equal(proba.values, tmp)


@pytest.mark.filterwarnings("ignore")
def test_decode_1d_with_wrong_feature():
    feature, group, tc, ep = get_testing_set_1d()
    with pytest.raises(RuntimeError) as e_info:
        nap.decode_1d(tc, group, ep, bin_size=1, feature=[1, 2, 3])
    assert str(e_info.value) == "Unknown format for feature in decode_1d"


@pytest.mark.filterwarnings("ignore")
def test_decode_1d_with_time_units():
    feature, group, tc, ep = get_testing_set_1d()
    for t, tu in zip([1, 1e3, 1e6], ["s", "ms", "us"]):
        decoded, proba = nap.decode_1d(tc, group, ep, 1.0 * t, time_units=tu)
        np.testing.assert_array_almost_equal(feature.values, decoded.values)


@pytest.mark.filterwarnings("ignore")
def get_testing_set_2d():
    features = nap.TsdFrame(
        t=np.arange(0, 100, 1),
        d=np.vstack((np.repeat(np.arange(0, 2), 50), np.tile(np.arange(0, 2), 50))).T,
    )
    group = nap.TsGroup(
        {
            0: nap.Ts(np.arange(0, 50, 2)),
            1: nap.Ts(np.arange(1, 51, 2)),
            2: nap.Ts(np.arange(50, 100, 2)),
            3: nap.Ts(np.arange(51, 101, 2)),
        }
    )

    tc, xy = nap.compute_2d_tuning_curves(
        group=group, features=features, nb_bins=2, minmax=(-0.5, 1.5, -0.5, 1.5)
    )
    ep = nap.IntervalSet(start=0, end=100)
    return features, group, tc, ep, tuple(xy)


@pytest.mark.filterwarnings("ignore")
def test_decode_2d():
    features, group, tc, ep, xy = get_testing_set_2d()
    decoded, proba = nap.decode_2d(tc, group, ep, 1, xy)

    assert isinstance(decoded, nap.TsdFrame)
    assert isinstance(proba, nap.TsdTensor)
    np.testing.assert_array_almost_equal(features.values, decoded.values)
    assert len(decoded) == 100
    assert len(proba) == 100
    tmp = np.zeros((100, 2))
    tmp[0:50:2, 0] = 1
    tmp[50:100:2, 1] = 1
    np.testing.assert_array_almost_equal(proba[:, :, 0], tmp)

    tmp = np.zeros((100, 2))
    tmp[1:50:2, 0] = 1
    tmp[51:100:2, 1] = 1
    np.testing.assert_array_almost_equal(proba[:, :, 1], tmp)


@pytest.mark.filterwarnings("ignore")
def test_decode_2d_with_TsdFrame():
    features, group, tc, ep, xy = get_testing_set_2d()
    count = group.count(bin_size=1, ep=ep)
    decoded, proba = nap.decode_2d(tc, count, ep, 1, xy)

    assert isinstance(decoded, nap.TsdFrame)
    assert isinstance(proba, nap.TsdTensor)
    np.testing.assert_array_almost_equal(features.values, decoded.values)
    assert len(decoded) == 100
    assert len(proba) == 100
    tmp = np.zeros((100, 2))
    tmp[0:50:2, 0] = 1
    tmp[50:100:2, 1] = 1
    np.testing.assert_array_almost_equal(proba[:, :, 0], tmp)

    tmp = np.zeros((100, 2))
    tmp[1:50:2, 0] = 1
    tmp[51:100:2, 1] = 1
    np.testing.assert_array_almost_equal(proba[:, :, 1], tmp)


@pytest.mark.filterwarnings("ignore")
def test_decode_2d_with_dict():
    features, group, tc, ep, xy = get_testing_set_2d()
    group = dict(group)
    decoded, proba = nap.decode_2d(tc, group, ep, 1, xy)

    assert isinstance(decoded, nap.TsdFrame)
    assert isinstance(proba, nap.TsdTensor)
    np.testing.assert_array_almost_equal(features.values, decoded.values)
    assert len(decoded) == 100
    assert len(proba) == 100
    tmp = np.zeros((100, 2))
    tmp[0:50:2, 0] = 1
    tmp[50:100:2, 1] = 1
    np.testing.assert_array_almost_equal(proba[:, :, 0], tmp)

    tmp = np.zeros((100, 2))
    tmp[1:50:2, 0] = 1
    tmp[51:100:2, 1] = 1
    np.testing.assert_array_almost_equal(proba[:, :, 1], tmp)


@pytest.mark.filterwarnings("ignore")
def test_decode_2d_with_feature():
    features, group, tc, ep, xy = get_testing_set_2d()
    decoded, proba = nap.decode_2d(tc, group, ep, 1, xy)
    np.testing.assert_array_almost_equal(features.values, decoded.values)


@pytest.mark.filterwarnings("ignore")
def test_decode_2d_with_time_units():
    features, group, tc, ep, xy = get_testing_set_2d()
    for t, tu in zip([1, 1e3, 1e6], ["s", "ms", "us"]):
        decoded, proba = nap.decode_2d(tc, group, ep, 1.0 * t, xy, time_units=tu)
        np.testing.assert_array_almost_equal(features.values, decoded.values)
