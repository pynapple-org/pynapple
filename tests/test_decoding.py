# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-03-30 11:16:39
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-01-29 11:15:41
#!/usr/bin/env python

"""Tests of decoding for `pynapple` package."""

import numpy as np
import pytest
import xarray as xr

import pynapple as nap


def get_testing_set_1d():
    feature = nap.Tsd(t=np.arange(0, 100, 1), d=np.repeat(np.arange(0, 2), 50))
    group = nap.TsGroup({i: nap.Ts(t=np.arange(0, 50) + 50 * i) for i in range(2)})
    tc = nap.compute_tuning_curves(
        group=group, features=feature, bins=2, range=(-0.5, 1.5)
    )
    epochs = nap.IntervalSet(start=0, end=100)
    return feature, group, tc, epochs


def test_decode_1d():
    feature, group, tc, epochs = get_testing_set_1d()
    decoded, proba = nap.decode(tc, group, epochs, bin_size=1)
    assert isinstance(decoded, nap.Tsd)
    assert isinstance(proba, nap.TsdFrame)
    np.testing.assert_array_almost_equal(feature.values, decoded.values)
    assert len(decoded) == 100
    assert len(proba) == 100
    tmp = np.ones((100, 2))
    tmp[50:, 0] = 0.0
    tmp[0:50, 1] = 0.0
    np.testing.assert_array_almost_equal(proba.values, tmp)


def test_decode_1d_with_TsdFrame():
    feature, group, tc, epochs = get_testing_set_1d()
    count = group.count(bin_size=1, ep=epochs)
    decoded, proba = nap.decode(tc, count, epochs, bin_size=1)
    assert isinstance(decoded, nap.Tsd)
    assert isinstance(proba, nap.TsdFrame)
    np.testing.assert_array_almost_equal(feature.values, decoded.values)
    assert len(decoded) == 100
    assert len(proba) == 100
    tmp = np.ones((100, 2))
    tmp[50:, 0] = 0.0
    tmp[0:50, 1] = 0.0
    np.testing.assert_array_almost_equal(proba.values, tmp)


def test_decode_1d_with_dict():
    feature, group, tc, epochs = get_testing_set_1d()
    group = dict(group)
    decoded, proba = nap.decode(tc, group, epochs, bin_size=1)
    assert isinstance(decoded, nap.Tsd)
    assert isinstance(proba, nap.TsdFrame)
    np.testing.assert_array_almost_equal(feature.values, decoded.values)
    assert len(decoded) == 100
    assert len(proba) == 100
    tmp = np.ones((100, 2))
    tmp[50:, 0] = 0.0
    tmp[0:50, 1] = 0.0
    np.testing.assert_array_almost_equal(proba.values, tmp)


def test_decode_1d_with_time_units():
    feature, group, tc, epochs = get_testing_set_1d()
    for t, tu in zip([1, 1e3, 1e6], ["s", "ms", "us"]):
        decoded, proba = nap.decode(tc, group, epochs, 1.0 * t, time_units=tu)
        np.testing.assert_array_almost_equal(feature.values, decoded.values)


def test_decoded_1d_raise_errors():
    feature, group, tc, epochs = get_testing_set_1d()
    with pytest.raises(Exception) as e_info:
        nap.decode(tc, np.random.rand(10), epochs, 1)
    assert str(e_info.value) == "Unknown format for group"

    feature, group, tc, epochs = get_testing_set_1d()
    _tc = xr.DataArray(data=np.random.rand(10, 3), dims=["time", "unit"])
    with pytest.raises(Exception) as e_info:
        nap.decode(_tc, group, epochs, 1)
    assert str(e_info.value) == "Different shapes for tuning_curves and group"

    feature, group, tc, epochs = get_testing_set_1d()
    tc.coords["unit"] = [0, 2]
    with pytest.raises(Exception) as e_info:
        nap.decode(tc, group, epochs, 1)
    assert str(e_info.value) == "Different indices for tuning curves and group keys"


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

    tc = nap.compute_tuning_curves(
        group=group, features=features, bins=2, range=[(-0.5, 1.5), (-0.5, 1.5)]
    )
    epochs = nap.IntervalSet(start=0, end=100)
    return features, group, tc, epochs


def test_decode_2d():
    features, group, tc, epochs = get_testing_set_2d()
    decoded, proba = nap.decode(tc, group, epochs, 1)

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


def test_decode_2d_with_TsdFrame():
    features, group, tc, epochs = get_testing_set_2d()
    count = group.count(bin_size=1, ep=epochs)
    decoded, proba = nap.decode(tc, count, epochs, 1)

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


def test_decode_2d_with_dict():
    features, group, tc, epochs = get_testing_set_2d()
    group = dict(group)
    decoded, proba = nap.decode(tc, group, epochs, 1)

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


def test_decode_2d_with_time_units():
    features, group, tc, epochs = get_testing_set_2d()
    for t, tu in zip([1, 1e3, 1e6], ["s", "ms", "us"]):
        decoded, proba = nap.decode(tc, group, epochs, 1.0 * t, time_units=tu)
        np.testing.assert_array_almost_equal(features.values, decoded.values)


def test_decoded_2d_raise_errors():
    features, group, tc, epochs = get_testing_set_2d()
    with pytest.raises(Exception) as e_info:
        nap.decode(tc, np.random.rand(10), epochs, 1)
    assert str(e_info.value) == "Unknown format for group"

    features, group, tc, epochs = get_testing_set_2d()
    tc = xr.DataArray(data=np.random.rand(10, 3), dims=["time", "unit"])
    with pytest.raises(Exception) as e_info:
        nap.decode(tc, group, epochs, 1)
    assert str(e_info.value) == "Different shapes for tuning_curves and group"

    features, group, tc, epochs = get_testing_set_2d()
    tc.coords["unit"] = [0, 2, 4, 6]
    with pytest.raises(Exception) as e_info:
        nap.decode(tc, group, epochs, 1)
    assert str(e_info.value) == "Different indices for tuning curves and group keys"
