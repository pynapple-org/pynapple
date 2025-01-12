import re

import numpy as np
import pytest

import pynapple as nap


############################################################
# Test for warping
############################################################
def get_input():
    return nap.Tsd(t=np.arange(10), d=np.arange(10))


def get_ep():
    return nap.IntervalSet(0, 10)


@pytest.mark.parametrize(
    "input, ep, binsize, align, padding_value, time_unit, expectation",
    [
        (
            {},
            get_ep(),
            1,
            "start",
            np.nan,
            "s",
            "Invalid type. Parameter input must be of type ['Ts', 'Tsd', 'TsdFrame', 'TsdTensor', 'TsGroup'].",
        ),
        (
            get_input(),
            {},
            1,
            "start",
            np.nan,
            "s",
            "Invalid type. Parameter ep must be of type ['IntervalSet'].",
        ),
        (
            get_input(),
            get_ep(),
            "a",
            "start",
            np.nan,
            "s",
            "Invalid type. Parameter binsize must be of type ['Number'].",
        ),
        (
            get_input(),
            get_ep(),
            1,
            1,
            np.nan,
            "s",
            "Invalid type. Parameter align must be of type ['str'].",
        ),
        (
            get_input(),
            get_ep(),
            1,
            "start",
            {},
            "s",
            "Invalid type. Parameter padding_value must be of type ['Number'].",
        ),
        (
            get_input(),
            get_ep(),
            1,
            "start",
            np.nan,
            1,
            "Invalid type. Parameter time_unit must be of type ['str'].",
        ),
    ],
)
def test_build_tensor_type_error(
    input, ep, binsize, align, padding_value, time_unit, expectation
):
    with pytest.raises(TypeError, match=re.escape(expectation)):
        nap.build_tensor(
            input=input,
            ep=ep,
            binsize=binsize,
            align=align,
            padding_value=padding_value,
            time_unit=time_unit,
        )


def test_build_tensor_runtime_error():
    group = nap.TsGroup(
        {
            0: nap.Ts(t=np.arange(0, 100)),
            1: nap.Ts(t=np.arange(0, 100, 0.5), time_units="s"),
            2: nap.Ts(t=np.arange(0, 100, 0.2), time_units="s"),
        }
    )
    ep = nap.IntervalSet(
        start=np.arange(0, 100, 20), end=np.arange(0, 100, 20) + np.arange(0, 10, 2)
    )

    with pytest.raises(
        RuntimeError,
        match=r"When input is a TsGroup or Ts object, binsize should be specified",
    ):
        nap.build_tensor(group, ep)

    with pytest.raises(RuntimeError, match=r"time_unit should be 's', 'ms' or 'us'"):
        nap.build_tensor(group, ep, 1, time_unit="a")

    with pytest.raises(RuntimeError, match=r"align should be 'start' or 'end'"):
        nap.build_tensor(group, ep, 1, align="a")


def test_build_tensor_with_group():
    group = nap.TsGroup(
        {
            0: nap.Ts(t=np.arange(0, 100)),
            1: nap.Ts(t=np.arange(0, 100, 0.5), time_units="s"),
            2: nap.Ts(t=np.arange(0, 100, 0.2), time_units="s"),
        }
    )
    ep = nap.IntervalSet(
        start=np.arange(0, 100, 20), end=np.arange(0, 100, 20) + np.arange(0, 10, 2)
    )

    expected = np.ones((len(group), len(ep), 8)) * np.nan
    for i, k in zip(range(len(ep)), range(2, 10, 2)):
        expected[:, i, 0:k] = 1
    for i, k in zip(range(len(group)), [1, 2, 5]):
        expected[i] *= k

    tensor = nap.build_tensor(group, ep, binsize=1)
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(group, ep, binsize=1, align="start")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(group, ep, binsize=1, align="end")
    np.testing.assert_array_almost_equal(tensor, np.flip(expected, axis=2))

    tensor = nap.build_tensor(group, ep, binsize=1, time_unit="s")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(group, ep, binsize=1e3, time_unit="ms")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(group, ep, binsize=1e6, time_unit="us")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(group, ep, binsize=1, align="start", padding_value=-1)
    expected[np.isnan(expected)] = -1
    np.testing.assert_array_almost_equal(tensor, expected)


def test_build_tensor_with_ts():
    ts = nap.Ts(t=np.arange(0, 100))
    ep = nap.IntervalSet(
        start=np.arange(0, 100, 20), end=np.arange(0, 100, 20) + np.arange(0, 10, 2)
    )

    expected = np.ones((len(ep), 8)) * np.nan
    for i, k in zip(range(len(ep)), range(2, 10, 2)):
        expected[i, 0:k] = 1

    tensor = nap.build_tensor(ts, ep, binsize=1)
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(ts, ep, binsize=1, align="start")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(ts, ep, binsize=1, align="end")
    np.testing.assert_array_almost_equal(tensor, np.flip(expected, axis=1))

    tensor = nap.build_tensor(ts, ep, binsize=1, time_unit="s")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(ts, ep, binsize=1e3, time_unit="ms")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(ts, ep, binsize=1e6, time_unit="us")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(ts, ep, binsize=1, align="start", padding_value=-1)
    expected[np.isnan(expected)] = -1
    np.testing.assert_array_almost_equal(tensor, expected)


def test_build_tensor_with_tsd():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    ep = nap.IntervalSet(
        start=np.arange(0, 100, 20), end=np.arange(0, 100, 20) + np.arange(0, 10, 2)
    )

    expected = np.ones((len(ep), 9)) * np.nan
    for i, k in zip(range(len(ep)), range(2, 10, 2)):
        expected[i, 0 : k + 1] = np.arange(k * 10, k * 10 + k + 1)

    tensor = nap.build_tensor(tsd, ep)
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(tsd, ep, align="start")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(tsd, ep, align="start", padding_value=-1)
    expected[np.isnan(expected)] = -1
    np.testing.assert_array_almost_equal(tensor, expected)

    expected = np.ones((len(ep), 9)) * np.nan
    for i, k in zip(range(len(ep)), range(2, 10, 2)):
        expected[i, -k - 1 :] = np.arange(k * 10, k * 10 + k + 1)

    tensor = nap.build_tensor(tsd, ep, align="end")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(tsd, ep, align="end", padding_value=-1)
    expected[np.isnan(expected)] = -1
    np.testing.assert_array_almost_equal(tensor, expected)


def test_build_tensor_with_tsdframe():
    tsdframe = nap.TsdFrame(t=np.arange(100), d=np.tile(np.arange(100)[:, None], 3))
    ep = nap.IntervalSet(
        start=np.arange(0, 100, 20), end=np.arange(0, 100, 20) + np.arange(0, 10, 2)
    )

    expected = np.ones((len(ep), 9)) * np.nan
    for i, k in zip(range(len(ep)), range(2, 10, 2)):
        expected[i, 0 : k + 1] = np.arange(k * 10, k * 10 + k + 1)
    expected = np.repeat(expected[None, :], 3, axis=0)

    tensor = nap.build_tensor(tsdframe, ep)
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(tsdframe, ep, align="start")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(tsdframe, ep, align="start", padding_value=-1)
    expected[np.isnan(expected)] = -1
    np.testing.assert_array_almost_equal(tensor, expected)

    expected = np.ones((len(ep), 9)) * np.nan
    for i, k in zip(range(len(ep)), range(2, 10, 2)):
        expected[i, -k - 1 :] = np.arange(k * 10, k * 10 + k + 1)
    expected = np.repeat(expected[None, :], 3, axis=0)

    tensor = nap.build_tensor(tsdframe, ep, align="end")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(tsdframe, ep, align="end", padding_value=-1)
    expected[np.isnan(expected)] = -1
    np.testing.assert_array_almost_equal(tensor, expected)


def test_build_tensor_with_tsdtensor():
    tsdtensor = nap.TsdTensor(
        t=np.arange(100), d=np.tile(np.arange(100)[:, None, None], (1, 2, 3))
    )
    ep = nap.IntervalSet(
        start=np.arange(0, 100, 20), end=np.arange(0, 100, 20) + np.arange(0, 10, 2)
    )

    expected = np.ones((len(ep), 9)) * np.nan
    for i, k in zip(range(len(ep)), range(2, 10, 2)):
        expected[i, 0 : k + 1] = np.arange(k * 10, k * 10 + k + 1)
    expected = np.tile(expected[None, None, :, :], (2, 3, 1, 1))

    tensor = nap.build_tensor(tsdtensor, ep)
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(tsdtensor, ep, align="start")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(tsdtensor, ep, align="start", padding_value=-1)
    expected[np.isnan(expected)] = -1
    np.testing.assert_array_almost_equal(tensor, expected)

    expected = np.ones((len(ep), 9)) * np.nan
    for i, k in zip(range(len(ep)), range(2, 10, 2)):
        expected[i, -k - 1 :] = np.arange(k * 10, k * 10 + k + 1)
    expected = np.tile(expected[None, None, :, :], (2, 3, 1, 1))

    tensor = nap.build_tensor(tsdtensor, ep, align="end")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(tsdtensor, ep, align="end", padding_value=-1)
    expected[np.isnan(expected)] = -1
    np.testing.assert_array_almost_equal(tensor, expected)
