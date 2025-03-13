import re

import numpy as np
import pytest

import pynapple as nap


############################################################
# Test for tensor building
############################################################
def get_group():
    return nap.TsGroup(
        {
            0: nap.Ts(t=np.arange(0, 100)),
            1: nap.Ts(t=np.arange(0, 100, 0.5), time_units="s"),
            2: nap.Ts(t=np.arange(0, 100, 0.2), time_units="s"),
        }
    )


def get_ts():
    return nap.Ts(t=np.arange(0, 100))


def get_tsd():
    return nap.Tsd(t=np.arange(100), d=np.arange(100))


def get_tsdframe():
    return nap.TsdFrame(t=np.arange(100), d=np.tile(np.arange(100)[:, None], 3))


def get_tsdtensor():
    return nap.TsdTensor(
        t=np.arange(100), d=np.tile(np.arange(100)[:, None, None], (1, 2, 3))
    )


def get_ep():
    return nap.IntervalSet(
        start=np.arange(0, 100, 20), end=np.arange(0, 100, 20) + np.arange(0, 10, 2)
    )


@pytest.mark.parametrize(
    "input, ep, bin_size, align, padding_value, time_unit, expectation",
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
            get_tsd(),
            {},
            1,
            "start",
            np.nan,
            "s",
            "Invalid type. Parameter ep must be of type ['IntervalSet'].",
        ),
        (
            get_tsd(),
            get_ep(),
            "a",
            "start",
            np.nan,
            "s",
            "Invalid type. Parameter bin_size must be of type ['Number'].",
        ),
        (
            get_tsd(),
            get_ep(),
            1,
            1,
            np.nan,
            "s",
            "Invalid type. Parameter align must be of type ['str'].",
        ),
        (
            get_tsd(),
            get_ep(),
            1,
            "start",
            {},
            "s",
            "Invalid type. Parameter padding_value must be of type ['Number'].",
        ),
        (
            get_tsd(),
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
    input, ep, bin_size, align, padding_value, time_unit, expectation
):
    with pytest.raises(TypeError, match=re.escape(expectation)):
        nap.build_tensor(
            input=input,
            ep=ep,
            bin_size=bin_size,
            align=align,
            padding_value=padding_value,
            time_unit=time_unit,
        )


def test_build_tensor_runtime_error():
    group = get_group()
    ep = get_ep()

    with pytest.raises(
        RuntimeError,
        match=r"When input is a TsGroup or Ts object, bin_size should be specified",
    ):
        nap.build_tensor(group, ep)

    with pytest.raises(RuntimeError, match=r"time_unit should be 's', 'ms' or 'us'"):
        nap.build_tensor(group, ep, 1, time_unit="a")

    with pytest.raises(RuntimeError, match=r"align should be 'start' or 'end'"):
        nap.build_tensor(group, ep, 1, align="a")


def test_build_tensor_with_group():
    group = get_group()
    ep = get_ep()

    expected = np.ones((len(group), len(ep), 8)) * np.nan
    for i, k in zip(range(len(ep)), range(2, 10, 2)):
        expected[:, i, 0:k] = 1
    for i, k in zip(range(len(group)), [1, 2, 5]):
        expected[i] *= k

    tensor = nap.build_tensor(group, ep, bin_size=1)
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(group, ep, bin_size=1, align="start")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(group, ep, bin_size=1, align="end")
    np.testing.assert_array_almost_equal(tensor, np.flip(expected, axis=2))

    tensor = nap.build_tensor(group, ep, bin_size=1, time_unit="s")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(group, ep, bin_size=1e3, time_unit="ms")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(group, ep, bin_size=1e6, time_unit="us")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(group, ep, bin_size=1, align="start", padding_value=-1)
    expected[np.isnan(expected)] = -1
    np.testing.assert_array_almost_equal(tensor, expected)


def test_build_tensor_with_ts():
    ts = get_ts()
    ep = get_ep()

    expected = np.ones((len(ep), 8)) * np.nan
    for i, k in zip(range(len(ep)), range(2, 10, 2)):
        expected[i, 0:k] = 1

    tensor = nap.build_tensor(ts, ep, bin_size=1)
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(ts, ep, bin_size=1, align="start")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(ts, ep, bin_size=1, align="end")
    np.testing.assert_array_almost_equal(tensor, np.flip(expected, axis=1))

    tensor = nap.build_tensor(ts, ep, bin_size=1, time_unit="s")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(ts, ep, bin_size=1e3, time_unit="ms")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(ts, ep, bin_size=1e6, time_unit="us")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(ts, ep, bin_size=1, align="start", padding_value=-1)
    expected[np.isnan(expected)] = -1
    np.testing.assert_array_almost_equal(tensor, expected)


def test_build_tensor_with_tsd():
    tsd = get_tsd()
    ep = get_ep()

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
    tsdframe = get_tsdframe()
    ep = get_ep()

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
    tsdtensor = get_tsdtensor()
    ep = get_ep()

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


#######################################################################
# Time Warping
#######################################################################


def get_group2():
    return nap.TsGroup(
        {
            0: nap.Ts(t=np.arange(0, 100, 0.2), time_units="s"),
            1: nap.Ts(t=np.arange(0, 100, 0.1), time_units="s"),
        }
    )


@pytest.mark.parametrize(
    "input, ep, num_bins, expectation",
    [
        (
            {},
            get_ep(),
            10,
            "Invalid type. Parameter input must be of type ['Ts', 'Tsd', 'TsdFrame', 'TsdTensor', 'TsGroup'].",
        ),
        (
            get_tsd(),
            {},
            10,
            "Invalid type. Parameter ep must be of type ['IntervalSet'].",
        ),
        (
            get_tsd(),
            get_ep(),
            "a",
            "Invalid type. Parameter num_bins must be of type ['int'].",
        ),
    ],
)
def test_warp_tensor_type_error(input, ep, num_bins, expectation):
    with pytest.raises(TypeError, match=re.escape(expectation)):
        nap.warp_tensor(input=input, ep=ep, num_bins=num_bins)


def test_warp_tensor_runtime_error():
    group = get_group()
    ep = get_ep()
    with pytest.raises(RuntimeError, match=r"num_bins should be positive integer."):
        nap.warp_tensor(group, ep, -10)
    with pytest.raises(RuntimeError, match=r"num_bins should be positive integer."):
        nap.warp_tensor(group, ep, 0)


def test_warp_tensor_with_tsgroup():
    group = get_group2()
    ep = get_ep()

    expected = np.zeros((len(group), len(ep), 10))
    for i in range(2):
        expected[i, :, :] = np.tile((np.arange(1, 5, 1) * (i + 1))[:, None], 10)

    tensor = nap.warp_tensor(group, ep, num_bins=10)
    np.testing.assert_array_almost_equal(tensor, expected)


def test_warp_tensor_with_ts():
    ts = get_group2()[0]
    ep = get_ep()

    expected = np.tile((np.arange(1, 5, 1))[:, None], 10)

    tensor = nap.warp_tensor(ts, ep, num_bins=10)
    np.testing.assert_array_almost_equal(tensor, expected)


def test_warp_tensor_with_tsd():
    tsd = get_tsd()

    # Equal
    ep = nap.IntervalSet(start=np.arange(20, 100, 20), end=np.arange(30, 110, 20))
    expected = np.array([np.arange(i, i + 11) for i in ep.start])
    tensor = nap.warp_tensor(tsd, ep, 11)
    np.testing.assert_array_almost_equal(tensor, expected)

    # More time points than bins
    expected2 = np.array(
        [arr.mean(1) for arr in np.array_split(expected[:, 0:10], 2, axis=1)]
    ).T
    tensor = nap.warp_tensor(tsd, ep, 2)
    np.testing.assert_array_almost_equal(tensor, expected2)

    # Less time points than bins
    expected3 = np.array([np.linspace(s, e, 20) for s, e in ep.values])
    tensor = nap.warp_tensor(tsd, ep, 20)
    np.testing.assert_array_almost_equal(tensor, expected3)


def test_warp_tensor_with_tsdframe():
    tsdframe = get_tsdframe()

    # Equal
    ep = nap.IntervalSet(start=np.arange(20, 100, 20), end=np.arange(30, 110, 20))
    expected = np.array([np.arange(i, i + 11) for i in ep.start])
    expected = np.repeat(expected[None, :, :], 3, axis=0)
    tensor = nap.warp_tensor(tsdframe, ep, 11)
    np.testing.assert_array_almost_equal(tensor, expected)

    # More time points than bins
    expected2 = np.array(
        [arr.mean(2) for arr in np.array_split(expected[:, :, 0:10], 2, axis=2)]
    )
    expected2 = np.moveaxis(expected2, source=0, destination=-1)
    tensor = nap.warp_tensor(tsdframe, ep, 2)
    np.testing.assert_array_almost_equal(tensor, expected2)

    # Less time points than bins
    expected3 = np.array([np.linspace(s, e, 20) for s, e in ep.values])
    expected3 = np.repeat(expected3[None, :, :], 3, axis=0)
    tensor = nap.warp_tensor(tsdframe, ep, 20)
    np.testing.assert_array_almost_equal(tensor, expected3)


def test_warp_tensor_with_tsdtensor():
    tsdtensor = get_tsdtensor()

    # Equal
    ep = nap.IntervalSet(start=np.arange(20, 100, 20), end=np.arange(30, 110, 20))
    expected = np.array([np.arange(i, i + 11) for i in ep.start])
    expected = np.repeat(expected[None, :, :], 3, axis=0)
    expected = np.repeat(expected[None, :, :, :], 2, axis=0)
    tensor = nap.warp_tensor(tsdtensor, ep, 11)
    np.testing.assert_array_almost_equal(tensor, expected)

    # More time points than bins
    expected2 = np.array(
        [arr.mean(-1) for arr in np.array_split(expected[:, :, :, 0:10], 2, axis=-1)]
    )
    expected2 = np.moveaxis(expected2, source=0, destination=-1)
    tensor = nap.warp_tensor(tsdtensor, ep, 2)
    np.testing.assert_array_almost_equal(tensor, expected2)

    # Less time points than bins
    expected3 = np.array([np.linspace(s, e, 20) for s, e in ep.values])
    expected3 = np.repeat(expected3[None, :, :], 3, axis=0)
    expected3 = np.repeat(expected3[None, :, :, :], 2, axis=0)
    tensor = nap.warp_tensor(tsdtensor, ep, 20)
    np.testing.assert_array_almost_equal(tensor, expected3)
