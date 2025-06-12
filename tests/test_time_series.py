"""Tests of time series for `pynapple` package."""

import pickle
import re
import warnings
from contextlib import nullcontext as does_not_raise
from numbers import Number
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.signal import decimate

import pynapple as nap
from pynapple.core.time_series import is_array_like

from .helper_tests import skip_if_backend

# tsd1 = nap.Tsd(t=np.arange(100), d=np.random.rand(100), time_units="s")
# tsd2 = nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 3), columns = ['a', 'b', 'c'])
# tsd3 = nap.TsdTensor(t=np.arange(100), d=np.random.rand(100, 5, 4), time_units="s")
# tsd4 = nap.Ts(t=np.arange(100), time_units="s")


def decimate_scipy(tsd, ep, order, down, filter_type):
    out_sci = []
    for iset in ep:
        out_sci.append(
            decimate(
                tsd.restrict(iset).d,
                q=down,
                n=order,
                ftype=filter_type,
                axis=0,
                zero_phase=True,
            )
        )
    out_sci = np.concatenate(out_sci, axis=0)
    return out_sci


def test_create_tsd():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    assert isinstance(tsd, nap.Tsd)


def test_create_empty_tsd():
    tsd = nap.Tsd(t=np.array([]), d=np.array([]))
    assert len(tsd) == 0


@pytest.mark.filterwarnings("ignore")
def test_create_tsd_from_number():
    tsd = nap.Tsd(t=1, d=2)


@pytest.mark.parametrize(
    "t, d, columns, metadata",
    [
        (np.arange(100), np.random.rand(100, 4), None, {}),
        (np.arange(100), np.random.rand(100, 4), ["a", "b", "c", "d"], {}),
        (
            np.arange(100),
            np.random.rand(100, 4),
            ["a", "b", "c", "d"],
            {"l1": np.arange(4), "l2": ["w", "x", "y", "z"]},
        ),
    ],
)
def test_create_tsdframe(t, d, columns, metadata):
    tsdframe = nap.TsdFrame(t=t, d=d, columns=columns, metadata=metadata)
    assert isinstance(tsdframe, nap.TsdFrame)
    if columns is not None:
        assert np.all(tsdframe.columns == np.array(columns))
    if len(metadata):
        for key, value in metadata.items():
            assert np.all(tsdframe._metadata[key] == np.array(value))


@pytest.mark.filterwarnings("ignore")
def test_create_empty_tsdframe():
    tsdframe = nap.TsdFrame(t=np.array([]), d=np.empty(shape=(0, 2)))
    assert len(tsdframe) == 0
    assert isinstance(tsdframe, nap.TsdFrame)

    with pytest.raises(AssertionError):
        tsdframe = nap.TsdFrame(t=np.arange(100))


def test_create_1d_tsdframe():
    tsdframe = nap.TsdFrame(t=np.arange(100), d=np.random.rand(100))
    assert isinstance(tsdframe, nap.TsdFrame)


def test_create_ts():
    ts = nap.Ts(t=np.arange(100))
    assert isinstance(ts, nap.Ts)


def test_create_ts_from_us():
    a = np.random.randint(0, 1000, 100)
    a.sort()
    ts = nap.Ts(t=a, time_units="us")
    np.testing.assert_array_almost_equal(ts.index, a / 1000 / 1000)


def test_create_ts_from_ms():
    a = np.random.randint(0, 1000, 100)
    a.sort()
    ts = nap.Ts(t=a, time_units="ms")
    np.testing.assert_array_almost_equal(ts.index, a / 1000)


def test_create_ts_from_s():
    a = np.random.randint(0, 1000, 100)
    a.sort()
    ts = nap.Ts(t=a, time_units="s")
    np.testing.assert_array_almost_equal(ts.index, a)


def test_create_tsdframe_from_us():
    a = np.random.randint(0, 1000, 100)
    d = np.random.rand(100, 3)
    a.sort()
    tsdframe = nap.TsdFrame(t=a, d=d, time_units="us")
    np.testing.assert_array_almost_equal(tsdframe.index, a / 1000 / 1000)


def test_create_tsdframe_from_ms():
    a = np.random.randint(0, 1000, 100)
    d = np.random.rand(100, 3)
    a.sort()
    tsdframe = nap.TsdFrame(t=a, d=d, time_units="ms")
    np.testing.assert_array_almost_equal(tsdframe.index, a / 1000)


def test_create_tsdframe_from_s():
    a = np.random.randint(0, 1000, 100)
    d = np.random.rand(100, 3)
    a.sort()
    tsdframe = nap.TsdFrame(t=a, d=d, time_units="s")
    np.testing.assert_array_almost_equal(tsdframe.index, a)


def test_create_ts_wrong_units():
    a = np.random.randint(0, 1000, 100)
    a.sort()
    with pytest.raises(ValueError):
        nap.Ts(t=a, time_units="min")


def test_create_tsd_wrong_units():
    a = np.random.randint(0, 1000, 100)
    a.sort()
    with pytest.raises(ValueError):
        nap.Tsd(t=a, d=a, time_units="min")


def test_create_tsdframe_wrong_units():
    a = np.random.randint(0, 1000, 100)
    d = np.random.rand(100, 3)
    a.sort()
    with pytest.raises(ValueError):
        nap.TsdFrame(t=a, d=d, time_units="min")


@pytest.mark.filterwarnings("ignore")
def test_create_ts_from_non_sorted():
    a = np.random.randint(0, 1000, 100)
    ts = nap.Ts(t=a, time_units="s")
    np.testing.assert_array_almost_equal(ts.index, np.sort(a))


@pytest.mark.filterwarnings("ignore")
def test_create_tsdframe_from_non_sorted():
    a = np.random.randint(0, 1000, 100)
    d = np.random.rand(100, 3)
    tsdframe = nap.TsdFrame(t=a, d=d, time_units="s")
    np.testing.assert_array_almost_equal(tsdframe.index, np.sort(a))


def test_create_ts_with_time_support():
    ep = nap.IntervalSet(start=[0, 20], end=[10, 30])
    ts = nap.Ts(t=np.arange(100), time_units="s", time_support=ep)
    assert len(ts) == 22
    np.testing.assert_array_almost_equal(ts.time_support.start, ep.start)
    np.testing.assert_array_almost_equal(ts.time_support.end, ep.end)


def test_create_tsd_with_time_support():
    ep = nap.IntervalSet(start=[0, 20], end=[10, 30])
    tsd = nap.Tsd(
        t=np.arange(100), d=np.random.rand(100), time_units="s", time_support=ep
    )
    assert len(tsd) == 22
    np.testing.assert_array_almost_equal(tsd.time_support.start, ep.start)
    np.testing.assert_array_almost_equal(tsd.time_support.end, ep.end)


def test_create_tsdframe_with_time_support():
    ep = nap.IntervalSet(start=[0, 20], end=[10, 30])
    tsdframe = nap.TsdFrame(
        t=np.arange(100), d=np.random.rand(100, 3), time_units="s", time_support=ep
    )
    assert len(tsdframe) == 22
    np.testing.assert_array_almost_equal(tsdframe.time_support.start, ep.start)
    np.testing.assert_array_almost_equal(tsdframe.time_support.end, ep.end)


@pytest.mark.filterwarnings("ignore")
def test_create_tsdtensor():
    tsd = nap.TsdTensor(t=np.arange(100), d=np.random.rand(100, 3, 2))
    assert isinstance(tsd, nap.TsdTensor)

    tsd = nap.TsdTensor(t=list(np.arange(100)), d=list(np.random.rand(100, 3, 2)))
    assert isinstance(tsd, nap.TsdTensor)


def test_create_empty_tsd():
    tsd = nap.TsdTensor(t=np.array([]), d=np.empty(shape=(0, 2, 3)))
    assert len(tsd) == 0


@pytest.mark.filterwarnings("ignore")
def test_raise_error_tsdtensor_init():
    with pytest.raises(
        RuntimeError,
        match=r"Unknown format for d. Accepted formats are numpy.ndarray, list, tuple or any array-like objects.",
    ):
        nap.TsdTensor(t=np.arange(100), d=None)

    # with pytest.raises(AssertionError, match=r"Data should have more than 2 dimensions. If ndim < 3, use TsdFrame or Tsd object"):
    #     nap.TsdTensor(t=np.arange(100), d=np.random.rand(100, 10))

    with pytest.raises(
        AssertionError
    ):  # , match=r"Length of values (10) does not match length of index (100)"):
        nap.TsdTensor(t=np.arange(100), d=np.random.rand(10, 10, 3))


def test_index_error():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    ts = nap.Ts(t=np.arange(100))

    if isinstance(tsd.d, np.ndarray):
        with pytest.raises(IndexError):
            tsd[1000] = 0

        with pytest.raises(IndexError):
            ts[1000]


def test_find_support():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    ep = tsd.find_support(1.0)
    assert ep[0, 0] == 0
    assert ep[0, 1] == 99.0 + 1e-6

    t = np.hstack((np.arange(10), np.arange(20, 30)))
    tsd = nap.Tsd(t=t, d=np.arange(20))
    ep = tsd.find_support(1.0)
    np.testing.assert_array_equal(ep.start, np.array([0.0, 20.0]))
    np.testing.assert_array_equal(ep.end, np.array([9.0 + 1e-6, 29 + 1e-6]))


def test_properties():
    t = np.arange(100)
    d = np.random.rand(100).astype(np.float32)  # to match pynajax
    tsd = nap.Tsd(t=t, d=d)

    assert hasattr(tsd, "t")
    assert hasattr(tsd, "d")
    assert hasattr(tsd, "start")
    assert hasattr(tsd, "end")
    assert hasattr(tsd, "shape")
    assert hasattr(tsd, "ndim")
    assert hasattr(tsd, "size")

    np.testing.assert_array_equal(tsd.t, t)
    np.testing.assert_array_equal(tsd.d, d)
    assert tsd.start == 0.0
    assert tsd.end == 99.0
    assert tsd.shape == (100,)
    assert tsd.ndim == 1
    assert tsd.size == 100

    with pytest.raises(RuntimeError):
        tsd.rate = 0


def test_base_tsd_class():
    class DummyTsd(nap.core.time_series._BaseTsd):
        def __init__(self, t, d):
            super().__init__(t, d)

        def __getitem__(self, key):
            return self.values.__getitem__(key)

    tsd = DummyTsd([0, 1], [1, 2])
    assert isinstance(tsd.rate, float)
    assert isinstance(tsd.index, nap.TsIndex)
    try:
        assert isinstance(tsd.values, np.ndarray)
    except AssertionError:
        assert nap.core.utils.is_array_like(tsd.values)  # for pynajax

    assert isinstance(tsd.__repr__(), str)

    with pytest.raises((IndexError, TypeError)):
        tsd["a"]


####################################################
# General test for time series
####################################################
@pytest.mark.parametrize(
    "tsd",
    [
        nap.Tsd(t=np.arange(100), d=np.random.rand(100), time_units="s"),
        nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 5), time_units="s"),
        nap.TsdTensor(t=np.arange(100), d=np.random.rand(100, 5, 2), time_units="s"),
        nap.Ts(t=np.arange(100), time_units="s"),
    ],
)
class TestTimeSeriesGeneral:
    def test_as_units(self, tsd):
        if hasattr(tsd, "as_units"):
            tmp2 = tsd.index
            np.testing.assert_array_almost_equal(tsd.as_units("s").index, tmp2)
            np.testing.assert_array_almost_equal(tsd.as_units("ms").index, tmp2 * 1e3)
            np.testing.assert_array_almost_equal(tsd.as_units("us").index, tmp2 * 1e6)
            # np.testing.assert_array_almost_equal(tsd.as_units(units="a").index.values, tmp2)

    def test_rate(self, tsd):
        rate = len(tsd) / tsd.time_support.tot_length("s")
        np.testing.assert_approx_equal(tsd.rate, rate)

    def test_times(self, tsd):
        tmp = tsd.index
        np.testing.assert_array_almost_equal(tsd.times("s"), tmp)
        np.testing.assert_array_almost_equal(tsd.times("ms"), tmp * 1e3)
        np.testing.assert_array_almost_equal(tsd.times("us"), tmp * 1e6)

    def test_start_end(self, tsd):
        assert tsd.start_time() == tsd.index[0]
        assert tsd.end_time() == tsd.index[-1]
        assert tsd.start == tsd.index[0]
        assert tsd.end == tsd.index[-1]

    def test_time_support_interval_set(self, tsd):
        assert isinstance(tsd.time_support, nap.IntervalSet)

    def test_time_support_start_end(self, tsd):
        np.testing.assert_approx_equal(tsd.time_support.start[0], 0)
        np.testing.assert_approx_equal(tsd.time_support.end[0], 99.0)

    def test_time_support_include_tsd(self, tsd):
        np.testing.assert_approx_equal(tsd.time_support.start[0], tsd.index[0])
        np.testing.assert_approx_equal(tsd.time_support.end[0], tsd.index[-1])

    def test_value_from_tsd(self, tsd):
        tsd2 = nap.Tsd(t=np.arange(0, 100, 0.1), d=np.random.rand(1000))
        tsd3 = tsd.value_from(tsd2)
        assert len(tsd) == len(tsd3)
        np.testing.assert_array_almost_equal(tsd2.values[::10], tsd3.values)

    @pytest.mark.parametrize(
        "mode, expectation",
        [
            ("before", does_not_raise()),
            ("closest", does_not_raise()),
            ("after", does_not_raise()),
            (
                "invalid",
                pytest.raises(
                    ValueError, match='Argument ``mode`` should be "closest",'
                ),
            ),
        ],
    )
    def test_value_from_tsd_mode_type(self, tsd, mode, expectation):
        tsd2 = nap.Tsd(t=np.arange(0, 100, 0.1), d=np.random.rand(1000))
        with expectation:
            tsd.value_from(tsd2, mode=mode)

    @pytest.mark.parametrize("mode", ["before", "closest", "after"])
    def test_value_from_tsd_mode(self, tsd, mode):
        # case 1: tim-stamps form tsd are subset of time-stamps of tsd2
        # In this case all modes should do the same thing
        tsd2 = nap.Tsd(t=np.arange(0, 100, 0.1), d=np.random.rand(1000))
        tsd3 = tsd.value_from(tsd2, mode=mode)
        assert len(tsd) == len(tsd3)
        np.testing.assert_array_almost_equal(tsd2.values[::10], tsd3.values)

        # case2: timestamps of tsd (integers) are not subset of that of tsd2.
        tsd2 = nap.Tsd(t=np.arange(0, 100, 0.3), d=np.random.rand(334))
        tsd3 = tsd.value_from(tsd2, mode=mode)
        # loop over epochs
        for iset in tsd.time_support:
            single_ep_tsd = tsd.restrict(iset)
            single_ep_tsd3 = tsd3.restrict(iset)
            single_ep_tsd2 = tsd2.restrict(iset)
            # extract the indices with searchsorted.
            if mode == "before":
                expected_idx = (
                    np.searchsorted(single_ep_tsd2.t, single_ep_tsd.t, side="right") - 1
                )
                # check that times are actually before
                assert np.all(single_ep_tsd2.t[expected_idx] <= single_ep_tsd3.t)
                # check that subsequent are after
                assert np.all(
                    single_ep_tsd2.t[expected_idx[:-1] + 1] > single_ep_tsd3.t[:-1]
                )
            elif mode == "after":
                expected_idx = np.searchsorted(
                    single_ep_tsd2.t, single_ep_tsd.t, side="left"
                )
                # check that times are actually before
                assert np.all(single_ep_tsd2.t[expected_idx] >= single_ep_tsd3.t)
                # check that subsequent are after
                assert np.all(
                    single_ep_tsd2.t[expected_idx[1:] - 1] < single_ep_tsd3.t[1:]
                )
            else:
                before = (
                    np.searchsorted(single_ep_tsd2.t, single_ep_tsd.t, side="right") - 1
                )
                after = np.searchsorted(single_ep_tsd2.t, single_ep_tsd.t, side="left")
                dt_before = np.abs(single_ep_tsd2.t[before] - single_ep_tsd.t)
                dt_after = np.abs(single_ep_tsd2.t[after] - single_ep_tsd.t)
                expected_idx = before.copy()
                # by default if equi-distance, it assigned to after.
                expected_idx[dt_after <= dt_before] = after[dt_after <= dt_before]

            np.testing.assert_array_equal(
                single_ep_tsd2.d[expected_idx], single_ep_tsd3.d
            )
            np.testing.assert_array_equal(single_ep_tsd.t, single_ep_tsd3.t)

    def test_value_from_tsdframe(self, tsd):
        tsdframe = nap.TsdFrame(t=np.arange(0, 100, 0.1), d=np.random.rand(1000, 3))
        tsdframe2 = tsd.value_from(tsdframe)
        assert len(tsd) == len(tsdframe2)
        np.testing.assert_array_almost_equal(tsdframe.values[::10], tsdframe2.values)

    @pytest.mark.parametrize("mode", ["before", "closest", "after"])
    def test_value_from_tsdframe_mode(self, tsd, mode):
        # case 1: tim-stamps form tsd are subset of time-stamps of tsd2
        # In this case all modes should do the same thing
        tsd2 = nap.Tsd(t=np.arange(0, 100, 0.1), d=np.random.rand(1000))
        tsd3 = tsd.value_from(tsd2, mode=mode)
        assert len(tsd) == len(tsd3)
        np.testing.assert_array_almost_equal(tsd2.values[::10], tsd3.values)

        # case2: timestamps of tsd (integers) are not subset of that of tsd2.
        tsd2 = nap.Tsd(t=np.arange(0, 100, 0.3), d=np.random.rand(334))
        tsd3 = tsd.value_from(tsd2, mode=mode)
        # loop over epochs
        for iset in tsd.time_support:
            single_ep_tsd = tsd.restrict(iset)
            single_ep_tsd3 = tsd3.restrict(iset)
            single_ep_tsd2 = tsd2.restrict(iset)
            # extract the indices with searchsorted.
            if mode == "before":
                expected_idx = (
                    np.searchsorted(single_ep_tsd2.t, single_ep_tsd.t, side="right") - 1
                )
                # check that times are actually before
                assert np.all(single_ep_tsd2.t[expected_idx] <= single_ep_tsd3.t)
                # check that subsequent are after
                assert np.all(
                    single_ep_tsd2.t[expected_idx[:-1] + 1] > single_ep_tsd3.t[:-1]
                )
            elif mode == "after":
                expected_idx = np.searchsorted(
                    single_ep_tsd2.t, single_ep_tsd.t, side="left"
                )
                # check that times are actually before
                assert np.all(single_ep_tsd2.t[expected_idx] >= single_ep_tsd3.t)
                # check that subsequent are after
                assert np.all(
                    single_ep_tsd2.t[expected_idx[1:] - 1] < single_ep_tsd3.t[1:]
                )
            else:
                before = (
                    np.searchsorted(single_ep_tsd2.t, single_ep_tsd.t, side="right") - 1
                )
                after = np.searchsorted(single_ep_tsd2.t, single_ep_tsd.t, side="left")
                dt_before = np.abs(tsd2.t[before] - tsd.t)
                dt_after = np.abs(tsd2.t[after] - tsd.t)
                expected_idx = before.copy()
                # by default if equi-distance, it assigned to after.
                expected_idx[dt_after <= dt_before] = after[dt_after <= dt_before]

            np.testing.assert_array_equal(
                single_ep_tsd2.d[expected_idx], single_ep_tsd3.d
            )
            np.testing.assert_array_equal(single_ep_tsd.t, single_ep_tsd3.t)

    def test_value_from_value_error(self, tsd):
        with pytest.raises(
            TypeError,
            match=r"First argument should be an instance of Tsd, TsdFrame or TsdTensor",
        ):
            tsd.value_from(np.arange(10))

    def test_value_from_with_restrict(self, tsd):
        ep = nap.IntervalSet(start=0, end=50, time_units="s")
        tsd2 = nap.Tsd(t=np.arange(0, 100, 0.1), d=np.random.rand(1000))
        tsd3 = tsd.value_from(tsd2, ep)
        assert len(tsd.restrict(ep)) == len(tsd3)
        np.testing.assert_array_almost_equal(
            tsd2.restrict(ep).values[::10], tsd3.values
        )

        tsdframe = nap.TsdFrame(t=np.arange(0, 100, 0.1), d=np.random.rand(1000, 2))
        tsdframe2 = tsd.value_from(tsdframe, ep)
        assert len(tsd.restrict(ep)) == len(tsdframe2)
        np.testing.assert_array_almost_equal(
            tsdframe.restrict(ep).values[::10], tsdframe2.values
        )

    def test_restrict(self, tsd):
        ep = nap.IntervalSet(start=0, end=50)
        assert len(tsd.restrict(ep)) == 51

    def test_restrict_error(self, tsd):
        with pytest.raises(TypeError, match=r"Argument should be IntervalSet"):
            tsd.restrict([0, 1])

    def test_restrict_multiple_epochs(self, tsd):
        ep = nap.IntervalSet(start=[0, 20], end=[10, 30])
        assert len(tsd.restrict(ep)) == 22

    def test_restrict_inherit_time_support(self, tsd):
        ep = nap.IntervalSet(start=0, end=50)
        tsd2 = tsd.restrict(ep)
        np.testing.assert_approx_equal(tsd2.time_support.start[0], ep.start[0])
        np.testing.assert_approx_equal(tsd2.time_support.end[0], ep.end[0])

    def test_get_interval(self, tsd):
        tsd2 = tsd.get(10, 20)
        assert len(tsd2) == 11
        np.testing.assert_array_equal(tsd2.index.values, tsd.index.values[10:21])
        if not isinstance(tsd, nap.Ts):
            np.testing.assert_array_equal(tsd2.values, tsd.values[10:21])

        with pytest.raises(Exception):
            tsd.get(20, 10)

        with pytest.raises(Exception):
            tsd.get(10, [20])

        with pytest.raises(Exception):
            tsd.get([10], 20)

    def test_get_timepoint(self, tsd):
        if not isinstance(tsd, nap.Ts):
            np.testing.assert_array_equal(tsd.get(-1), tsd[0])
            np.testing.assert_array_equal(tsd.get(0), tsd[0])
            np.testing.assert_array_equal(tsd.get(0.1), tsd[0])
            np.testing.assert_array_equal(tsd.get(0.5), tsd[1])
            np.testing.assert_array_equal(tsd.get(0.6), tsd[1])
            np.testing.assert_array_equal(tsd.get(1), tsd[1])
            np.testing.assert_array_equal(tsd.get(1000), tsd[-1])

    def test_dropna(self, tsd):
        if not isinstance(tsd, nap.Ts):
            new_tsd = tsd.dropna()
            np.testing.assert_array_equal(tsd.index.values, new_tsd.index.values)
            np.testing.assert_array_equal(tsd.values, new_tsd.values)

            tmp = np.random.rand(*tsd.shape)
            tmp[0 : len(tmp) // 2] = np.nan
            tsd = tsd.__class__(t=tsd.t, d=tmp)

            new_tsd = tsd.dropna()

            tokeep = np.array([~np.any(np.isnan(tsd[i])) for i in range(len(tsd))])
            np.testing.assert_array_equal(
                tsd.index.values[tokeep], new_tsd.index.values
            )
            np.testing.assert_array_equal(tsd.values[tokeep], new_tsd.values)

            newtsd2 = tsd.restrict(new_tsd.time_support)
            np.testing.assert_array_equal(newtsd2.index.values, new_tsd.index.values)
            np.testing.assert_array_equal(newtsd2.values, new_tsd.values)

            new_tsd = tsd.dropna(update_time_support=False)
            np.testing.assert_array_equal(
                tsd.index.values[tokeep], new_tsd.index.values
            )
            np.testing.assert_array_equal(tsd.values[tokeep], new_tsd.values)
            np.testing.assert_array_equal(new_tsd.time_support, tsd.time_support)

            tsd = tsd.__class__(t=tsd.t, d=np.ones(tsd.shape) * np.nan)
            new_tsd = tsd.dropna()
            assert len(new_tsd) == 0
            assert len(new_tsd.time_support) == 0

    def test_convolve_raise_errors(self, tsd):
        if not isinstance(tsd, nap.Ts):
            with pytest.raises(IOError) as e_info:
                tsd.convolve([1, 2, 3])
            assert (
                str(e_info.value)
                == "Input should be a numpy array (or jax array if pynajax is installed)."
            )

            with pytest.raises(IOError) as e_info:
                tsd.convolve(np.array([]))
            assert str(e_info.value) == "Input array is length 0"

            with pytest.raises(IOError) as e_info:
                tsd.convolve(np.ones(3), trim="a")
            assert (
                str(e_info.value)
                == "Unknow argument. trim should be 'both', 'left' or 'right'."
            )

            with pytest.raises(IOError) as e_info:
                tsd.convolve(np.ones((2, 3, 4)))
            assert str(e_info.value) == "Array should be 1 or 2 dimension."

            with pytest.raises(IOError) as e_info:
                tsd.convolve(np.ones(3), ep=[1, 2, 3, 4])
            assert str(e_info.value) == "ep should be an object of type IntervalSet"

    def test_convolve_1d_kernel(self, tsd):
        array = np.random.randn(10)
        if not isinstance(tsd, nap.Ts):
            tsd2 = tsd.convolve(array)
            tmp = tsd.values.reshape(tsd.shape[0], -1)
            tmp2 = np.zeros_like(tmp)
            for i in range(tmp.shape[-1]):
                tmp2[:, i] = np.convolve(tmp[:, i], array, mode="full")[4:-5]
            np.testing.assert_array_almost_equal(
                tmp2, tsd2.values.reshape(tsd2.shape[0], -1)
            )

            ep = nap.IntervalSet(start=[0, 60], end=[40, 100])
            tsd3 = tsd.convolve(array, ep)

            tmp3 = []
            for i in range(len(ep)):
                tmp2 = tsd.restrict(ep[i]).values
                tmp2 = np.array(tmp2.reshape(tmp2.shape[0], -1))  # for pynajax
                for j in range(tmp2.shape[-1]):
                    tmp2[:, j] = np.convolve(tmp2[:, j], array, mode="full")[4:-5]
                tmp3.append(tmp2)
                np.testing.assert_array_almost_equal(
                    tmp2, tsd3.restrict(ep[i]).values.reshape(tmp2.shape[0], -1)
                )

            # Trim
            for trim, sl in zip(
                ["left", "both", "right"],
                [slice(9, None), slice(4, -5), slice(None, -9)],
            ):
                tsd2 = tsd.convolve(array, trim=trim)
                tmp = tsd.values.reshape(tsd.shape[0], -1)
                tmp2 = np.zeros_like(tmp)
                for i in range(tmp.shape[-1]):
                    tmp2[:, i] = np.convolve(tmp[:, i], array, mode="full")[sl]
                np.testing.assert_array_almost_equal(
                    tmp2, tsd2.values.reshape(tsd2.shape[0], -1)
                )

    def test_convolve_2d_kernel(self, tsd):
        array = np.random.randn(10, 3)
        if not isinstance(tsd, nap.Ts):
            # no epochs
            tsd2 = tsd.convolve(array)
            tmp = tsd.values.reshape(tsd.shape[0], -1)

            output = []

            for i in range(tmp.shape[1]):
                for j in range(array.shape[1]):
                    output.append(
                        np.convolve(tmp[:, i], array[:, j], mode="full")[4:-5]
                    )

            output = np.array(output).T
            np.testing.assert_array_almost_equal(
                output, tsd2.values.reshape(tsd.shape[0], -1)
            )

            # epochs
            ep = nap.IntervalSet(start=[0, 60], end=[40, 100])
            tsd2 = tsd.convolve(array, ep)

            for k in range(len(ep)):
                tmp = tsd.restrict(ep[k])
                tmp2 = tmp.values.reshape(tmp.shape[0], -1)
                output = []
                for i in range(tmp2.shape[1]):
                    for j in range(array.shape[1]):
                        output.append(
                            np.convolve(tmp2[:, i], array[:, j], mode="full")[4:-5]
                        )
                output = np.array(output).T
                np.testing.assert_array_almost_equal(
                    output, tsd2.restrict(ep[k]).values.reshape(tmp.shape[0], -1)
                )

    def test_smooth(self, tsd):
        if not isinstance(tsd, nap.Ts):
            from scipy import signal

            tsd2 = tsd.smooth(1, size_factor=10)

            tmp = tsd.values.reshape(tsd.shape[0], -1)
            tmp2 = np.zeros_like(tmp)
            std = int(tsd.rate * 1)
            M = std * 11
            window = signal.windows.gaussian(M, std=std)
            window = window / window.sum()
            for i in range(tmp.shape[-1]):
                tmp2[:, i] = np.convolve(tmp[:, i], window, mode="full")[
                    M // 2 : 1 - M // 2 - 1
                ]
            np.testing.assert_array_almost_equal(
                tmp2, tsd2.values.reshape(tsd2.shape[0], -1)
            )

            tsd2 = tsd.smooth(1000, time_units="ms")
            np.testing.assert_array_almost_equal(
                tmp2, tsd2.values.reshape(tsd2.shape[0], -1)
            )

            tsd2 = tsd.smooth(1000000, time_units="us")
            np.testing.assert_array_almost_equal(
                tmp2, tsd2.values.reshape(tsd2.shape[0], -1)
            )

            tsd2 = tsd.smooth(1, size_factor=200, norm=False)
            tmp = tsd.values.reshape(tsd.shape[0], -1)
            tmp2 = np.zeros_like(tmp)
            std = int(tsd.rate * 1)
            M = std * 201
            window = signal.windows.gaussian(M, std=std)
            for i in range(tmp.shape[-1]):
                tmp2[:, i] = np.convolve(tmp[:, i], window, mode="full")[
                    M // 2 : 1 - M // 2 - 1
                ]
            np.testing.assert_array_almost_equal(
                tmp2, tsd2.values.reshape(tsd2.shape[0], -1)
            )

            tsd2 = tsd.smooth(1, windowsize=10, norm=False)
            tmp = tsd.values.reshape(tsd.shape[0], -1)
            tmp2 = np.zeros_like(tmp)
            std = int(tsd.rate * 1)
            M = int(tsd.rate * 11)
            window = signal.windows.gaussian(M, std=std)
            for i in range(tmp.shape[-1]):
                tmp2[:, i] = np.convolve(tmp[:, i], window, mode="full")[
                    M // 2 : 1 - M // 2 - 1
                ]
            np.testing.assert_array_almost_equal(
                tmp2, tsd2.values.reshape(tsd2.shape[0], -1)
            )

    def test_smooth_raise_error(self, tsd):
        if not isinstance(tsd, nap.Ts):
            with pytest.raises(IOError) as e_info:
                tsd.smooth("a")
            assert str(e_info.value) == "std should be type int or float"

            with pytest.raises(IOError) as e_info:
                tsd.smooth(1, size_factor="b")
            assert str(e_info.value) == "size_factor should be of type int"

            with pytest.raises(IOError) as e_info:
                tsd.smooth(1, norm=1)
            assert str(e_info.value) == "norm should be of type boolean"

            with pytest.raises(IOError) as e_info:
                tsd.smooth(1, time_units=0)
            assert str(e_info.value) == "time_units should be of type str"

            with pytest.raises(IOError) as e_info:
                tsd.smooth(1, windowsize="a")
            assert str(e_info.value) == "windowsize should be type int or float"

    @pytest.mark.parametrize("down", [2, 3])
    @pytest.mark.parametrize(
        "ep", [nap.IntervalSet(0, 100), nap.IntervalSet([0, 41], [39, 100])]
    )
    @pytest.mark.parametrize("filter_type", ["iir", "fir"])
    @pytest.mark.parametrize("order", [8, 6])
    def test_decimate_vs_scipy(self, tsd, down, ep, filter_type, order):
        if isinstance(tsd, nap.Ts):
            pytest.skip("Ts do not implement decimate...")
        out = tsd.decimate(down, ep=ep, filter_type=filter_type, order=order)
        out_sci = decimate_scipy(tsd, ep, order, down, filter_type)
        np.testing.assert_array_equal(out.d, out_sci)

    @pytest.mark.parametrize("down", [2, 3])
    @pytest.mark.parametrize(
        "ep", [nap.IntervalSet(0, 100), nap.IntervalSet([0, 41], [39, 100])]
    )
    def test_decimate_time_axis(self, tsd, down, ep):
        if isinstance(tsd, nap.Ts):
            pytest.skip("Ts do not implement decimate...")
        out = tsd.decimate(down, ep=ep)
        for iset in ep:
            np.testing.assert_array_equal(
                out.restrict(iset).t, tsd.restrict(iset).t[::down]
            )

    @pytest.mark.parametrize(
        "down, expectation",
        [
            (2, does_not_raise()),
            ("3", pytest.raises(IOError, match="Invalid value for 'down'")),
        ],
    )
    def test_decimate_down_input_error(self, tsd, down, expectation):
        if isinstance(tsd, nap.Ts):
            pytest.skip("Ts do not implement decimate...")
        with expectation:
            tsd.decimate(down)

    @pytest.mark.parametrize(
        "order, expectation",
        [
            (6, does_not_raise()),
            (-1, pytest.raises(IOError, match="Invalid value for 'order'")),
            ("6", pytest.raises(IOError, match="Invalid value for 'order'")),
        ],
    )
    def test_decimate_order_input_error(self, tsd, order, expectation):
        if isinstance(tsd, nap.Ts):
            pytest.skip("Ts do not implement decimate...")
        with expectation:
            tsd.decimate(2, order=order)

    @pytest.mark.parametrize(
        "filter_type, expectation",
        [
            ("fir", does_not_raise()),
            ("iir", does_not_raise()),
            ("invalid", pytest.raises(IOError, match="'filter_type' should be one")),
            (1, pytest.raises(IOError, match="'filter_type' should be one")),
        ],
    )
    def test_decimate_filter_type_error(self, tsd, filter_type, expectation):
        if isinstance(tsd, nap.Ts):
            pytest.skip("Ts do not implement decimate...")
        with expectation:
            tsd.decimate(2, filter_type=filter_type)

    @pytest.mark.parametrize(
        "ep, expectation",
        [
            (nap.IntervalSet(0, 40), does_not_raise()),
            (None, does_not_raise()),
            ([0, 40], pytest.raises(IOError, match="ep should be an object")),
        ],
    )
    def test_decimate_epoch_error(self, tsd, ep, expectation):
        if isinstance(tsd, nap.Ts):
            pytest.skip("Ts do not implement decimate...")
        with expectation:
            tsd.decimate(2, ep=ep)

    @pytest.mark.parametrize(
        "ep, align, padding_value, expectation",
        [
            ([], "start", np.nan, "Argument ep should be of type IntervalSet"),
            (nap.IntervalSet(0, 1), "a", np.nan, "align should be 'start' or 'end'"),
        ],
    )
    def test_to_tensor_runtime_errors(self, tsd, ep, align, padding_value, expectation):
        with pytest.raises(RuntimeError, match=re.escape(expectation)):
            if isinstance(tsd, nap.Ts):
                tsd.trial_count(ep, align, padding_value)
            else:
                tsd.to_trial_tensor(ep, align, padding_value)

    def test_to_tensor(self, tsd):
        if hasattr(tsd, "values"):
            ep = nap.IntervalSet(
                start=np.arange(0, 100, 20),
                end=np.arange(0, 100, 20) + np.arange(0, 10, 2),
            )

            expected = np.ones((len(ep), 9, *tsd.shape[1:])) * np.nan
            for i, k in zip(range(len(ep)), range(2, 10, 2)):
                # expected[i, 0 : k + 1] = np.arange(k * 10, k * 10 + k + 1)
                expected[i, 0 : k + 1] = tsd.get(ep.start[i], ep.end[i]).values

            if expected.ndim > 2:  # Need to move the last axis front
                expected = np.moveaxis(expected, (0, 1), (-2, -1))

            tensor = tsd.to_trial_tensor(ep)
            np.testing.assert_array_almost_equal(tensor, expected)

            tensor = tsd.to_trial_tensor(ep, align="start")
            np.testing.assert_array_almost_equal(tensor, expected)

            tensor = tsd.to_trial_tensor(ep, align="start", padding_value=-1)
            expected[np.isnan(expected)] = -1
            np.testing.assert_array_almost_equal(tensor, expected)

            expected = np.ones((len(ep), 9, *tsd.shape[1:])) * np.nan
            for i, k in zip(range(len(ep)), range(2, 10, 2)):
                expected[i, -k - 1 :] = tsd.get(ep.start[i], ep.end[i]).values

            if expected.ndim > 2:  # Need to move the last axis front
                expected = np.moveaxis(expected, (0, 1), (-2, -1))

            tensor = tsd.to_trial_tensor(ep, align="end")
            np.testing.assert_array_almost_equal(tensor, expected)

            tensor = tsd.to_trial_tensor(ep, align="end", padding_value=-1)
            expected[np.isnan(expected)] = -1
            np.testing.assert_array_almost_equal(tensor, expected)

    @pytest.mark.parametrize(
        "align, expectation",
        [
            ("a", "align should be 'start', 'center' or 'end'"),
        ],
    )
    def test_time_diff_runtime_errors(self, tsd, align, expectation):
        with pytest.raises(RuntimeError, match=re.escape(expectation)):
            tsd.time_diff(align=align)

    @pytest.mark.parametrize(
        "epochs, expectation",
        [
            (nap.IntervalSet(0, 40), does_not_raise()),
            (None, does_not_raise()),
            (
                [0, 40],
                pytest.raises(
                    TypeError, match="epochs should be an object of type IntervalSet"
                ),
            ),
        ],
    )
    def test_time_diff_type_errors(self, tsd, epochs, expectation):
        with expectation:
            tsd.time_diff(epochs=epochs)

    @pytest.mark.parametrize(
        "align, epochs, expected",
        [
            # default arguments
            (None, None, nap.Tsd(d=np.ones(99), t=np.arange(0.5, 99.5))),
            # alignment
            ("end", None, nap.Tsd(d=np.ones(99), t=np.arange(1, 100))),
            ("center", None, nap.Tsd(d=np.ones(99), t=np.arange(0.5, 99.5))),
            ("start", None, nap.Tsd(d=np.ones(99), t=np.arange(0, 99))),
            # empty time support
            (
                "start",
                nap.IntervalSet(start=[], end=[]),
                nap.Tsd(d=[], t=[]),
            ),
            # empty epochs
            (
                "start",
                nap.IntervalSet(start=[10, 50, 100], end=[20, 60, 110]),
                nap.Tsd(d=np.ones(20), t=list(range(10, 20)) + list(range(50, 60))),
            ),
            # single epoch
            (
                "start",
                nap.IntervalSet(start=[10, 30]),
                nap.Tsd(d=np.ones(20), t=list(range(10, 30))),
            ),
            # single point in epochs
            (
                "start",
                nap.IntervalSet(start=[10, 50, 99], end=[20, 60, 100]),
                nap.Tsd(d=np.ones(20), t=list(range(10, 20)) + list(range(50, 60))),
            ),
            # two points in epochs
            (
                "start",
                nap.IntervalSet(start=[10, 50, 98], end=[20, 60, 100]),
                nap.Tsd(
                    d=np.ones(21),
                    t=np.concatenate([np.arange(10, 20), np.arange(50, 60), [98]]),
                ),
            ),
        ],
    )
    def test_time_diff(self, tsd, align, epochs, expected):
        if align is None:
            actual = tsd.time_diff(epochs=epochs)
        else:
            actual = tsd.time_diff(align=align, epochs=epochs)
        np.testing.assert_array_almost_equal(actual.values, expected.values)
        np.testing.assert_array_almost_equal(actual.index, expected.index)


####################################################
# Test for tsd
####################################################
@pytest.mark.parametrize(
    "tsd",
    [
        nap.Tsd(t=np.arange(100), d=np.random.rand(100), time_units="s"),
    ],
)
class TestTsd:
    @pytest.mark.parametrize("delta_ep", [(1, -1), (-1, -1), (1, 1)])
    def test_bin_average_time_support(self, tsd, delta_ep):
        ep = nap.IntervalSet(
            tsd.time_support.start[0] + delta_ep[0],
            tsd.time_support.end[0] + delta_ep[1],
        )
        out = tsd.bin_average(0.1, ep=ep)
        assert np.all(out.time_support == ep)

    @pytest.mark.parametrize("delta_ep", [(1, -1), (-1, -1), (1, 1)])
    def test_convolve_time_support(self, tsd, delta_ep):
        ep = nap.IntervalSet(
            tsd.time_support.start[0] + delta_ep[0],
            tsd.time_support.end[0] + delta_ep[1],
        )
        out = tsd.convolve(np.ones(10), ep=ep)
        assert np.all(out.time_support == ep)

    @pytest.mark.parametrize("delta_ep", [(1, -1), (-1, -1), (1, 1)])
    def test_interpolate_time_support(self, tsd, delta_ep):
        ep = nap.IntervalSet(
            tsd.time_support.start[0] + delta_ep[0],
            tsd.time_support.end[0] + delta_ep[1],
        )
        ts = nap.Ts(np.linspace(0, 10, 20))
        out = tsd.interpolate(ts, ep=ep)
        assert np.all(out.time_support == ep)

    def test_as_series(self, tsd):
        assert isinstance(tsd.as_series(), pd.Series)

    def test__getitems__(self, tsd):
        a = tsd[0:10]
        b = nap.Tsd(t=tsd.index[0:10], d=tsd.values[0:10])
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(a.index, b.index)
        np.testing.assert_array_almost_equal(a.values, b.values)
        np.testing.assert_array_almost_equal(a.time_support, tsd.time_support)

    def test_count(self, tsd):
        count = tsd.count(1)
        assert len(count) == 99
        np.testing.assert_array_almost_equal(count.index, np.arange(0.5, 99, 1))

        count = tsd.count(bin_size=1)
        assert len(count) == 99
        np.testing.assert_array_almost_equal(count.index, np.arange(0.5, 99, 1))

        count = tsd.count(bin_size=1, dtype=np.int16)
        assert len(count) == 99
        assert count.dtype == np.dtype(np.int16)

    def test_count_time_units(self, tsd):
        for b, tu in zip([1, 1e3, 1e6], ["s", "ms", "us"]):
            count = tsd.count(b, time_units=tu)
            assert len(count) == 99
            np.testing.assert_array_almost_equal(count.index, np.arange(0.5, 99, 1))

            count = tsd.count(b, time_units=tu)
            assert len(count) == 99
            np.testing.assert_array_almost_equal(count.index, np.arange(0.5, 99, 1))

    def test_count_with_ep(self, tsd):
        ep = nap.IntervalSet(start=0, end=100)
        count = tsd.count(1, ep)
        assert len(count) == 100
        np.testing.assert_array_almost_equal(count.values, np.ones(100))

        count = tsd.count(1, ep=ep)
        assert len(count) == 100
        np.testing.assert_array_almost_equal(count.values, np.ones(100))

    def test_count_with_ep_only(self, tsd):
        ep = nap.IntervalSet(start=0, end=100)
        count = tsd.count(ep=ep)
        assert len(count) == 1
        np.testing.assert_array_almost_equal(count.values, np.array([100]))

        count = tsd.count(ep=ep)
        assert len(count) == 1
        np.testing.assert_array_almost_equal(count.values, np.array([100]))

        count = tsd.count()
        assert len(count) == 1
        np.testing.assert_array_almost_equal(count.values, np.array([100]))

    def test_count_errors(self, tsd):
        with pytest.raises(
            TypeError, match=r"bin_size argument should be float or int."
        ):
            tsd.count(bin_size={})

        with pytest.raises(
            TypeError, match=r"ep argument should be of type IntervalSet"
        ):
            tsd.count(ep={})

        with pytest.raises(
            ValueError, match=r"time_units argument should be 's', 'ms' or 'us'."
        ):
            tsd.count(bin_size=1, time_units={})

    def test_bin_average(self, tsd):
        meantsd = tsd.bin_average(10)
        assert len(meantsd) == 10
        np.testing.assert_array_almost_equal(meantsd.index, np.arange(5, 100, 10))
        bins = np.arange(tsd.time_support.start[0], tsd.time_support.end[0] + 1, 10)
        idx = np.digitize(tsd.index, bins)
        tmp = np.array([np.mean(tsd.values[idx == i]) for i in np.unique(idx)])
        np.testing.assert_array_almost_equal(meantsd.values, tmp)

    def test_bin_average_with_ep(self, tsd):
        ep = nap.IntervalSet(start=0, end=40)
        meantsd = tsd.bin_average(10, ep)
        assert len(meantsd) == 4
        np.testing.assert_array_almost_equal(meantsd.index, np.arange(5, 40, 10))
        bins = np.arange(ep.start[0], ep.end[0] + 10, 10)
        # tsd = tsd.restrict(ep)
        idx = np.digitize(tsd.index, bins)
        tmp = np.array([np.mean(tsd.values[idx == i]) for i in np.unique(idx)])
        np.testing.assert_array_almost_equal(meantsd.values, tmp[0:-1])

    def test_threshold(self, tsd):
        thrs = tsd.threshold(0.5, "above")
        assert len(thrs) == int(np.sum(tsd.values > 0.5))
        thrs = tsd.threshold(0.5, "below")
        assert len(thrs) == int(np.sum(tsd.values < 0.5))
        thrs = tsd.threshold(0.5, "aboveequal")
        assert len(thrs) == int(np.sum(tsd.values >= 0.5))
        thrs = tsd.threshold(0.5, "belowequal")
        assert len(thrs) == int(np.sum(tsd.values <= 0.5))

    def test_threshold_time_support(self, tsd):
        thrs = tsd.threshold(0.5, "above")
        time_support = thrs.time_support
        thrs2 = tsd.restrict(time_support)
        assert len(thrs2) == np.sum(tsd.values > 0.5)

        with pytest.raises(ValueError) as e_info:
            tsd.threshold(0.5, "bla")
        assert str(
            e_info.value
        ) == "Method {} for thresholding is not accepted.".format("bla")

    def test_slice_with_int_tsd(self, tsd):
        tsd = tsd.copy()
        array_slice = np.arange(10, dtype=int)
        tsd_index = nap.Tsd(t=tsd.index[: len(array_slice)], d=array_slice)

        with pytest.raises(ValueError) as e:
            indexed = tsd[tsd_index]
        assert (
            str(e.value) == "When indexing with a Tsd, it must contain boolean values"
        )

        with pytest.raises(ValueError) as e:
            tsd[tsd_index] = 0
        assert (
            str(e.value) == "When indexing with a Tsd, it must contain boolean values"
        )

    def test_slice_with_bool_tsd(self, tsd):
        thr = 0.5
        tsd_index = tsd > thr
        raw_values = tsd.values
        np_indexed_vals = raw_values[tsd_index.values]
        indexed = tsd[tsd_index]

        assert isinstance(indexed, nap.Tsd)
        np.testing.assert_array_almost_equal(indexed.values, np_indexed_vals)
        if nap.nap_config.backend != "jax":
            tsd[tsd_index] = 0
            np.testing.assert_array_almost_equal(tsd.values[tsd_index.values], 0)

    def test_slice_with_bool_tsd(self, tsd):
        thr = 0.5
        tsd_index = tsd > thr
        raw_values = tsd.values
        np_indexed_vals = raw_values[tsd_index.values]
        indexed = tsd[tsd_index]

        assert isinstance(indexed, nap.Tsd)
        np.testing.assert_array_almost_equal(indexed.values, np_indexed_vals)
        if nap.nap_config.backend != "jax":
            tsd[tsd_index] = 0
            np.testing.assert_array_almost_equal(tsd.values[tsd_index.values], 0)

    def test_data(self, tsd):
        np.testing.assert_array_almost_equal(tsd.values, tsd.data())

    def test_to_tsgroup(self, tsd):
        t = []
        d = []
        group = {}
        for i in range(3):
            t.append(np.sort(np.random.rand(10) * 100))
            d.append(np.ones(10) * i)
            group[i] = nap.Ts(t=t[-1])

        times = np.array(t).flatten()
        data = np.array(d).flatten()
        idx = np.argsort(times)
        times = times[idx]
        data = data[idx]

        tsd = nap.Tsd(t=times, d=data)

        tsgroup = tsd.to_tsgroup()

        assert len(tsgroup) == 3
        np.testing.assert_array_almost_equal(np.arange(3), tsgroup.index)
        for i in range(3):
            np.testing.assert_array_almost_equal(tsgroup[i].index, t[i])

    def test_save_npz(self, tsd):
        with pytest.raises(TypeError) as e:
            tsd.save(dict)

        with pytest.raises(RuntimeError) as e:
            tsd.save("./")
        assert str(e.value) == "Invalid filename input. {} is directory.".format(
            Path("./").resolve()
        )

        fake_path = "./fake/path"
        with pytest.raises(RuntimeError) as e:
            tsd.save(fake_path + "/file.npz")
        assert str(e.value) == "Path {} does not exist.".format(
            Path(fake_path).resolve()
        )

        tsd.save("tsd.npz")
        assert "tsd.npz" in [f.name for f in Path(".").iterdir()]

        tsd.save("tsd2")
        assert "tsd2.npz" in [f.name for f in Path(".").iterdir()]

        file = np.load("tsd.npz")

        keys = list(file.keys())
        assert "t" in keys
        assert "d" in keys
        assert "start" in keys
        assert "end" in keys

        np.testing.assert_array_almost_equal(file["t"], tsd.index)
        np.testing.assert_array_almost_equal(file["d"], tsd.values)
        np.testing.assert_array_almost_equal(file["start"], tsd.time_support.start)
        np.testing.assert_array_almost_equal(file["end"], tsd.time_support.end)

        Path("tsd.npz").unlink()
        Path("tsd2.npz").unlink()

    def test_interpolate(self, tsd):
        y = np.arange(0, 1001)

        tsd = nap.Tsd(t=np.arange(0, 101), d=y[0::10])

        # Ts
        ts = nap.Ts(t=y / 10)
        tsd2 = tsd.interpolate(ts)
        np.testing.assert_array_almost_equal(tsd2.values, y)

        # Tsd
        ts = nap.Tsd(t=y / 10, d=np.zeros_like(y))
        tsd2 = tsd.interpolate(ts)
        np.testing.assert_array_almost_equal(tsd2.values, y)

        # TsdFrame
        ts = nap.TsdFrame(t=y / 10, d=np.zeros((len(y), 2)))
        tsd2 = tsd.interpolate(ts)
        np.testing.assert_array_almost_equal(tsd2.values, y)

        with pytest.raises(IOError) as e:
            tsd.interpolate([0, 1, 2])
        assert (
            str(e.value)
            == "First argument should be an instance of Ts, Tsd, TsdFrame or TsdTensor"
        )

        with pytest.raises(IOError) as e:
            tsd.interpolate(ts, left="a")
        assert str(e.value) == "Argument left should be of type float or int"

        with pytest.raises(IOError) as e:
            tsd.interpolate(ts, right="a")
        assert str(e.value) == "Argument right should be of type float or int"

        with pytest.raises(IOError) as e:
            tsd.interpolate(ts, ep=[1, 2, 3, 4])
        assert str(e.value) == "ep should be an object of type IntervalSet"

        # Right left
        ep = nap.IntervalSet(start=0, end=5)
        tsd = nap.Tsd(t=np.arange(1, 4), d=np.arange(3), time_support=ep)
        ts = nap.Ts(t=np.arange(0, 5))
        tsd2 = tsd.interpolate(ts, left=1234)
        assert float(tsd2.values[0]) == 1234.0
        tsd2 = tsd.interpolate(ts, right=4321)
        assert float(tsd2.values[-1]) == 4321.0

    def test_interpolate_with_ep(self, tsd):
        y = np.arange(0, 1001)
        tsd = nap.Tsd(t=np.arange(0, 101), d=y[0::10])
        ts = nap.Ts(t=y / 10)
        ep = nap.IntervalSet(start=np.arange(0, 100, 20), end=np.arange(10, 110, 20))
        tsd2 = tsd.interpolate(ts, ep)
        tmp = ts.restrict(ep).index * 10
        np.testing.assert_array_almost_equal(tmp, tsd2.values)

        # Empty ep
        ep = nap.IntervalSet(start=200, end=300)
        tsd2 = tsd.interpolate(ts, ep)
        assert len(tsd2) == 0

    def test_interpolate_with_single_time_point(self, tsd):
        y = np.array([0.5])
        tsd = nap.Tsd(t=np.arange(0, 101), d=np.arange(0, 101))
        ts = nap.Ts(t=y)
        tsd2 = tsd.interpolate(ts)
        np.testing.assert_array_almost_equal(y, tsd2.values)

    def test_derivative(self, tsd):
        times = np.arange(0, 10, 0.001)
        data = np.sin(times)
        tsd = nap.Tsd(t=times, d=data)
        expected = np.cos(times)  # Derivative of sin(x) is cos(x)
        derivative = tsd.derivative()
        np.testing.assert_allclose(derivative.values, expected, atol=1e-3)

    def test_derivative_with_ep(self, tsd):
        times = np.arange(0, 100)
        data = np.tile(np.arange(0, 10), 10)
        ep = nap.IntervalSet(start=np.arange(0, 100, 10), end=np.arange(10, 110, 10))
        expected = np.ones(len(times))  # Derivative should be 1

        # With time support
        tsd = nap.Tsd(t=times, d=data, time_support=ep)
        derivative = tsd.derivative()
        assert np.all(derivative == expected)

        # With ep parameter
        tsd = nap.Tsd(t=times, d=data)
        derivative = tsd.derivative(ep=ep)
        assert np.all(derivative == expected)


####################################################
# Test for tsdframe
####################################################
@pytest.mark.parametrize(
    "tsdframe",
    [
        nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 1), time_units="s"),
        nap.TsdFrame(
            t=np.arange(1), d=np.random.rand(1, 3), time_support=nap.IntervalSet(0, 2)
        ),
        nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 3), time_units="s"),
        nap.TsdFrame(
            t=np.arange(100),
            d=np.random.rand(100, 3),
            time_units="s",
            columns=["a", "b", "c"],
        ),
        nap.TsdFrame(
            t=np.arange(100),
            d=np.random.rand(100, 3),
            time_units="s",
            metadata={"l1": np.arange(3), "l2": ["x", "x", "y"]},
        ),
        nap.TsdFrame(
            t=np.arange(100),
            d=np.random.rand(100, 3),
            time_units="s",
            columns=["a", "b", "c"],
            metadata={"l1": np.arange(3), "l2": ["x", "x", "y"]},
        ),
    ],
)
class TestTsdFrame:
    @pytest.mark.parametrize("delta_ep", [(1, -1), (-1, -1), (1, 1)])
    def test_bin_average_time_support(self, tsdframe, delta_ep):
        ep = nap.IntervalSet(
            tsdframe.time_support.start[0] + delta_ep[0],
            tsdframe.time_support.end[0] + delta_ep[1],
        )
        out = tsdframe.bin_average(0.1, ep=ep)
        assert np.all(out.time_support == ep)

    @pytest.mark.parametrize("delta_ep", [(1, -1), (-1, -1), (1, 1)])
    def test_convolve_time_support(self, tsdframe, delta_ep):
        if len(tsdframe) > 1:
            ep = nap.IntervalSet(
                tsdframe.time_support.start[0] + delta_ep[0],
                tsdframe.time_support.end[0] + delta_ep[1],
            )
            out = tsdframe.convolve(np.ones(10), ep=ep)
            assert np.all(out.time_support == ep)

    @pytest.mark.parametrize("delta_ep", [(1, -1), (-1, -1), (1, 1)])
    def test_interpolate_time_support(self, tsdframe, delta_ep):
        ep = nap.IntervalSet(
            tsdframe.time_support.start[0] + delta_ep[0],
            tsdframe.time_support.end[0] + delta_ep[1],
        )
        ts = nap.Ts(np.linspace(0, 10, 20))
        out = tsdframe.interpolate(ts, ep=ep)
        assert np.all(out.time_support == ep)

    def test_as_dataframe(self, tsdframe):
        assert isinstance(tsdframe.as_dataframe(), pd.DataFrame)

    def test_copy(self, tsdframe):
        tscopy = tsdframe.copy()
        np.testing.assert_array_almost_equal(tscopy.values, tsdframe.values)
        assert np.all(tscopy.columns == tsdframe.columns)
        if len(tsdframe.metadata_columns):
            pd.testing.assert_frame_equal(tscopy.metadata, tsdframe.metadata)

    @pytest.mark.parametrize(
        "index, nap_type",
        [
            (0, nap.Tsd),
            ([0, 2], nap.TsdFrame),
            ([False, True, False], nap.TsdFrame),
            ([False, True, True], nap.TsdFrame),
        ],
    )
    def test_horizontal_slicing(self, tsdframe, index, nap_type):
        index = index if isinstance(index, int) else index[: tsdframe.shape[1]]
        assert isinstance(tsdframe[:, index], nap_type)
        np.testing.assert_array_almost_equal(
            tsdframe[:, index].values, tsdframe.values[:, index]
        )
        assert isinstance(tsdframe[:, index].time_support, nap.IntervalSet)
        np.testing.assert_array_almost_equal(
            tsdframe.time_support, tsdframe[:, index].time_support
        )

        if isinstance(tsdframe[:, index], nap.TsdFrame) and len(
            tsdframe.metadata_columns
        ):
            assert np.all(
                tsdframe[:, index].metadata_columns == tsdframe.metadata_columns
            )
            assert np.all(
                tsdframe[:, index].metadata_index == tsdframe.metadata_index[index]
            )

    @pytest.mark.parametrize(
        "index",
        [
            0,
            slice(0, 10),
            [0, 2],  # not jax compatible
            np.hstack([np.zeros(10, bool), True, np.zeros(89, bool)]),
            np.hstack([np.zeros(10, bool), True, True, True, np.zeros(87, bool)]),
        ],
    )
    def test_vertical_slicing(self, tsdframe, index):
        if isinstance(index, int):
            assert (not isinstance(tsdframe[index], nap.TsdFrame)) and is_array_like(
                tsdframe[index]
            )
        elif len(tsdframe) > 1:
            #     # jax and numpy compatible check
            #     assert (not isinstance(tsdframe[index], nap.TsdFrame)) and is_array_like(
            #         tsdframe[index]
            #     )
            # else:
            if isinstance(index, list) and nap.nap_config.backend == "jax":
                index = np.array(index)
            assert isinstance(tsdframe[index], nap.TsdFrame)

        if len(tsdframe) > 1:
            output = tsdframe[index]
            if isinstance(output, nap.TsdFrame):
                if len(output == 1):
                    # use ravel to ignore shape mismatch
                    np.testing.assert_array_almost_equal(
                        tsdframe.values[index].ravel(), output.values.ravel()
                    )
                else:
                    np.testing.assert_array_almost_equal(
                        tsdframe.values[index], output.values
                    )
                assert isinstance(output.time_support, nap.IntervalSet)
                np.testing.assert_array_almost_equal(
                    output.time_support, tsdframe.time_support
                )
                if len(tsdframe.metadata_columns):
                    assert np.all(output.metadata_columns == tsdframe.metadata_columns)
                    assert np.all(output.metadata_index == tsdframe.metadata_index)

    @pytest.mark.parametrize(
        "row",
        [
            # 0,
            [0, 2],
            slice(20, 30),
            np.hstack([np.zeros(10, bool), True, True, True, np.zeros(87, bool)]),
            np.hstack([np.zeros(10, bool), True, np.zeros(89, bool)]),
        ],
    )
    @pytest.mark.parametrize(
        "col, expected",
        [
            (0, nap.Tsd),
            ([1], nap.TsdFrame),
            ([0, 1], nap.TsdFrame),
            (slice(0, 2), nap.TsdFrame),
            ([True, False, True], nap.TsdFrame),
            ([False, True, False], nap.TsdFrame),
        ],
    )
    def test_vert_and_horz_slicing(self, tsdframe, row, col, expected):
        if len(tsdframe) > 1:
            if tsdframe.shape[1] == 1:
                if isinstance(col, list) and isinstance(col[0], int):
                    col = [0]
                elif isinstance(col, list) and isinstance(col[0], bool):
                    col = [col[0]]

            # get details about row index
            row_array = isinstance(row, (list, np.ndarray))
            if row_array and isinstance(row[0], (bool, np.bool_)):
                row_len = np.sum(row)
            elif row_array and isinstance(row[0], Number):
                row_len = len(row)
            else:
                row_len = 1

            # get details about column index
            col_array = isinstance(col, (list, np.ndarray))
            if col_array and isinstance(col[0], (bool, np.bool_)):
                col_len = np.sum(col)
            elif col_array and isinstance(col[0], Number):
                col_len = len(col)
            else:
                col_len = 1

            # this is when shape mismatch is a problem
            if (row_len > 1) and (col_len > 1):
                if row_len == col_len:
                    assert isinstance(tsdframe[row, col], nap.Tsd)
                else:
                    # shape mismatch
                    # Numpy: IndexError, JAX: ValueError
                    # (numpy | jax error messages)
                    with pytest.raises(
                        (IndexError, ValueError),
                        match="shape mismatch|Incompatible shapes for ",
                    ):
                        tsdframe[row, col]

            elif isinstance(row, Number) and isinstance(col, Number):
                assert isinstance(tsdframe[row, col], Number)

            else:
                assert isinstance(tsdframe[row, col], expected)

                output = tsdframe[row, col]

                if isinstance(output, nap.TsdFrame):
                    if len(tsdframe[row, col] == 1):
                        # use ravel to ignore shape mismatch
                        np.testing.assert_array_almost_equal(
                            tsdframe.values[row, col].ravel(),
                            tsdframe[row, col].values.ravel(),
                        )
                    else:
                        np.testing.assert_array_almost_equal(
                            tsdframe.values[row, col], tsdframe[row, col].values
                        )
                    assert isinstance(tsdframe[row, col].time_support, nap.IntervalSet)
                    np.testing.assert_array_almost_equal(
                        tsdframe[row, col].time_support, tsdframe.time_support
                    )
                    if isinstance(tsdframe[row, col], nap.TsdFrame) and len(
                        tsdframe[row, col].metadata_columns
                    ):
                        assert np.all(
                            tsdframe[row, col].metadata_columns
                            == tsdframe.metadata_columns
                        )
                        assert np.all(
                            tsdframe[row, col].metadata_index
                            == tsdframe.metadata_index[col]
                        )
                else:
                    np.testing.assert_array_almost_equal(
                        output, tsdframe.values[row, col]
                    )

    @pytest.mark.parametrize("index", [0, [0, 2]])
    def test_str_indexing(self, tsdframe, index):
        columns = tsdframe.columns

        if isinstance(columns[0], str):
            np.testing.assert_array_almost_equal(
                tsdframe.values[:, index], tsdframe[columns[index]].values
            )
            if isinstance(tsdframe[:, index], nap.TsdFrame) and len(
                tsdframe.metadata_columns
            ):
                assert np.all(
                    tsdframe[columns[index]].metadata_columns
                    == tsdframe.metadata_columns
                )
                assert np.all(
                    tsdframe[columns[index]].metadata_index
                    == tsdframe.metadata_index[index]
                )

        if len(tsdframe.metadata_columns):
            for col in tsdframe.metadata_columns:
                assert np.all(tsdframe[col] == tsdframe._metadata[col])

        with pytest.raises(Exception):
            tsdframe["d"]

        with pytest.raises(Exception):
            tsdframe[["d", "e"]]

    def test_operators(self, tsdframe):
        v = tsdframe.values

        a = tsdframe + 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v + 0.5))

        a = tsdframe - 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v - 0.5))

        a = tsdframe * 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v * 0.5))

        a = tsdframe / 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v / 0.5))

        a = tsdframe // 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v // 0.5))

        a = tsdframe % 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v % 0.5))

        a = tsdframe**0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        np.testing.assert_array_almost_equal(a.values, v**0.5)

        a = tsdframe > 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v > 0.5))

        a = tsdframe >= 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v >= 0.5))

        a = tsdframe < 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v < 0.5))

        a = tsdframe <= 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v <= 0.5))

        tsdframe = nap.TsdFrame(
            t=np.arange(10), d=np.tile(np.arange(10)[:, np.newaxis], 3)
        )
        v = tsdframe.values
        a = tsdframe == 5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v == 5))

        a = tsdframe != 5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v != 5))

    def test_data(self, tsdframe):
        np.testing.assert_array_almost_equal(tsdframe.values, tsdframe.data())

    def test_bin_average(self, tsdframe):
        meantsd = tsdframe.bin_average(10)
        if len(tsdframe) == 100:
            assert len(meantsd) == 10
            np.testing.assert_array_almost_equal(meantsd.index, np.arange(5, 100, 10))
            bins = np.arange(
                tsdframe.time_support.start[0], tsdframe.time_support.end[0] + 1, 10
            )
            tmp = (
                tsdframe.as_dataframe()
                .groupby(np.digitize(tsdframe.index, bins))
                .mean()
            )
            np.testing.assert_array_almost_equal(meantsd.values, tmp.values)

    def test_bin_average_with_ep(self, tsdframe):
        if len(tsdframe) > 1:
            ep = nap.IntervalSet(start=0, end=40)
            meantsd = tsdframe.bin_average(10, ep)
            assert len(meantsd) == 4
            np.testing.assert_array_almost_equal(meantsd.index, np.arange(5, 40, 10))
            bins = np.arange(ep.start[0], ep.end[0] + 10, 10)
            tsdframe = tsdframe.restrict(ep)
            tmp = (
                tsdframe.as_dataframe()
                .groupby(np.digitize(tsdframe.index, bins))
                .mean()
            )
            np.testing.assert_array_almost_equal(
                meantsd.values, tmp.loc[np.arange(1, 5)].values
            )

    def test_save_npz(self, tsdframe):
        with pytest.raises(TypeError) as e:
            tsdframe.save(dict)

        with pytest.raises(RuntimeError) as e:
            tsdframe.save("./")
        assert str(e.value) == "Invalid filename input. {} is directory.".format(
            Path("./").resolve()
        )

        fake_path = "./fake/path"
        with pytest.raises(RuntimeError) as e:
            tsdframe.save(fake_path + "/file.npz")
        assert str(e.value) == "Path {} does not exist.".format(
            Path(fake_path).resolve()
        )

        tsdframe.save("tsdframe.npz")
        assert "tsdframe.npz" in [f.name for f in Path(".").iterdir()]

        tsdframe.save("tsdframe2")
        assert "tsdframe2.npz" in [f.name for f in Path(".").iterdir()]

        file = np.load("tsdframe.npz", allow_pickle=True)

        keys = list(file.keys())
        assert "t" in keys
        assert "d" in keys
        assert "start" in keys
        assert "end" in keys
        assert "columns" in keys

        np.testing.assert_array_almost_equal(file["t"], tsdframe.index)
        np.testing.assert_array_almost_equal(file["d"], tsdframe.values)
        np.testing.assert_array_almost_equal(file["start"], tsdframe.time_support.start)
        np.testing.assert_array_almost_equal(file["end"], tsdframe.time_support.end)

        if isinstance(tsdframe.columns[0], str):
            assert np.all(file["columns"] == tsdframe.columns)
        else:
            np.testing.assert_array_almost_equal(file["columns"], tsdframe.columns)

        if len(tsdframe.metadata_columns):
            assert "_metadata" in keys
            metadata = file["_metadata"].item()
            assert metadata.keys() == tsdframe._metadata.keys()
            for key in metadata.keys():
                assert np.all(metadata[key] == tsdframe._metadata[key])

        Path("tsdframe.npz").unlink()
        Path("tsdframe2.npz").unlink()

    def test_interpolate(self, tsdframe):
        y = np.arange(0, 1001)
        data_stack = np.stack(
            [
                np.arange(0, 1001),
            ]
            * 4
        ).T

        tsdframe = nap.TsdFrame(t=np.arange(0, 101), d=data_stack[0::10, :])

        # Ts
        ts = nap.Ts(t=y / 10)
        tsdframe2 = tsdframe.interpolate(ts)
        np.testing.assert_array_almost_equal(tsdframe2.values, data_stack)

        # Tsd
        ts = nap.Tsd(t=y / 10, d=np.zeros_like(y))
        tsdframe2 = tsdframe.interpolate(ts)
        np.testing.assert_array_almost_equal(tsdframe2.values, data_stack)

        # TsdFrame
        ts = nap.TsdFrame(t=y / 10, d=np.zeros((len(y), 2)))
        tsdframe2 = tsdframe.interpolate(ts)
        np.testing.assert_array_almost_equal(tsdframe2.values, data_stack)

        with pytest.raises(IOError) as e:
            tsdframe.interpolate([0, 1, 2])
        assert (
            str(e.value)
            == "First argument should be an instance of Ts, Tsd, TsdFrame or TsdTensor"
        )

        # Right left
        ep = nap.IntervalSet(start=0, end=5)
        tsdframe = nap.Tsd(t=np.arange(1, 4), d=np.arange(3), time_support=ep)
        ts = nap.Ts(t=np.arange(0, 5))
        tsdframe2 = tsdframe.interpolate(ts, left=1234)
        assert float(tsdframe2.values[0]) == 1234.0
        tsdframe2 = tsdframe.interpolate(ts, right=4321)
        assert float(tsdframe2.values[-1]) == 4321.0

    def test_interpolate_with_ep(self, tsdframe):
        y = np.arange(0, 1001)
        data_stack = np.stack(
            [
                np.arange(0, 1001),
            ]
            * 4
        ).T

        tsdframe = nap.TsdFrame(t=np.arange(0, 101), d=data_stack[0::10, :])
        ts = nap.Ts(t=y / 10)
        ep = nap.IntervalSet(start=np.arange(0, 100, 20), end=np.arange(10, 110, 20))
        tsdframe2 = tsdframe.interpolate(ts, ep)
        tmp = ts.restrict(ep).index * 10
        print(tmp, tsdframe2.values)
        print(tmp.shape, tsdframe2.values.shape)
        print(tmp.mean(0), tsdframe2.values.mean(0))
        np.testing.assert_array_almost_equal(tmp, tsdframe2.values[:, 0])

        # Empty ep
        ep = nap.IntervalSet(start=200, end=300)
        tsdframe2 = tsdframe.interpolate(ts, ep)
        assert len(tsdframe2) == 0

    def test_derivative(self, tsdframe):
        # Test with known derivatives (sine and cosine)
        times = np.arange(0, 10, 0.001)
        data = np.column_stack([np.sin(times), np.cos(times)])
        tsdframe = nap.TsdFrame(t=times, d=data)
        expected_derivative = np.column_stack([np.cos(times), -1 * np.sin(times)])
        derivative = tsdframe.derivative()
        np.testing.assert_allclose(derivative.values, expected_derivative, atol=1e-3)

    def test_derivative_with_ep(self, tsdframe):
        times = np.arange(0, 10)
        data = np.array([[i, -i] for i in range(10)])
        ep = nap.IntervalSet(start=[1, 6], end=[4, 10])
        expected = np.array([[1, -1] for _ in range(8)])

        # With time support
        tsdframe = nap.TsdFrame(t=times, d=data, time_support=ep)
        derivative = tsdframe.derivative()
        assert np.all(derivative.values == expected)

        # Test with ep parameter
        tsdframe = nap.TsdFrame(t=times, d=data)
        derivative = tsdframe.derivative(ep=ep)
        assert np.all(derivative.values == expected)

    def test_convolve_keep_columns(self, tsdframe):
        array = np.random.randn(10)
        tsdframe = nap.TsdFrame(
            t=np.arange(100),
            d=np.random.rand(100, 3),
            time_units="s",
            columns=["a", "b", "c"],
        )
        tsd2 = tsdframe.convolve(array)

        assert isinstance(tsd2, nap.TsdFrame)
        np.testing.assert_array_equal(tsd2.columns, tsdframe.columns)

    # def test_deprecation_warning(self, tsdframe):
    #     columns = tsdframe.columns
    #     # warning using loc
    #     with pytest.warns(DeprecationWarning):
    #         tsdframe.loc[columns[0]]
    #     if isinstance(columns[0], str):
    #         # suppressed warning with getitem, which implicitly uses loc
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("error")
    #             tsdframe[columns[0]]


####################################################
# Test for ts
####################################################
@pytest.mark.parametrize(
    "ts",
    [
        nap.Ts(t=np.arange(100), time_units="s"),
    ],
)
class TestTs:
    def test_save_npz(self, ts):
        with pytest.raises(TypeError) as e:
            ts.save(dict)

        with pytest.raises(RuntimeError) as e:
            ts.save("./")
        assert str(e.value) == "Invalid filename input. {} is directory.".format(
            Path("./").resolve()
        )

        fake_path = "./fake/path"
        with pytest.raises(RuntimeError) as e:
            ts.save(fake_path + "/file.npz")
        assert str(e.value) == "Path {} does not exist.".format(
            Path(fake_path).resolve()
        )

        ts.save("ts.npz")
        assert "ts.npz" in [f.name for f in Path(".").iterdir()]

        ts.save("ts2")
        assert "ts2.npz" in [f.name for f in Path(".").iterdir()]

        file = np.load("ts.npz")

        keys = list(file.keys())
        assert "t" in keys
        assert "start" in keys
        assert "end" in keys

        np.testing.assert_array_almost_equal(file["t"], ts.index)
        np.testing.assert_array_almost_equal(file["start"], ts.time_support.start)
        np.testing.assert_array_almost_equal(file["end"], ts.time_support.end)

        Path("ts.npz").unlink()
        Path("ts2.npz").unlink()

    def test_fillna(self, ts):
        with pytest.raises(AssertionError):
            ts.fillna([1])

        tsd = ts.fillna(0)

        assert isinstance(tsd, nap.Tsd)

        np.testing.assert_array_equal(np.zeros(len(ts)), tsd.values)
        np.testing.assert_array_equal(ts.index.values, tsd.index.values)

    def test_set(self, ts):
        ts.__setitem__(0, 1)

    def test_get(self, ts):
        assert isinstance(ts[0], nap.Ts)
        assert len(ts[0]) == 1
        assert len(ts[0:10]) == 10
        assert len(ts[[0, 2, 5]]) == 3

        np.testing.assert_array_equal(ts[[0, 2, 5]].index, np.array([0, 2, 5]))

        assert len(ts[0:10, 0]) == 10

    def test_count(self, ts):
        count = ts.count(1)
        assert len(count) == 99
        np.testing.assert_array_almost_equal(count.index, np.arange(0.5, 99, 1))

        count = ts.count(bin_size=1)
        assert len(count) == 99
        np.testing.assert_array_almost_equal(count.index, np.arange(0.5, 99, 1))

        count = ts.count(bin_size=1, dtype=np.int16)
        assert len(count) == 99
        assert count.dtype == np.dtype(np.int16)

    def test_count_time_units(self, ts):
        for b, tu in zip([1, 1e3, 1e6], ["s", "ms", "us"]):
            count = ts.count(b, time_units=tu)
            assert len(count) == 99
            np.testing.assert_array_almost_equal(count.index, np.arange(0.5, 99, 1))

            count = ts.count(b, time_units=tu)
            assert len(count) == 99
            np.testing.assert_array_almost_equal(count.index, np.arange(0.5, 99, 1))

    def test_count_with_ep(self, ts):
        ep = nap.IntervalSet(start=0, end=100)
        count = ts.count(1, ep)
        assert len(count) == 100
        np.testing.assert_array_almost_equal(count.values, np.ones(100))

        count = ts.count(1, ep=ep)
        assert len(count) == 100
        np.testing.assert_array_almost_equal(count.values, np.ones(100))

    def test_count_with_ep_only(self, ts):
        ep = nap.IntervalSet(start=0, end=100)
        count = ts.count(ep=ep)
        assert len(count) == 1
        np.testing.assert_array_almost_equal(count.values, np.array([100]))

        count = ts.count(ep=ep)
        assert len(count) == 1
        np.testing.assert_array_almost_equal(count.values, np.array([100]))

        count = ts.count()
        assert len(count) == 1
        np.testing.assert_array_almost_equal(count.values, np.array([100]))

    def test_count_errors(self, ts):
        with pytest.raises(TypeError):
            ts.count(bin_size={})

        with pytest.raises(TypeError):
            ts.count(ep={})

        with pytest.raises(ValueError):
            ts.count(bin_size=1, time_units={})

    @pytest.mark.parametrize(
        "dtype, expectation",
        [
            (None, does_not_raise()),
            (float, does_not_raise()),
            (int, does_not_raise()),
            (np.int32, does_not_raise()),
            (np.int64, does_not_raise()),
            (np.float32, does_not_raise()),
            (np.float64, does_not_raise()),
            (1, pytest.raises(ValueError, match=f"1 is not a valid numpy dtype")),
        ],
    )
    def test_count_dtype(self, dtype, expectation, ts):
        with expectation:
            count = ts.count(bin_size=0.1, dtype=dtype)
            if dtype:
                assert np.issubdtype(count.dtype, dtype)

    @pytest.mark.parametrize(
        "ep, bin_size, align, padding_value, time_unit, expectation",
        [
            ([], 1, "start", np.nan, "s", "Argument ep should be of type IntervalSet"),
            (
                nap.IntervalSet(0, 1),
                "a",
                "start",
                np.nan,
                "s",
                "bin_size should be of type int or float",
            ),
            (
                nap.IntervalSet(0, 1),
                1,
                "a",
                np.nan,
                "s",
                "align should be 'start' or 'end'",
            ),
            (
                nap.IntervalSet(0, 1),
                1,
                "start",
                np.nan,
                1,
                "time_unit should be 's', 'ms' or 'us'",
            ),
        ],
    )
    def test_trial_count_runtime_errors(
        self, ts, ep, bin_size, align, padding_value, time_unit, expectation
    ):
        with pytest.raises(RuntimeError, match=re.escape(expectation)):
            ts.trial_count(ep, bin_size, align, padding_value, time_unit)

    def test_trial_count(self, ts):
        ep = nap.IntervalSet(
            start=np.arange(0, 100, 20), end=np.arange(0, 100, 20) + np.arange(0, 10, 2)
        )

        expected = np.ones((len(ep), 8)) * np.nan
        for i, k in zip(range(len(ep)), range(2, 10, 2)):
            expected[i, 0:k] = 1

        tensor = ts.trial_count(ep, bin_size=1)
        np.testing.assert_array_almost_equal(tensor, expected)

        tensor = ts.trial_count(ep, bin_size=1, align="start")
        np.testing.assert_array_almost_equal(tensor, expected)

        tensor = ts.trial_count(ep, bin_size=1, align="end")
        np.testing.assert_array_almost_equal(tensor, np.flip(expected, axis=1))

        tensor = ts.trial_count(ep, bin_size=1, time_unit="s")
        np.testing.assert_array_almost_equal(tensor, expected)

        tensor = ts.trial_count(ep, bin_size=1e3, time_unit="ms")
        np.testing.assert_array_almost_equal(tensor, expected)

        tensor = ts.trial_count(ep, bin_size=1e6, time_unit="us")
        np.testing.assert_array_almost_equal(tensor, expected)

        tensor = ts.trial_count(ep, bin_size=1, align="start", padding_value=-1)
        expected[np.isnan(expected)] = -1
        np.testing.assert_array_almost_equal(tensor, expected)


####################################################
# Test for tsdtensor
####################################################
@pytest.mark.parametrize(
    "tsdtensor",
    [
        nap.TsdTensor(t=np.arange(100), d=np.random.rand(100, 3, 2), time_units="s"),
    ],
)
class TestTsdTensor:
    @pytest.mark.parametrize("delta_ep", [(1, -1), (-1, -1), (1, 1)])
    def test_bin_average_time_support(self, delta_ep, tsdtensor):
        ep = nap.IntervalSet(
            tsdtensor.time_support.start[0] + delta_ep[0],
            tsdtensor.time_support.end[0] + delta_ep[1],
        )
        out = tsdtensor.bin_average(0.1, ep=ep)
        assert np.all(out.time_support == ep)

    @pytest.mark.parametrize("delta_ep", [(1, -1), (-1, -1), (1, 1)])
    def test_convolve_time_support(self, tsdtensor, delta_ep):
        ep = nap.IntervalSet(
            tsdtensor.time_support.start[0] + delta_ep[0],
            tsdtensor.time_support.end[0] + delta_ep[1],
        )
        out = tsdtensor.convolve(np.ones(10), ep=ep)
        assert np.all(out.time_support == ep)

    @pytest.mark.parametrize("delta_ep", [(1, -1), (-1, -1), (1, 1)])
    def test_interpolate_time_support(self, tsdtensor, delta_ep):
        ep = nap.IntervalSet(
            tsdtensor.time_support.start[0] + delta_ep[0],
            tsdtensor.time_support.end[0] + delta_ep[1],
        )
        ts = nap.Ts(np.linspace(0, 10, 20))
        out = tsdtensor.interpolate(ts, ep=ep)
        assert np.all(out.time_support == ep)

    def test_return_ndarray(self, tsdtensor):
        np.testing.assert_array_equal(tsdtensor[0], tsdtensor.values[0])

    def test_horizontal_slicing(self, tsdtensor):
        assert isinstance(tsdtensor[:, 0], nap.TsdFrame)
        np.testing.assert_array_almost_equal(
            tsdtensor[:, 0].values, tsdtensor.values[:, 0]
        )
        assert isinstance(tsdtensor[:, 0].time_support, nap.IntervalSet)
        np.testing.assert_array_almost_equal(
            tsdtensor.time_support, tsdtensor[:, 0].time_support
        )

        assert isinstance(tsdtensor[:, [0, 2]], nap.TsdTensor)
        np.testing.assert_array_almost_equal(
            tsdtensor.values[:, [0, 2]], tsdtensor[:, [0, 2]].values
        )
        assert isinstance(tsdtensor[:, [0, 2]].time_support, nap.IntervalSet)
        np.testing.assert_array_almost_equal(
            tsdtensor.time_support, tsdtensor[:, [0, 2]].time_support
        )

    def test_vertical_slicing(self, tsdtensor):
        assert isinstance(tsdtensor[0:10], nap.TsdTensor)
        np.testing.assert_array_almost_equal(
            tsdtensor.values[0:10], tsdtensor[0:10].values
        )
        assert isinstance(tsdtensor[0:10].time_support, nap.IntervalSet)
        np.testing.assert_array_almost_equal(
            tsdtensor[0:10].time_support, tsdtensor.time_support
        )

    def test_operators(self, tsdtensor):
        v = tsdtensor.values

        a = tsdtensor + 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v + 0.5))

        a = tsdtensor - 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v - 0.5))

        a = tsdtensor * 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v * 0.5))

        a = tsdtensor / 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v / 0.5))

        a = tsdtensor // 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v // 0.5))

        a = tsdtensor % 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v % 0.5))

        a = tsdtensor**0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        np.testing.assert_array_almost_equal(a.values, v**0.5)

        a = tsdtensor > 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v > 0.5))

        a = tsdtensor >= 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v >= 0.5))

        a = tsdtensor < 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v < 0.5))

        a = tsdtensor <= 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v <= 0.5))

        tsdtensor = nap.TsdTensor(
            t=np.arange(10), d=np.atleast_3d(np.tile(np.arange(10)[:, np.newaxis], 3))
        )
        v = tsdtensor.values
        a = tsdtensor == 5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v == 5))

        a = tsdtensor != 5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v != 5))

    def test_data(self, tsdtensor):
        np.testing.assert_array_almost_equal(tsdtensor.values, tsdtensor.data())

    def test_bin_average(self, tsdtensor):
        meantsd = tsdtensor.bin_average(10)
        assert len(meantsd) == 10
        np.testing.assert_array_almost_equal(meantsd.index, np.arange(5, 100, 10))
        bins = np.arange(
            tsdtensor.time_support.start[0], tsdtensor.time_support.end[0] + 1, 10
        )
        idx = np.digitize(tsdtensor.index, bins)
        tmp = []
        for i in np.unique(idx):
            tmp.append(np.mean(tsdtensor.values[idx == i], 0))
        tmp = np.array(tmp)
        np.testing.assert_array_almost_equal(meantsd.values, tmp)

    def test_save_npz(self, tsdtensor):
        with pytest.raises(TypeError) as e:
            tsdtensor.save(dict)

        with pytest.raises(RuntimeError) as e:
            tsdtensor.save("./")
        assert str(e.value) == "Invalid filename input. {} is directory.".format(
            Path("./").resolve()
        )

        fake_path = "./fake/path"
        with pytest.raises(RuntimeError) as e:
            tsdtensor.save(fake_path + "/file.npz")
        assert str(e.value) == "Path {} does not exist.".format(
            Path(fake_path).resolve()
        )

        tsdtensor.save("tsdtensor.npz")
        assert "tsdtensor.npz" in [f.name for f in Path(".").iterdir()]

        tsdtensor.save("tsdtensor2")
        assert "tsdtensor2.npz" in [f.name for f in Path(".").iterdir()]

        file = np.load("tsdtensor.npz")

        keys = list(file.keys())
        assert "t" in keys
        assert "d" in keys
        assert "start" in keys
        assert "end" in keys

        np.testing.assert_array_almost_equal(file["t"], tsdtensor.index)
        np.testing.assert_array_almost_equal(file["d"], tsdtensor.values)
        np.testing.assert_array_almost_equal(
            file["start"], tsdtensor.time_support.start
        )
        np.testing.assert_array_almost_equal(file["end"], tsdtensor.time_support.end)

        Path("tsdtensor.npz").unlink()
        Path("tsdtensor2.npz").unlink()

    def test_interpolate(self, tsdtensor):
        y = np.arange(0, 1001)
        data_stack = np.stack(
            [
                np.stack(
                    [
                        np.arange(0, 1001),
                    ]
                    * 4
                )
            ]
            * 3
        ).T

        tsdtensor = nap.TsdTensor(t=np.arange(0, 101), d=data_stack[0::10, ...])

        # Ts
        ts = nap.Ts(t=y / 10)
        tsdtensor2 = tsdtensor.interpolate(ts)
        np.testing.assert_array_almost_equal(tsdtensor2.values, data_stack)

        # Tsd
        ts = nap.Tsd(t=y / 10, d=np.zeros_like(y))
        tsdtensor2 = tsdtensor.interpolate(ts)
        np.testing.assert_array_almost_equal(tsdtensor2.values, data_stack)

        # TsdFrame
        ts = nap.TsdFrame(t=y / 10, d=np.zeros((len(y), 2)))
        tsdtensor2 = tsdtensor.interpolate(ts)
        np.testing.assert_array_almost_equal(tsdtensor2.values, data_stack)

        with pytest.raises(IOError) as e:
            tsdtensor.interpolate([0, 1, 2])
        assert (
            str(e.value)
            == "First argument should be an instance of Ts, Tsd, TsdFrame or TsdTensor"
        )

        # Right left
        ep = nap.IntervalSet(start=0, end=5)
        tsdtensor = nap.Tsd(t=np.arange(1, 4), d=np.arange(3), time_support=ep)
        ts = nap.Ts(t=np.arange(0, 5))
        tsdtensor2 = tsdtensor.interpolate(ts, left=1234)
        assert float(tsdtensor2.values[0]) == 1234.0
        tsdtensor2 = tsdtensor.interpolate(ts, right=4321)
        assert float(tsdtensor2.values[-1]) == 4321.0

    def test_interpolate_with_ep(self, tsdtensor):
        y = np.arange(0, 1001)
        data_stack = np.stack(
            [
                np.stack(
                    [
                        np.arange(0, 1001),
                    ]
                    * 4
                ),
            ]
            * 3
        ).T

        tsdtensor = nap.TsdTensor(t=np.arange(0, 101), d=data_stack[::10, ...])

        ts = nap.Ts(t=y / 10)

        ep = nap.IntervalSet(start=np.arange(0, 100, 20), end=np.arange(10, 110, 20))
        tsdframe2 = tsdtensor.interpolate(ts, ep)
        tmp = ts.restrict(ep).index * 10

        np.testing.assert_array_almost_equal(tmp, tsdframe2.values[:, 0, 0])

        # Empty ep
        ep = nap.IntervalSet(start=200, end=300)
        tsdframe2 = tsdtensor.interpolate(ts, ep)
        assert len(tsdframe2) == 0

    def test_derivative(self, tsdtensor):
        t = np.arange(10)
        d = np.array([[[i, i], [-i, -i]] for i in range(10)])
        tsdtensor = nap.TsdTensor(t=t, d=d)
        derivative = tsdtensor.derivative()
        expected = np.array([[[1, 1], [-1, -1]] for _ in range(10)])
        assert np.all(derivative.values == expected)

    def test_derivative_with_ep(self, tsdtensor):
        t = np.arange(10)
        d = np.array([[[i, i], [-i, -i]] for i in range(10)])
        ep = nap.IntervalSet(start=[1, 6], end=[4, 10])
        expected = np.array([[[1, 1], [-1, -1]] for _ in range(8)])

        # With time support
        tsdtensor = nap.TsdTensor(t=t, d=d, time_support=ep)
        derivative = tsdtensor.derivative()
        assert np.all(derivative.values == expected)

        # With ep parameter
        tsdtensor = nap.TsdTensor(t=t, d=d)
        derivative = tsdtensor.derivative(ep=ep)
        assert np.all(derivative.values == expected)

    def test_indexing_with_boolean_tsd(self, tsdtensor):
        # Create a boolean Tsd for indexing
        index_tsd = nap.Tsd(
            t=tsdtensor.t, d=np.random.choice([True, False], size=len(tsdtensor))
        )

        # Test indexing
        result = tsdtensor[index_tsd]

        assert isinstance(result, nap.TsdTensor)
        assert len(result) == index_tsd.d.sum()
        np.testing.assert_array_equal(result.t, tsdtensor.t[index_tsd.d])
        np.testing.assert_array_equal(result.values, tsdtensor.values[index_tsd.d])

    def test_indexing_with_boolean_tsd_and_additional_slicing(self, tsdtensor):
        # Create a boolean Tsd for indexing
        index_tsd = nap.Tsd(
            t=tsdtensor.t, d=np.random.choice([True, False], size=len(tsdtensor))
        )

        # Test indexing with additional dimension slicing
        result = tsdtensor[index_tsd, 1]

        assert isinstance(result, nap.TsdFrame)
        assert len(result) == index_tsd.d.sum()
        np.testing.assert_array_equal(result.t, tsdtensor.t[index_tsd.d])
        np.testing.assert_array_equal(result.values, tsdtensor.values[index_tsd.d, 1])

    def test_indexing_with_non_boolean_tsd_raises_error(self, tsdtensor):
        # Create a non-boolean Tsd
        index_tsd = nap.Tsd(t=np.array([10, 20, 30, 40]), d=np.array([1, 2, 3, 0]))

        # Test indexing with non-boolean Tsd should raise an error
        with pytest.raises(
            ValueError, match="When indexing with a Tsd, it must contain boolean values"
        ):
            tsdtensor[index_tsd]

    def test_indexing_with_mismatched_boolean_tsd_raises_error(self, tsdtensor):
        # Create a boolean Tsd with mismatched length
        index_tsd = nap.Tsd(
            t=np.arange(len(tsdtensor) + 5),
            d=np.random.choice([True, False], size=len(tsdtensor) + 5),
        )

        # Test indexing with mismatched boolean Tsd should raise an error
        with pytest.raises(IndexError, match="boolean index did not match"):
            tsdtensor[index_tsd]

    @skip_if_backend("jax")
    def test_setitem_with_boolean_tsd(self, tsdtensor):
        # Create a boolean Tsd for indexing
        index_tsd = nap.Tsd(
            t=tsdtensor.t, d=np.random.choice([True, False], size=len(tsdtensor))
        )

        # Create new values to set
        new_values = np.random.rand(*tsdtensor.shape[1:])

        # Set values using boolean indexing
        tsdtensor[index_tsd] = new_values

        # Check if values were set correctly
        np.testing.assert_array_equal(
            tsdtensor.values[index_tsd.d], np.stack([new_values] * sum(index_tsd.d))
        )

        # Check if other values remained unchanged
        np.testing.assert_array_equal(
            tsdtensor.values[~index_tsd.d], tsdtensor.values[~index_tsd.d]
        )


@pytest.mark.parametrize(
    "obj",
    [
        nap.Tsd(t=np.arange(10), d=np.random.rand(10), time_units="s"),
        nap.TsdFrame(
            t=np.arange(10),
            d=np.random.rand(10, 3),
            time_units="s",
            columns=["a", "b", "c"],
        ),
        nap.TsdTensor(t=np.arange(10), d=np.random.rand(10, 3, 2), time_units="s"),
    ],
)
def test_pickling(obj):
    """Test that pikling works as expected."""
    # pickle and unpickle ts_group
    pickled_obj = pickle.dumps(obj)
    unpickled_obj = pickle.loads(pickled_obj)

    # Ensure time is the same
    assert np.all(obj.t == unpickled_obj.t)

    # Ensure data is the same
    assert np.all(obj.d == unpickled_obj.d)

    # Ensure time support is the same
    assert np.all(obj.time_support == unpickled_obj.time_support)


#

####################################################
# Test for slicing
####################################################


@pytest.mark.parametrize(
    "start, end, mode, n_points, expectation",
    [
        (1, 3, "closest_t", None, does_not_raise()),
        (
            None,
            3,
            "closest_t",
            None,
            pytest.raises(ValueError, match="'start' must be an int or a float"),
        ),
        (
            2,
            "a",
            "closest_t",
            None,
            pytest.raises(
                ValueError,
                match="'end' must be an int or a float. Type <class 'str'> provided instead!",
            ),
        ),
        (
            1,
            3,
            "closest_t",
            "a",
            pytest.raises(
                TypeError,
                match="'n_points' must be of type int or None. Type <class 'str'> provided instead!",
            ),
        ),
        (
            1,
            None,
            "closest_t",
            1,
            pytest.raises(
                ValueError, match="'n_points' can be used only when 'end' is specified!"
            ),
        ),
        (
            1,
            3,
            "pineapple",
            None,
            pytest.raises(
                ValueError,
                match="'mode' only accepts 'before_t', 'after_t', 'closest_t' or 'restrict'.",
            ),
        ),
        (
            3,
            1,
            "closest_t",
            None,
            pytest.raises(ValueError, match="'start' should not precede 'end'"),
        ),
        (
            1,
            3,
            "restrict",
            1,
            pytest.raises(
                ValueError,
                match="Fixing the number of time points is incompatible with 'restrict' mode.",
            ),
        ),
        (1.0, 3.0, "closest_t", None, does_not_raise()),
        (1.0, None, "closest_t", None, does_not_raise()),
    ],
)
def test_get_slice_raise_errors(start, end, mode, n_points, expectation):
    ts = nap.Ts(t=np.array([1, 2, 3, 4]))
    with expectation:
        ts._get_slice(start, end, mode, n_points)


@pytest.mark.parametrize(
    "start, end, mode, expected_slice, expected_array",
    [
        (1, 3, "after_t", slice(0, 2), np.array([1, 2])),
        (1, 3, "before_t", slice(0, 2), np.array([1, 2])),
        (1, 3, "closest_t", slice(0, 2), np.array([1, 2])),
        (1, 2.7, "after_t", slice(0, 2), np.array([1, 2])),
        (1, 2.7, "before_t", slice(0, 1), np.array([1])),
        (1, 2.7, "closest_t", slice(0, 2), np.array([1, 2])),
        (1, 2.4, "after_t", slice(0, 2), np.array([1, 2])),
        (1, 2.4, "before_t", slice(0, 1), np.array([1])),
        (1, 2.4, "closest_t", slice(0, 1), np.array([1])),
        (1.1, 3, "after_t", slice(1, 2), np.array([2])),
        (1.1, 3, "before_t", slice(0, 2), np.array([1, 2])),
        (1.1, 3, "closest_t", slice(0, 2), np.array([1, 2])),
        (1.6, 3, "after_t", slice(1, 2), np.array([2])),
        (1.6, 3, "before_t", slice(0, 2), np.array([1, 2])),
        (1.6, 3, "closest_t", slice(1, 2), np.array([2])),
        (1.6, 1.8, "before_t", slice(0, 0), np.array([])),
        (1.6, 1.8, "after_t", slice(1, 1), np.array([])),
        (1.6, 1.8, "closest_t", slice(1, 1), np.array([])),
        (1.4, 1.6, "closest_t", slice(0, 1), np.array([1])),
        (3, 3, "after_t", slice(2, 2), np.array([])),
        (3, 3, "before_t", slice(2, 2), np.array([])),
        (3, 3, "closest_t", slice(2, 2), np.array([])),
        (0, 3, "after_t", slice(0, 2), np.array([1, 2])),
        (0, 3, "before_t", slice(0, 2), np.array([1, 2])),
        (0, 3, "closest_t", slice(0, 2), np.array([1, 2])),
        (0, 4, "after_t", slice(0, 3), np.array([1, 2, 3])),
        (0, 4, "before_t", slice(0, 3), np.array([1, 2, 3])),
        (0, 4, "closest_t", slice(0, 3), np.array([1, 2, 3])),
        (4, 4, "after_t", slice(3, 3), np.array([])),
        (4, 4, "before_t", slice(3, 3), np.array([])),
        (4, 4, "closest_t", slice(3, 3), np.array([])),
        (4, 5, "after_t", slice(3, 4), np.array([4])),
        (4, 5, "before_t", slice(3, 3), np.array([])),
        (4, 5, "closest_t", slice(3, 3), np.array([])),
        (0, 1, "after_t", slice(0, 0), np.array([])),
        (0, 1, "before_t", slice(0, 1), np.array([1])),
        (0, 1, "closest_t", slice(0, 0), np.array([])),
        (0, None, "after_t", slice(0, 1), np.array([1])),
        (0, None, "before_t", slice(0, 0), np.array([])),
        (0, None, "closest_t", slice(0, 1), np.array([1])),
        (1, None, "after_t", slice(0, 1), np.array([1])),
        (1, None, "before_t", slice(0, 1), np.array([1])),
        (1, None, "closest_t", slice(0, 1), np.array([1])),
        (5, None, "after_t", slice(3, 3), np.array([])),
        (5, None, "before_t", slice(3, 4), np.array([4])),
        (5, None, "closest_t", slice(3, 4), np.array([4])),
        (1, 3, "restrict", slice(0, 3), np.array([1, 2, 3])),
        (1, 2.7, "restrict", slice(0, 2), np.array([1, 2])),
        (1, 2.4, "restrict", slice(0, 2), np.array([1, 2])),
        (1.1, 3, "restrict", slice(1, 3), np.array([2, 3])),
        (1.6, 3, "restrict", slice(1, 3), np.array([2, 3])),
        (1.6, 1.8, "restrict", slice(1, 1), np.array([])),
        (1.4, 1.6, "restrict", slice(1, 1), np.array([])),
        (3, 3, "restrict", slice(2, 3), np.array([3])),
        (0, 3, "restrict", slice(0, 3), np.array([1, 2, 3])),
        (0, 4, "restrict", slice(0, 4), np.array([1, 2, 3, 4])),
        (4, 4, "restrict", slice(3, 4), np.array([4])),
        (4, 5, "restrict", slice(3, 4), np.array([4])),
        (0, 1, "restrict", slice(0, 1), np.array([1])),
    ],
)
@pytest.mark.parametrize(
    "ts",
    [
        nap.Ts(t=np.array([1, 2, 3, 4])),
        nap.Tsd(t=np.array([1, 2, 3, 4]), d=np.array([1, 2, 3, 4])),
        nap.TsdFrame(t=np.array([1, 2, 3, 4]), d=np.array([1, 2, 3, 4])[:, None]),
        nap.TsdTensor(
            t=np.array([1, 2, 3, 4]), d=np.array([1, 2, 3, 4])[:, None, None]
        ),
    ],
)
def test_get_slice_value(start, end, mode, expected_slice, expected_array, ts):
    out_slice = ts._get_slice(start, end=end, mode=mode)
    out_array = ts.t[out_slice]
    assert out_slice == expected_slice
    assert np.all(out_array == expected_array)
    if mode == "restrict":
        iset = nap.IntervalSet(start, end)
        out_restrict = ts.restrict(iset)
        assert np.all(out_restrict.t == out_array)


def test_get_slice_vs_get_random_val_start_end_value():
    np.random.seed(123)
    ts = nap.Ts(np.linspace(0.2, 0.8, 100))
    se_vec = np.random.uniform(0, 1, size=(10000, 2))
    starts = np.min(se_vec, axis=1)
    ends = np.max(se_vec, axis=1)
    for start, end in zip(starts, ends):
        out_slice = ts.get_slice(start=start, end=end)
        out_ts = ts[out_slice]
        out_get = ts.get(start, end)
        assert np.all(out_get.t == out_ts.t)


def test_get_slice_vs_get_random_val_start_value():
    np.random.seed(123)
    ts = nap.Ts(np.linspace(0.2, 0.8, 100))
    starts = np.random.uniform(0, 1, size=(10000,))

    for start in starts:
        out_slice = ts.get_slice(start=start, end=None)
        out_ts = ts[out_slice]
        out_get = ts.get(start)
        assert np.all(out_get.t == out_ts.t)


@pytest.mark.parametrize(
    "end, n_points, expectation",
    [
        (1, 3, does_not_raise()),
        (None, 3, pytest.raises(ValueError, match="'n_points' can be used only when")),
    ],
)
@pytest.mark.parametrize("time_unit", ["s", "ms", "us"])
@pytest.mark.parametrize("mode", ["closest_t", "before_t", "after_t"])
def test_get_slice_n_points(end, n_points, expectation, time_unit, mode):
    ts = nap.Ts(t=np.array([1, 2, 3, 4]))
    with expectation:
        ts._get_slice(1, end, n_points=n_points, mode=mode)


@pytest.mark.parametrize(
    "start, end, n_points, mode, expected_slice, expected_array",
    [
        # smaller than n_points
        (1, 2, 2, "after_t", slice(0, 1), np.array([1])),
        (1, 2, 2, "before_t", slice(0, 1), np.array([1])),
        (1, 2, 2, "closest_t", slice(0, 1), np.array([1])),
        # larger than n_points
        (1, 5, 2, "after_t", slice(0, 4, 2), np.array([1, 3])),
        (1, 5, 2, "before_t", slice(0, 4, 2), np.array([1, 3])),
        (1, 5, 2, "closest_t", slice(0, 4, 2), np.array([1, 3])),
        # larger than n_points with rounding down
        (1, 5.2, 2, "after_t", slice(0, 4, 2), np.array([1, 3])),
        (1, 5.2, 2, "before_t", slice(0, 4, 2), np.array([1, 3])),
        (1, 5.2, 2, "closest_t", slice(0, 4, 2), np.array([1, 3])),
        # larger than n_points with rounding down
        (1, 6.2, 2, "after_t", slice(0, 6, 3), np.array([1, 4])),
        (1, 6.2, 2, "before_t", slice(0, 4, 2), np.array([1, 3])),
        (1, 6.2, 2, "closest_t", slice(0, 4, 2), np.array([1, 3])),
        # larger than n_points with rounding up
        (1, 5.6, 2, "after_t", slice(0, 4, 2), np.array([1, 3])),
        (1, 5.6, 2, "before_t", slice(0, 4, 2), np.array([1, 3])),
        (1, 5.6, 2, "closest_t", slice(0, 4, 2), np.array([1, 3])),
        # larger than n_points with rounding up
        (1, 6.6, 2, "after_t", slice(0, 6, 3), np.array([1, 4])),
        (1, 6.6, 2, "before_t", slice(0, 4, 2), np.array([1, 3])),
        (1, 6.6, 2, "closest_t", slice(0, 6, 3), np.array([1, 4])),
    ],
)
@pytest.mark.parametrize(
    "ts",
    [
        nap.Ts(t=np.arange(1, 10)),
        nap.Tsd(t=np.arange(1, 10), d=np.arange(1, 10)),
        nap.TsdFrame(t=np.arange(1, 10), d=np.arange(1, 10)[:, None]),
        nap.TsdTensor(t=np.arange(1, 10), d=np.arange(1, 10)[:, None, None]),
    ],
)
def test_get_slice_value_step(
    start, end, n_points, mode, expected_slice, expected_array, ts
):
    out_slice = ts._get_slice(start, end=end, mode=mode, n_points=n_points)
    out_array = ts.t[out_slice]
    assert out_slice == expected_slice
    assert np.all(out_array == expected_array)


@pytest.mark.parametrize(
    "start, end, expected_slice, expected_array",
    [
        (1, 3, slice(0, 3), np.array([1, 2, 3])),
        (1, 2.7, slice(0, 2), np.array([1, 2])),
        (1, 2.4, slice(0, 2), np.array([1, 2])),
        (1.1, 3, slice(1, 3), np.array([2, 3])),
        (1.6, 3, slice(1, 3), np.array([2, 3])),
        (1.6, 1.8, slice(1, 1), np.array([])),
        (1.4, 1.6, slice(1, 1), np.array([])),
        (3, 3, slice(2, 3), np.array([3])),
        (0, 3, slice(0, 3), np.array([1, 2, 3])),
        (0, 4, slice(0, 4), np.array([1, 2, 3, 4])),
        (4, 4, slice(3, 4), np.array([4])),
        (4, 5, slice(3, 4), np.array([4])),
        (0, 1, slice(0, 1), np.array([1])),
        (0, None, slice(0, 1), np.array([1])),
        (1, None, slice(0, 1), np.array([1])),
        (4, None, slice(3, 4), np.array([4])),
        (5, None, slice(3, 4), np.array([4])),
        (-1, 0, slice(0, 0), np.array([])),
        (5, 6, slice(4, 4), np.array([])),
    ],
)
@pytest.mark.parametrize(
    "ts",
    [
        nap.Ts(t=np.array([1, 2, 3, 4])),
        nap.Tsd(t=np.array([1, 2, 3, 4]), d=np.array([1, 2, 3, 4])),
        nap.TsdFrame(t=np.array([1, 2, 3, 4]), d=np.array([1, 2, 3, 4])[:, None]),
        nap.TsdTensor(
            t=np.array([1, 2, 3, 4]), d=np.array([1, 2, 3, 4])[:, None, None]
        ),
    ],
)
def test_get_slice_public(start, end, expected_slice, expected_array, ts):
    out_slice = ts.get_slice(start, end=end)
    out_array = ts.t[out_slice]
    assert out_slice == expected_slice
    assert np.all(out_array == expected_array)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"columns": [1, 2]},
        {"metadata": {"banana": [3, 4]}},
        {"columns": ["a", "b"], "metadata": {"banana": [3, 4]}},
    ],
)
@pytest.mark.parametrize(
    "tsd",
    [
        nap.Ts(t=np.arange(10), time_support=nap.IntervalSet(0, 15)),
        nap.Ts(t=np.arange(10), time_support=nap.IntervalSet(0, 15)),
        nap.Tsd(t=np.arange(10), d=np.arange(10), time_support=nap.IntervalSet(0, 15)),
        nap.Tsd(t=np.arange(10), d=np.arange(10), time_support=nap.IntervalSet(0, 15)),
        nap.TsdFrame(
            t=np.arange(10),
            d=np.zeros((10, 2)),
            time_support=nap.IntervalSet(0, 15),
            columns=["a", "b"],
        ),
        nap.TsdFrame(
            t=np.arange(10),
            d=np.zeros((10, 2)),
            time_support=nap.IntervalSet(0, 15),
            metadata={"pineapple": [1, 2]},
        ),
        nap.TsdFrame(
            t=np.arange(10),
            d=np.zeros((10, 2)),
            time_support=nap.IntervalSet(0, 15),
            load_array=True,
        ),
        nap.TsdFrame(
            t=np.arange(10),
            d=np.zeros((10, 2)),
            time_support=nap.IntervalSet(0, 15),
            load_array=True,
            columns=["a", "b"],
            metadata={"pineapple": [1, 2]},
        ),
        nap.TsdTensor(
            t=np.arange(10), d=np.zeros((10, 2, 3)), time_support=nap.IntervalSet(0, 15)
        ),
    ],
)
def test_define_instance(tsd, kwargs):
    t = tsd.t
    d = getattr(tsd, "d", None)
    iset = tsd.time_support
    cols = kwargs.get("columns", None)

    # metadata index must be cols if provided.
    # clear metadata if cols are provided to avoid errors
    if (cols is not None) and ("metadata" not in kwargs):
        kwargs["metadata"] = {}

    out = tsd._define_instance(t, iset, values=d, **kwargs)

    # check data
    np.testing.assert_array_equal(out.t, t)
    np.testing.assert_array_equal(out.time_support, iset)
    if hasattr(tsd, "d"):
        np.testing.assert_array_equal(out.d, d)

    # if TsdFrame check kwargs
    if isinstance(tsd, nap.TsdFrame):
        val = kwargs.get("columns", getattr(tsd, "columns"))
        assert np.all(val == getattr(out, "columns"))
        # get expected metadata
        meta = kwargs.get("metadata", getattr(tsd, "metadata"))
        for (
            key,
            val,
        ) in meta.items():
            assert np.all(out.metadata[key] == val)


time_array = np.arange(-4, 10, 1)
time_target_array = np.arange(-2, 10) + 0.01
starts = np.array([0])  # np.array([0, 8])
ends = np.array([10])  # np.array([6, 10])
a = nap.Tsd(
    t=time_target_array,
    d=np.arange(time_target_array.shape[0]),
    time_support=nap.IntervalSet(starts, ends),
)
ts = nap.Ts(time_array)


@pytest.mark.parametrize(
    "tsd, ts, mode, expected_data",
    [
        (
            nap.Tsd(
                t=np.arange(10) + 0.01,
                d=np.arange(10),
                time_support=nap.IntervalSet(0, 10),
            ),
            nap.Ts(np.arange(-4, 10, 1)),
            "before",
            np.array([np.nan] + [*range(0, 9)]),
        ),
        (
            nap.Tsd(
                t=np.arange(9) + 0.01,
                d=np.arange(9),
                time_support=nap.IntervalSet(0, 10),
            ),
            nap.Ts(np.arange(-4, 10, 1)),
            "after",
            np.array([*range(0, 9)] + [np.nan]),
        ),
    ],
)
def test_value_from_out_of_range(tsd, ts, mode, expected_data):
    out = ts.value_from(tsd, mode=mode)
    # type should be float even if tsd is int
    assert np.issubdtype(tsd.d.dtype, np.integer)
    assert np.issubdtype(out.d.dtype, np.floating)
    np.testing.assert_array_equal(out.d, expected_data)
