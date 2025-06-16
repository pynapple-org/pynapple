"""Tests of ts group for `pynapple` package."""

import pickle
import re
import warnings
from collections import UserDict
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pynapple as nap


@pytest.fixture
def group():
    """Fixture to be used in all tests."""
    return {
        0: nap.Ts(t=np.arange(0, 200)),
        1: nap.Ts(t=np.arange(0, 200, 0.5), time_units="s"),
        2: nap.Ts(t=np.arange(0, 300, 0.2), time_units="s"),
    }


@pytest.fixture
def ts_group():
    # Placeholder setup for Ts and Tsd objects. Adjust as necessary.
    ts1 = nap.Ts(t=np.arange(10))
    ts2 = nap.Ts(t=np.arange(5))
    data = {1: ts1, 2: ts2}
    group = nap.TsGroup(data, meta=[10, 11])
    return group


@pytest.fixture
def ts_group_one_group():
    # Placeholder setup for Ts and Tsd objects. Adjust as necessary.
    ts1 = nap.Ts(t=np.arange(10))
    data = {1: ts1}
    group = nap.TsGroup(data, meta=[10])
    return group


class TestTsGroup1:
    def test_create_ts_group(self, group):
        tsgroup = nap.TsGroup(group)
        assert isinstance(tsgroup, UserDict)
        assert len(tsgroup) == 3

    def test_create_ts_group_from_iter(self, group):
        tsgroup = nap.TsGroup(group.values())
        assert isinstance(tsgroup, UserDict)
        assert len(tsgroup) == 3

    def test_create_ts_group_from_invalid(self):
        with pytest.raises(AttributeError):
            tsgroup = nap.TsGroup(np.arange(0, 200))

    @pytest.mark.parametrize(
        "test_dict, expectation",
        [
            (
                {"1": nap.Ts(np.arange(10)), "2": nap.Ts(np.arange(10))},
                does_not_raise(),
            ),
            ({"1": nap.Ts(np.arange(10)), 2: nap.Ts(np.arange(10))}, does_not_raise()),
            (
                {"1": nap.Ts(np.arange(10)), 1: nap.Ts(np.arange(10))},
                pytest.raises(
                    ValueError, match="Two dictionary keys contain the same integer"
                ),
            ),
            (
                {"1.": nap.Ts(np.arange(10)), 2: nap.Ts(np.arange(10))},
                pytest.raises(ValueError, match="All keys must be convertible"),
            ),
            ({-1: nap.Ts(np.arange(10)), 1: nap.Ts(np.arange(10))}, does_not_raise()),
            (
                {1.5: nap.Ts(np.arange(10)), 1: nap.Ts(np.arange(10))},
                pytest.raises(ValueError, match="All keys must have integer value"),
            ),
        ],
    )
    def test_initialize_from_dict(self, test_dict, expectation):
        with expectation:
            nap.TsGroup(test_dict)

    def test_create_empty_tsgroup(self):
        tsgroup = nap.TsGroup(data={}, time_support=nap.IntervalSet(0, 1))

        assert isinstance(tsgroup, nap.TsGroup)
        assert len(tsgroup) == 0
        # Need to make sure the metadata has the rate attribute
        assert "rate" in tsgroup.metadata.columns

    @pytest.mark.parametrize(
        "tsgroup",
        [
            nap.TsGroup({"1": nap.Ts(np.arange(10)), "2": nap.Ts(np.arange(10))}),
            nap.TsGroup({"1": nap.Ts(np.arange(10)), 2: nap.Ts(np.arange(10))}),
            nap.TsGroup({-1: nap.Ts(np.arange(10)), 1: nap.Ts(np.arange(10))}),
        ],
    )
    def test_metadata_len_match(self, tsgroup):
        assert tsgroup._metadata.shape[0] == len(tsgroup)

    def test_create_ts_group_from_array(self):
        with warnings.catch_warnings(record=True) as w:
            nap.TsGroup(
                {
                    0: np.arange(0, 200),
                    1: np.arange(0, 200, 0.5),
                    2: np.arange(0, 300, 0.2),
                }
            )
        assert (
            str(w[0].message)
            == "Elements should not be passed as <class 'numpy.ndarray'>. Default time units is seconds when creating the Ts object."
        )

    def test_create_ts_group_with_time_support(self, group):
        ep = nap.IntervalSet(start=0, end=100)
        tsgroup = nap.TsGroup(group, time_support=ep)
        np.testing.assert_array_almost_equal(tsgroup.time_support, ep)
        first = [tsgroup[i].index[0] for i in tsgroup]
        last = [tsgroup[i].index[-1] for i in tsgroup]
        assert np.all(first >= ep[0, 0])
        assert np.all(last <= ep[0, 1])

    def test_create_ts_group_with_empty_time_support(self):
        with pytest.raises(RuntimeError) as e_info:
            tmp = nap.TsGroup(
                {
                    0: nap.Ts(t=np.array([])),
                    1: nap.Ts(t=np.array([])),
                    2: nap.Ts(t=np.array([])),
                }
            )
        assert (
            str(e_info.value)
            == "Union of time supports is empty. Consider passing a time support as argument."
        )

    def test_create_ts_group_with_bypass_check(self):
        tmp = {
            0: nap.Ts(t=np.arange(0, 100)),
            1: nap.Ts(t=np.arange(0, 200, 0.5), time_units="s"),
            2: nap.Ts(t=np.arange(0, 300, 0.2), time_units="s"),
        }
        tsgroup = nap.TsGroup(
            tmp, time_support=nap.IntervalSet(0, 100), bypass_check=True
        )
        for i in tmp.keys():
            np.testing.assert_array_almost_equal(tmp[i].index, tsgroup[i].index)

        tmp = {
            0: nap.Ts(t=np.arange(0, 100)),
            1: nap.Ts(t=np.arange(0, 200, 0.5), time_units="s"),
            2: nap.Ts(t=np.arange(0, 300, 0.2), time_units="s"),
        }
        tsgroup = nap.TsGroup(tmp, bypass_check=True)
        for i in tmp.keys():
            np.testing.assert_array_almost_equal(tmp[i].index, tsgroup[i].index)

    def test_create_ts_group_with_metainfo(self, group):
        sr_info = pd.Series(index=[0, 1, 2], data=[0, 0, 0], name="sr")
        ar_info = np.ones(3) * 1
        tsgroup = nap.TsGroup(group, sr=sr_info, ar=ar_info)
        assert tsgroup._metadata.shape == (3, 3)
        np.testing.assert_array_almost_equal(tsgroup._metadata["sr"], sr_info.values)
        np.testing.assert_array_almost_equal(
            tsgroup._metadata.index, sr_info.index.values
        )
        np.testing.assert_array_almost_equal(tsgroup._metadata["ar"], ar_info)

    def test_keys(self, group):
        tsgroup = nap.TsGroup(group)
        assert tsgroup.keys() == [0, 1, 2]

    def test_rates_property(self, group):
        tsgroup = nap.TsGroup(group)
        np.testing.assert_array_almost_equal(tsgroup.rates, tsgroup._metadata["rate"])

    def test_items(self, group):
        tsgroup = nap.TsGroup(group)
        items = tsgroup.items()
        assert isinstance(items, list)
        for i, it in items:
            pd.testing.assert_series_equal(tsgroup[i].as_series(), it.as_series())

    def test_items(self, group):
        tsgroup = nap.TsGroup(group)
        values = tsgroup.values()
        assert isinstance(values, list)
        for i, it in enumerate(values):
            pd.testing.assert_series_equal(tsgroup[i].as_series(), it.as_series())

    def test_slicing(self, group):
        tsgroup = nap.TsGroup(group)
        assert isinstance(tsgroup[0], nap.Ts)
        pd.testing.assert_series_equal(group[0].as_series(), tsgroup[0].as_series())
        assert isinstance(tsgroup[[0, 2]], nap.TsGroup)
        assert len(tsgroup[[0, 2]]) == 2
        assert tsgroup[[0, 2]].keys() == [0, 2]

    def test_slicing_error(self, group):
        tsgroup = nap.TsGroup(group)
        with pytest.raises(Exception):
            tmp = tsgroup[4]

    def test_get_rate(self, group):
        tsgroup = nap.TsGroup(group)
        rate = tsgroup._metadata["rate"]
        np.testing.assert_array_almost_equal(rate, tsgroup.get_info("rate"))

    def test_restrict(self, group):
        tsgroup = nap.TsGroup(group)
        ep = nap.IntervalSet(start=0, end=100)
        tsgroup2 = tsgroup.restrict(ep)
        first = [tsgroup2[i].index[0] for i in tsgroup2]
        last = [tsgroup2[i].index[-1] for i in tsgroup2]
        assert np.all(first >= ep[0, 0])
        assert np.all(last <= ep[0, 1])

    def test_value_from(self, group):
        tsgroup = nap.TsGroup(group)
        tsd = nap.Tsd(t=np.arange(0, 300, 0.1), d=np.arange(3000))
        tsgroup2 = tsgroup.value_from(tsd)
        np.testing.assert_array_almost_equal(tsgroup2[0].values, np.arange(0, 2000, 10))
        np.testing.assert_array_almost_equal(tsgroup2[1].values, np.arange(0, 2000, 5))
        np.testing.assert_array_almost_equal(tsgroup2[2].values, np.arange(0, 3000, 2))

    def test_value_from_raise_type_errors(self, group):
        tsgroup = nap.TsGroup(group)
        tsd = nap.Tsd(t=np.arange(0, 300, 0.1), d=np.arange(3000))

        with pytest.raises(
            TypeError,
            match=r"First argument should be an instance of Tsd, TsdFrame or TsdTensor",
        ):
            tsgroup.value_from(tsd={})

        with pytest.raises(
            TypeError, match=r"Argument ep should be of type IntervalSet or None"
        ):
            tsgroup.value_from(tsd=tsd, ep={})

        with pytest.raises(
            ValueError,
            match=r"Argument mode should be 'closest', 'before', or 'after'. 1 provided instead.",
        ):
            tsgroup.value_from(tsd=tsd, mode=1)

    @pytest.mark.parametrize("mode", ["before", "closest", "after"])
    def test_value_from_tsd_mode(self, group, mode):
        # case 1: tim-stamps form tsd are subset of time-stamps of tsd2
        # In this case all modes should do the same thing
        tsgroup = nap.TsGroup(group)
        tsd = nap.Tsd(t=np.arange(0, 300, 0.1), d=np.arange(3000))
        tsgroup2 = tsgroup.value_from(tsd, mode=mode)
        assert len(tsgroup) == len(tsgroup2)
        np.testing.assert_array_almost_equal(tsgroup2[0].values, np.arange(0, 2000, 10))
        np.testing.assert_array_almost_equal(tsgroup2[1].values, np.arange(0, 2000, 5))
        np.testing.assert_array_almost_equal(tsgroup2[2].values, np.arange(0, 3000, 2))

        # case2: timestamps of tsd (integers) are not subset of that of tsd2.
        tsd2 = nap.Tsd(t=np.arange(0.0, 300.3, 0.3), d=np.random.rand(1002))

        tsgroup2 = tsgroup.value_from(tsd2, mode=mode)
        # loop over epochs
        for iset in tsd.time_support:
            single_ep_tsgroup = tsgroup.restrict(iset)
            single_ep_tsgroup2 = tsgroup2.restrict(iset)
            single_ep_tsd2 = tsd2.restrict(iset)

            for idx, ts in single_ep_tsgroup.items():
                ts2 = single_ep_tsgroup2[idx]
                # extract the indices with searchsorted.
                if mode == "before":
                    expected_idx = (
                        np.searchsorted(single_ep_tsd2.t, ts.t, side="right") - 1
                    )
                    # check that times are actually before
                    assert np.all(single_ep_tsd2.t[expected_idx] <= ts2.t)
                    # check that subsequent are after
                    assert np.all(single_ep_tsd2.t[expected_idx[:-1] + 1] > ts2.t[:-1])
                    valid = np.ones(len(ts), dtype=bool)
                elif mode == "after":
                    expected_idx = np.searchsorted(single_ep_tsd2.t, ts.t, side="left")
                    # avoid border errors with searchsorted
                    valid = expected_idx < len(single_ep_tsd2)
                    # check that times are actually before
                    assert np.all(single_ep_tsd2.t[expected_idx[valid]] >= ts2.t[valid])
                    # check that subsequent are after
                    assert np.all(single_ep_tsd2.t[expected_idx[1:] - 1] < ts2.t[1:])
                    expected_idx = expected_idx[valid]
                else:
                    before = np.searchsorted(single_ep_tsd2.t, ts.t, side="right") - 1
                    after = np.searchsorted(single_ep_tsd2.t, ts.t, side="left")
                    dt_before = np.abs(single_ep_tsd2.t[before] - ts.t)
                    # void border errors with searchsorted
                    valid = after < len(single_ep_tsd2)
                    dt_after = np.abs(single_ep_tsd2.t[after[valid]] - ts.t[valid])
                    expected_idx = before[valid].copy()
                    # by default if equi-distance, it assigned to after.
                    expected_idx[dt_after <= dt_before[valid]] = after[valid][
                        dt_after <= dt_before[valid]
                    ]
                np.testing.assert_array_equal(
                    single_ep_tsd2.d[expected_idx], ts2.d[valid]
                )
                np.testing.assert_array_equal(ts.t, ts2.t)

    def test_value_from_with_restrict(self, group):
        tsgroup = nap.TsGroup(group)
        tsd = nap.Tsd(t=np.arange(0, 300, 0.1), d=np.arange(3000))
        ep = nap.IntervalSet(start=0, end=100)
        tsgroup2 = tsgroup.value_from(tsd, ep)
        np.testing.assert_array_almost_equal(tsgroup2[0].values, np.arange(0, 1010, 10))
        np.testing.assert_array_almost_equal(tsgroup2[1].values, np.arange(0, 1005, 5))
        np.testing.assert_array_almost_equal(tsgroup2[2].values, np.arange(0, 1002, 2))

    @pytest.mark.parametrize(
        "ep",
        [
            None,
            nap.IntervalSet(start=0, end=50),
            nap.IntervalSet(start=0, end=100),
            nap.IntervalSet(start=0, end=300),
            nap.IntervalSet(start=[0, 120], end=[50, 221]),
            nap.IntervalSet(start=[20, 201], end=[150, 300]),
        ],
    )
    @pytest.mark.parametrize("bin_size", [None, 1.0, 1, 0.1])
    @pytest.mark.parametrize("metadata", [None, {"label": ["a", "b", "c"]}])
    def test_count(self, group, ep, bin_size, metadata):
        tsgroup = nap.TsGroup(group, time_support=ep, metadata=metadata)
        dt = np.sum(tsgroup.time_support.end - tsgroup.time_support.start)

        count = tsgroup.count(bin_size)

        res = [[] for _ in range(len(tsgroup))]
        for s, e in tsgroup.time_support.values:
            if (bin_size is None) or (bin_size > (e - s)):
                nbins = 2
                # add 1E-6 to make final bin inclusive, like jitrestrict_with_count
                bin_edges = np.array([s, e + 1e-6])
            else:
                # define bin edges like jitcount
                lbound = s
                bin_edges = [lbound]
                while lbound < e:
                    lbound += bin_size
                    lbound = np.round(lbound, 9)
                    bin_edges.append(lbound)

                nbins = int(np.ceil((e - s + bin_size) / bin_size))

            for u in tsgroup:
                # use digitize so last bin is closed on the right edge
                members = np.digitize(tsgroup[u].t, bin_edges)
                # exclude out of bounds bins
                members = members[(members > 0) & (members < nbins)]
                # ensure number of bins returned is nbins
                counts = np.bincount(members, minlength=nbins)
                res[u].extend(counts[1:])  # exclude bin 0 which is empty

        res = np.array(res).T
        np.testing.assert_array_almost_equal(count.values, res)
        # check metadata
        if metadata is not None:
            np.testing.assert_array_equal(
                tsgroup.get_info("label"), count.get_info("label")
            )

        # check dtype
        count = tsgroup.count(bin_size, dtype=np.int16)
        assert count.dtype == np.dtype(np.int16)

    @pytest.mark.parametrize(
        "ep",
        [
            None,
            nap.IntervalSet(start=0, end=50),
            nap.IntervalSet(start=0, end=100),
            nap.IntervalSet(start=0, end=300),
            nap.IntervalSet(start=[0, 120], end=[50, 221]),
            nap.IntervalSet(start=[20, 201], end=[150, 300]),
        ],
    )
    @pytest.mark.parametrize("bin_size", [None, 1.0, 1, 0.1])
    @pytest.mark.parametrize("metadata", [None, {"label": ["a", "b", "c"]}])
    def test_count_with_ep(self, group, ep, bin_size, metadata):
        tsgroup = nap.TsGroup(group, metadata=metadata)

        count = tsgroup.count(bin_size=bin_size, ep=ep)
        if ep is None:
            ep = tsgroup.time_support
        dt = np.sum(ep.end - ep.start)

        res = [[] for _ in range(len(tsgroup))]
        for s, e in ep.values:
            if (bin_size is None) or (bin_size > (e - s)):
                nbins = 2
                # add 1E-6 to make final bin inclusive, like jitrestrict_with_count
                bin_edges = np.array([s, e + 1e-6])
            else:
                # define bin edges like jitcount
                lbound = s
                bin_edges = [lbound]
                while lbound < e:
                    lbound += bin_size
                    lbound = np.round(lbound, 9)
                    bin_edges.append(lbound)

                nbins = int(np.ceil((e - s + bin_size) / bin_size))

            for u in tsgroup:
                # use digitize so last bin is closed on the right edge
                members = np.digitize(tsgroup[u].t, bin_edges)
                # exclude out of bounds bins
                members = members[(members > 0) & (members < nbins)]
                # ensure number of bins returned is nbins
                counts = np.bincount(members, minlength=nbins)
                res[u].extend(counts[1:])  # exclude bin 0 which is empty

        res = np.array(res).T
        np.testing.assert_array_almost_equal(count.values, res)
        # check metadata
        if metadata is not None:
            np.testing.assert_array_equal(
                tsgroup.get_info("label"), count.get_info("label")
            )

    def test_count_time_units(self, group):
        ep = nap.IntervalSet(start=0, end=100)
        tsgroup = nap.TsGroup(group, time_support=ep)
        for b, tu in zip([1, 1e3, 1e6], ["s", "ms", "us"]):
            count = tsgroup.count(b, time_units=tu)
            np.testing.assert_array_almost_equal(
                count.loc[0].values[0:-1].flatten(), np.ones(len(count) - 1)
            )
            np.testing.assert_array_almost_equal(
                count.loc[1].values[0:-1].flatten(), np.ones(len(count) - 1) * 2
            )
            np.testing.assert_array_almost_equal(
                count.loc[2].values[0:-1].flatten(), np.ones(len(count) - 1) * 5
            )

    def test_count_errors(self, group):
        tsgroup = nap.TsGroup(group)
        with pytest.raises(TypeError):
            tsgroup.count(bin_size={})

        with pytest.raises(TypeError):
            tsgroup.count(ep={})

        with pytest.raises(ValueError):
            tsgroup.count(bin_size=1, time_units={})

    def test_get_interval(self, group):
        tsgroup = nap.TsGroup(group)
        tsgroup2 = tsgroup.get(10, 20)

        assert isinstance(tsgroup2, nap.TsGroup)

        assert all(
            map(lambda x: len(x[0]) == x[1], zip(tsgroup2.values(), [11, 21, 51]))
        )

        for a, b in zip(tsgroup.values(), tsgroup2.values()):
            np.testing.assert_array_almost_equal(
                a.t[np.searchsorted(a.t, 10) : np.searchsorted(a.t, 20, "right")], b.t
            )

        np.testing.assert_array_almost_equal(
            tsgroup.time_support.values,
            tsgroup2.time_support.values,
        )

        tsgroup3 = tsgroup.get(10)

        assert isinstance(tsgroup2, nap.TsGroup)

        assert all(map(lambda x: len(x) == 1, tsgroup3.values()))

        assert all(map(lambda x: x.t[0] == 10.0, tsgroup3.values()))

        with pytest.raises(Exception):
            tsgroup.get(20, 10)

        with pytest.raises(Exception):
            tsgroup.get(10, [20])

        with pytest.raises(Exception):
            tsgroup.get([10], 20)

    def test_threshold_slicing(self, group):
        sr_info = pd.Series(index=[0, 1, 2], data=[0, 1, 2], name="sr")
        tsgroup = nap.TsGroup(group, sr=sr_info)
        assert tsgroup.getby_threshold("sr", 1).keys() == [2]
        assert tsgroup.getby_threshold("sr", 1, ">").keys() == [2]
        assert tsgroup.getby_threshold("sr", 1, "<").keys() == [0]
        assert tsgroup.getby_threshold("sr", 1, ">=").keys() == [1, 2]
        assert tsgroup.getby_threshold("sr", 1, "<=").keys() == [0, 1]

    def test_threshold_error(self, group):
        sr_info = pd.Series(index=[0, 1, 2], data=[0, 1, 2], name="sr")
        tsgroup = nap.TsGroup(group, sr=sr_info)
        op = "!="
        with pytest.raises(RuntimeError) as e_info:
            tsgroup.getby_threshold("sr", 1, op)
        assert str(e_info.value) == "Operation {} not recognized.".format(op)

    def test_intervals_slicing(self, group):
        sr_info = pd.Series(index=[0, 1, 2], data=[0, 1, 2], name="sr")
        tsgroup = nap.TsGroup(group, sr=sr_info)
        lgroup, bincenter = tsgroup.getby_intervals("sr", [0, 1, 2])
        np.testing.assert_array_almost_equal(bincenter, np.array([0.5, 1.5]))
        assert lgroup[0].keys() == [0]
        assert lgroup[1].keys() == [1]

    def test_category_slicing(self, group):
        sr_info = pd.Series(index=[0, 1, 2], data=["a", "a", "b"], name="sr")
        tsgroup = nap.TsGroup(group, sr=sr_info)
        dgroup = tsgroup.getby_category("sr")
        assert isinstance(dgroup, dict)
        assert list(dgroup.keys()) == ["a", "b"]
        assert dgroup["a"].keys() == [0, 1]
        assert dgroup["b"].keys() == [2]

    def test_to_tsd(self, group):
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

        tsgroup = nap.TsGroup(group)

        tsd = tsgroup.to_tsd()

        np.testing.assert_array_almost_equal(tsd.index, times)
        np.testing.assert_array_almost_equal(tsd.values, data)

        alpha = np.random.randn(3)
        tsgroup.set_info(alpha=alpha)
        tsd2 = tsgroup.to_tsd("alpha")
        np.testing.assert_array_almost_equal(tsd2.index, times)
        np.testing.assert_array_almost_equal(
            tsd2.values, np.array([alpha[int(i)] for i in data])
        )

        tsd3 = tsgroup.to_tsd(alpha)
        np.testing.assert_array_almost_equal(tsd3.index, times)
        np.testing.assert_array_almost_equal(
            tsd3.values, np.array([alpha[int(i)] for i in data])
        )

        beta = pd.Series(index=np.arange(3), data=np.random.randn(3))
        tsd4 = tsgroup.to_tsd(beta)
        np.testing.assert_array_almost_equal(tsd4.index, times)
        np.testing.assert_array_almost_equal(
            tsd4.values, np.array([beta[int(i)] for i in data])
        )

    def test_to_tsd_runtime_errors(self, group):
        tsgroup = nap.TsGroup(group)

        with pytest.raises(Exception) as e_info:
            tsgroup.to_tsd(
                pd.Series(
                    index=np.arange(len(tsgroup) + 1), data=np.arange(len(tsgroup) + 1)
                )
            )
        assert str(e_info.value) == "Index are not equals"

        with pytest.raises(Exception) as e_info:
            tsgroup.to_tsd(np.arange(len(tsgroup) + 1))
        assert str(e_info.value) == "Values is not the same length."

        with pytest.raises(Exception) as e_info:
            tsgroup.to_tsd("error")
        assert str(e_info.value) == "Key error not in metadata of TsGroup"

        with pytest.raises(Exception) as e_info:
            tsgroup.to_tsd(dict)
        assert (
            str(e_info.value)
            == """Unknown argument format. Must be pandas.Series, numpy.ndarray or a string from one of the following values : [rate]"""
        )

        tsgroup.set_info(alpha=np.random.rand(len(tsgroup)))

        with pytest.raises(Exception) as e_info:
            tsgroup.to_tsd(dict)
        assert (
            str(e_info.value)
            == """Unknown argument format. Must be pandas.Series, numpy.ndarray or a string from one of the following values : [rate, alpha]"""
        )

    def test_trial_count(self, group):
        tsgroup = nap.TsGroup(group)
        ep = nap.IntervalSet(
            start=np.arange(0, 100, 20), end=np.arange(0, 100, 20) + np.arange(0, 10, 2)
        )

        expected = np.ones((len(group), len(ep), 8)) * np.nan
        for i, k in zip(range(len(ep)), range(2, 10, 2)):
            expected[:, i, 0:k] = 1
        for i, k in zip(range(len(group)), [1, 2, 5]):
            expected[i] *= k

        tensor = tsgroup.trial_count(ep, bin_size=1)
        np.testing.assert_array_almost_equal(tensor, expected)

        tensor = tsgroup[[0]].trial_count(ep, bin_size=1)
        np.testing.assert_array_almost_equal(tensor, expected[0:1])

        tensor = tsgroup.trial_count(ep, bin_size=1, align="start")
        np.testing.assert_array_almost_equal(tensor, expected)

        tensor = tsgroup.trial_count(ep, bin_size=1, align="end")
        np.testing.assert_array_almost_equal(tensor, np.flip(expected, axis=2))

        tensor = tsgroup.trial_count(ep, bin_size=1, time_unit="s")
        np.testing.assert_array_almost_equal(tensor, expected)

        tensor = tsgroup.trial_count(ep, bin_size=1e3, time_unit="ms")
        np.testing.assert_array_almost_equal(tensor, expected)

        tensor = tsgroup.trial_count(ep, bin_size=1e6, time_unit="us")
        np.testing.assert_array_almost_equal(tensor, expected)

        tensor = tsgroup.trial_count(ep, bin_size=1, align="start", padding_value=-1)
        expected[np.isnan(expected)] = -1
        np.testing.assert_array_almost_equal(tensor, expected)

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
        self, group, ep, bin_size, align, padding_value, time_unit, expectation
    ):
        tsgroup = nap.TsGroup(group)
        with pytest.raises(RuntimeError, match=re.escape(expectation)):
            tsgroup.trial_count(ep, bin_size, align, padding_value, time_unit)

    @pytest.mark.parametrize(
        "align, expectation",
        [
            ("a", "align should be 'start', 'center' or 'end'"),
        ],
    )
    def test_time_diff_runtime_errors(self, group, align, expectation):
        tsgroup = nap.TsGroup(group)
        with pytest.raises(RuntimeError, match=re.escape(expectation)):
            tsgroup.time_diff(align=align)

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
    def test_time_diff_epoch_error(self, group, epochs, expectation):
        tsgroup = nap.TsGroup(group)
        with expectation:
            tsgroup.time_diff(epochs=epochs)

    @pytest.mark.parametrize(
        "align, epochs, expectation",
        [
            # default arguments
            (
                None,
                None,
                {
                    0: nap.Tsd(d=np.ones(199), t=np.arange(0.5, 199.5)),
                    1: nap.Tsd(d=np.full(399, 0.5), t=np.arange(0.25, 199.5, 0.5)),
                    2: nap.Tsd(d=np.full(1499, 0.2), t=np.arange(0.1, 299.8, 0.2)),
                },
            ),
            # empty time support
            (
                "start",
                nap.IntervalSet(start=[], end=[]),
                {i: nap.Tsd(d=[], t=[]) for i in range(3)},
            ),
            # empty epochs
            (
                "start",
                nap.IntervalSet(start=[10, 50, 300], end=[20, 60, 310]),
                {
                    0: nap.Tsd(
                        d=np.ones(20),
                        t=np.concatenate([np.arange(10, 20), np.arange(50, 60)]),
                    ),
                    1: nap.Tsd(
                        d=np.full(40, 0.5),
                        t=np.concatenate(
                            [np.arange(10, 20, 0.5), np.arange(50, 60, 0.5)]
                        ),
                    ),
                    2: nap.Tsd(
                        d=np.full(100, 0.2),
                        t=np.concatenate(
                            [np.arange(10, 20, 0.2), np.arange(50, 60, 0.2)]
                        ),
                    ),
                },
            ),
            # single epoch
            (
                "start",
                nap.IntervalSet(start=[10, 50]),
                {
                    0: nap.Tsd(d=np.ones(40), t=np.arange(10, 50)),
                    1: nap.Tsd(d=np.full(80, 0.5), t=np.arange(10, 50, 0.5)),
                    2: nap.Tsd(d=np.full(200, 0.2), t=np.arange(10, 50, 0.2)),
                },
            ),
            # single point in epochs
            (
                "start",
                nap.IntervalSet(start=[10, 50, 299.8], end=[20, 60, 300]),
                {
                    0: nap.Tsd(
                        d=np.ones(20),
                        t=np.concatenate([np.arange(10, 20), np.arange(50, 60)]),
                    ),
                    1: nap.Tsd(
                        d=np.full(40, 0.5),
                        t=np.concatenate(
                            [np.arange(10, 20, 0.5), np.arange(50, 60, 0.5)]
                        ),
                    ),
                    2: nap.Tsd(
                        d=np.full(100, 0.2),
                        t=np.concatenate(
                            [np.arange(10, 20, 0.2), np.arange(50, 60, 0.2)]
                        ),
                    ),
                },
            ),
            # two points in epochs
            (
                "start",
                nap.IntervalSet(start=[10, 50, 299.6], end=[20, 60, 300]),
                {
                    0: nap.Tsd(
                        d=np.ones(20),
                        t=np.concatenate([np.arange(10, 20), np.arange(50, 60)]),
                    ),
                    1: nap.Tsd(
                        d=np.full(40, 0.5),
                        t=np.concatenate(
                            [np.arange(10, 20, 0.5), np.arange(50, 60, 0.5)]
                        ),
                    ),
                    2: nap.Tsd(
                        d=np.full(101, 0.2),
                        t=np.concatenate(
                            [np.arange(10, 20, 0.2), np.arange(50, 60, 0.2), [299.6]]
                        ),
                    ),
                },
            ),
        ],
    )
    def test_time_diff(self, group, align, epochs, expectation):
        tsgroup = nap.TsGroup(group)

        if align is None:
            actual = tsgroup.time_diff(epochs=epochs)
        else:
            actual = tsgroup.time_diff(align=align, epochs=epochs)

        assert isinstance(actual, dict)
        assert len(actual) == len(tsgroup)
        assert len(actual) == len(expectation)

        for ts_idx in tsgroup.index:
            assert ts_idx in actual
            assert isinstance(actual[ts_idx], nap.Tsd)
            np.testing.assert_array_almost_equal(
                actual[ts_idx].times(), expectation[ts_idx].times()
            )
            np.testing.assert_array_almost_equal(
                actual[ts_idx].values, expectation[ts_idx].values
            )

    def test_save_npz(self, group):
        group = {
            0: nap.Tsd(t=np.arange(0, 20), d=np.random.rand(20)),
            1: nap.Tsd(t=np.arange(0, 20, 0.5), d=np.random.rand(40)),
            2: nap.Tsd(t=np.arange(0, 10, 0.2), d=np.random.rand(50)),
        }

        tsgroup = nap.TsGroup(
            group,
            meta=np.arange(len(group), dtype=np.int64),
            meta2=np.array(["a", "b", "c"]),
        )

        with pytest.raises(TypeError) as e:
            tsgroup.save(dict)

        with pytest.raises(RuntimeError) as e:
            tsgroup.save("./")
        assert str(e.value) == "Invalid filename input. {} is directory.".format(
            Path("./").resolve()
        )

        fake_path = "./fake/path"
        with pytest.raises(RuntimeError) as e:
            tsgroup.save(fake_path + "/file.npz")
        assert str(e.value) == "Path {} does not exist.".format(
            Path(fake_path).resolve()
        )

        tsgroup.save("tsgroup.npz")
        assert "tsgroup.npz" in [f.name for f in Path(".").iterdir()]

        tsgroup.save("tsgroup2")
        assert "tsgroup2.npz" in [f.name for f in Path(".").iterdir()]

        file = np.load("tsgroup.npz", allow_pickle=True)

        keys = list(file.keys())
        for k in ["t", "d", "start", "end", "index", "_metadata"]:
            assert k in keys

        metadata = pd.DataFrame.from_dict(file["_metadata"].item())
        for k in ["meta", "meta2"]:
            assert k in metadata.keys()

        times = []
        index = []
        data = []
        for n in group.keys():
            times.append(group[n].index)
            index.append(np.ones(len(group[n])) * n)
            data.append(group[n].values)
        times = np.hstack(times)
        index = np.hstack(index)
        data = np.hstack(data)
        idx = np.argsort(times)
        times = times[idx]
        data = data[idx]
        index = index[idx]

        np.testing.assert_array_almost_equal(file["start"], tsgroup.time_support.start)
        np.testing.assert_array_almost_equal(file["end"], tsgroup.time_support.end)
        np.testing.assert_array_almost_equal(file["t"], times)
        np.testing.assert_array_almost_equal(file["d"], data)
        np.testing.assert_array_almost_equal(file["index"], index)
        np.testing.assert_array_almost_equal(
            metadata["meta"], np.arange(len(group), dtype=np.int64)
        )
        assert np.all(metadata["meta2"] == np.array(["a", "b", "c"]))
        file.close()

        tsgroup3 = nap.TsGroup(
            {
                0: nap.Ts(t=np.arange(0, 20)),
            }
        )
        tsgroup3.save("tsgroup3")

        with np.load("tsgroup3.npz") as file:
            assert "d" not in list(file.keys())
            np.testing.assert_array_almost_equal(file["t"], tsgroup3[0].index)

        Path("tsgroup.npz").unlink()
        Path("tsgroup2.npz").unlink()
        Path("tsgroup3.npz").unlink()

    @pytest.mark.parametrize(
        "keys, expectation",
        [
            (1, does_not_raise()),
            ([1, 2], does_not_raise()),
            ([1, 2], does_not_raise()),
            (np.array([1, 2]), does_not_raise()),
            (np.array([False, True, True]), does_not_raise()),
            ([False, True, True], does_not_raise()),
            (True, does_not_raise()),
            (4, pytest.raises(KeyError, match="Key 4 not in group index.")),
            (
                [3, 4],
                pytest.raises(KeyError, match=r"Key \[3, 4\] not in group index."),
            ),
            ([2, 3], pytest.raises(KeyError, match=r"Key \[3\] not in group index.")),
        ],
    )
    def test_indexing_type(self, group, keys, expectation):
        ts_group = nap.TsGroup(group)
        with expectation:
            out = ts_group[keys]

    def test_setitem_metadata_vals(self, group):
        group = nap.TsGroup(group)
        group["a"] = np.arange(len(group))
        assert all(group._metadata["a"] == np.arange(len(group)))

    def test_setitem_metadata_twice(self, group):
        group = nap.TsGroup(group)
        group["a"] = np.arange(len(group))
        group["a"] = np.arange(len(group)) + 10
        assert all(group._metadata["a"] == np.arange(len(group)) + 10)

    def test_getitem_ts_object(self, ts_group):
        assert isinstance(ts_group[1], nap.Ts)

    @pytest.mark.parametrize(
        "bool_idx",
        [
            [True, False],
            [False, True],
            [True, True],
            np.array([True, False], dtype=bool),
            np.array([False, True], dtype=bool),
            np.array([True, True], dtype=bool),
        ],
    )
    def test_getitem_bool_indexing(self, bool_idx, ts_group):
        out = ts_group[bool_idx]
        assert isinstance(out, nap.TsGroup)
        assert len(out) == sum(bool_idx)
        idx = np.where(bool_idx)[0]
        if len(idx) == 1:
            slc = slice(idx[0], idx[0] + 1)
        else:
            slc = slice(0, 2)
        assert all(out.keys()[i] == ts_group.keys()[slc][i] for i in range(len(idx)))
        for key_i in np.where(bool_idx)[0]:
            key = ts_group.keys()[key_i]
            assert np.all(out[[key]].rates == ts_group._metadata.loc[key]["rate"])
            assert np.all(out[[key]].meta == ts_group._metadata.loc[key]["meta"])
            assert np.all(out[key].t == ts_group[key].t)

    @pytest.mark.parametrize(
        "idx",
        [
            [1],
            [2],
            [1, 2],
            [2, 1],
            np.array([1]),
            np.array([2]),
            np.array([1, 2]),
            np.array([2, 1]),
        ],
    )
    def test_getitem_int_indexing(self, idx, ts_group):
        out = ts_group[idx]
        # check that sorting keys doesn't make a diff
        srt_idx = np.sort(idx)
        assert isinstance(out, nap.TsGroup)
        assert np.all(out.rates == ts_group[srt_idx].rates)
        assert np.all(out.meta == ts_group[srt_idx].meta)
        for k in idx:
            assert np.all(out[k].t == ts_group[k].t)

    def test_getitem_metadata_direct(self, ts_group):
        assert np.all(ts_group.rates == np.array([10 / 9, 5 / 9]))

    def test_getitem_key_error(self, ts_group):
        with pytest.raises(KeyError, match="Key nonexistent not in group index."):
            _ = ts_group["nonexistent"]

    def test_getitem_attribute_error(self, ts_group):
        with pytest.raises(AttributeError, match="'TsGroup' object has no attribute"):
            _ = ts_group.nonexistent_metadata

    @pytest.mark.parametrize(
        "bool_idx, expectation",
        [
            (
                np.ones((3,), dtype=bool),
                pytest.raises(IndexError, match="Boolean index length must be equal"),
            ),
            (
                np.ones((2, 1), dtype=bool),
                pytest.raises(IndexError, match="Only 1-dimensional boolean indices"),
            ),
            (
                np.array(True),
                pytest.raises(IndexError, match="Only 1-dimensional boolean indices"),
            ),
        ],
    )
    def test_getitem_boolean_fail(self, ts_group, bool_idx, expectation):
        with expectation:
            out = ts_group[bool_idx]

    def test_merge_complete(self, ts_group):
        with pytest.raises(TypeError, match="Input at positions(.*)are not TsGroup!"):
            nap.TsGroup.merge_group(ts_group, str, dict)

        ts_group2 = nap.TsGroup(
            {
                3: nap.Ts(t=np.arange(15)),
                4: nap.Ts(t=np.arange(20)),
            },
            time_support=ts_group.time_support,
            meta=np.array([12, 13]),
        )
        merged = ts_group.merge(ts_group2)
        assert len(merged) == 4
        assert np.all(merged.keys() == np.array([1, 2, 3, 4]))
        assert np.all(merged.meta == np.array([10, 11, 12, 13]))
        np.testing.assert_equal(merged.metadata_columns, ts_group.metadata_columns)

    @pytest.mark.parametrize(
        "col_name, ignore_metadata, expectation",
        [
            ("meta", False, does_not_raise()),
            ("meta", True, does_not_raise()),
            (
                "wrong_name",
                False,
                pytest.raises(
                    ValueError,
                    match="TsGroup at position 2 has different metadata columns.*",
                ),
            ),
            ("wrong_name", True, does_not_raise()),
        ],
    )
    def test_merge_metadata(self, ts_group, col_name, ignore_metadata, expectation):
        metadata = pd.DataFrame([12, 13], index=[3, 4], columns=[col_name])
        ts_group2 = nap.TsGroup(
            {
                3: nap.Ts(t=np.arange(15)),
                4: nap.Ts(t=np.arange(20)),
            },
            time_support=ts_group.time_support,
            **metadata,
        )

        with expectation:
            merged = ts_group.merge(ts_group2, ignore_metadata=ignore_metadata)

        if ignore_metadata:
            assert merged.metadata_columns[0] == "rate"
        elif col_name == "meta":
            np.testing.assert_equal(merged.metadata_columns, ts_group.metadata_columns)

    @pytest.mark.parametrize(
        "index, reset_index, expectation",
        [
            (
                np.array([1, 2]),
                False,
                pytest.raises(
                    ValueError, match="TsGroup at position 2 has overlapping keys.*"
                ),
            ),
            (np.array([1, 2]), True, does_not_raise()),
            (np.array([3, 4]), False, does_not_raise()),
            (np.array([3, 4]), True, does_not_raise()),
        ],
    )
    def test_merge_index(self, ts_group, index, reset_index, expectation):
        ts_group2 = nap.TsGroup(
            dict(zip(index, [nap.Ts(t=np.arange(15)), nap.Ts(t=np.arange(20))])),
            time_support=ts_group.time_support,
            meta=np.array([12, 13]),
        )

        with expectation:
            merged = ts_group.merge(ts_group2, reset_index=reset_index)

        if reset_index:
            assert np.all(merged.keys() == np.arange(4))
        elif np.all(index == np.array([3, 4])):
            assert np.all(merged.keys() == np.array([1, 2, 3, 4]))

    @pytest.mark.parametrize(
        "time_support, reset_time_support, expectation",
        [
            (None, False, does_not_raise()),
            (None, True, does_not_raise()),
            (
                nap.IntervalSet(start=0, end=1),
                False,
                pytest.raises(
                    ValueError,
                    match="TsGroup at position 2 has different time support.*",
                ),
            ),
            (nap.IntervalSet(start=0, end=1), True, does_not_raise()),
        ],
    )
    def test_merge_time_support(
        self, ts_group, time_support, reset_time_support, expectation
    ):
        if time_support is None:
            time_support = ts_group.time_support

        ts_group2 = nap.TsGroup(
            {
                3: nap.Ts(t=np.arange(15)),
                4: nap.Ts(t=np.arange(20)),
            },
            time_support=time_support,
            meta=np.array([12, 13]),
        )

        with expectation:
            merged = ts_group.merge(ts_group2, reset_time_support=reset_time_support)

        if reset_time_support:
            np.testing.assert_array_almost_equal(
                ts_group.time_support.as_units("s").to_numpy(),
                merged.time_support.as_units("s").to_numpy(),
            )


def test_pickling(ts_group):
    """Test that pikling works as expected."""
    # pickle and unpickle ts_group
    pickled_obj = pickle.dumps(ts_group)
    unpickled_obj = pickle.loads(pickled_obj)

    # Ensure the type is the same
    assert type(ts_group) is type(unpickled_obj), "Types are different"

    # Ensure that TsGroup have same len
    assert len(ts_group) == len(unpickled_obj)

    # Ensure that metadata content is the same
    assert np.all(unpickled_obj._metadata.keys() == ts_group._metadata.keys())
    for key in ts_group._metadata.keys():
        assert np.all(unpickled_obj._metadata[key] == ts_group._metadata[key])

    # Ensure that metadata columns are the same
    assert np.all(unpickled_obj._metadata.columns == ts_group._metadata.columns)

    # Ensure that the Ts are the same
    assert all(
        [
            np.all(ts_group[key].t == unpickled_obj[key].t)
            for key in unpickled_obj.keys()
        ]
    )

    # Ensure time support is the same
    assert np.all(ts_group.time_support == unpickled_obj.time_support)


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
def test_count_dtype(dtype, expectation, ts_group, ts_group_one_group):
    with expectation:
        count = ts_group.count(bin_size=0.1, dtype=dtype)
        count_one = ts_group_one_group.count(bin_size=0.1, dtype=dtype)
        if dtype:
            assert np.issubdtype(count.dtype, dtype)
            assert np.issubdtype(count_one.dtype, dtype)
