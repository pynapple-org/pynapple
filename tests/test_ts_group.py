

"""Tests of ts group for `pynapple` package."""

import pickle
import warnings
from collections import UserDict
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

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
            ({"1": nap.Ts(np.arange(10)), "2": nap.Ts(np.arange(10))}, does_not_raise()),
            ({"1": nap.Ts(np.arange(10)), 2: nap.Ts(np.arange(10))}, does_not_raise()),
            ({"1": nap.Ts(np.arange(10)), 1: nap.Ts(np.arange(10))},
             pytest.raises(ValueError, match="Two dictionary keys contain the same integer")),
            ({"1.": nap.Ts(np.arange(10)), 2: nap.Ts(np.arange(10))},
             pytest.raises(ValueError, match="All keys must be convertible")),
            ({-1: nap.Ts(np.arange(10)), 1: nap.Ts(np.arange(10))}, does_not_raise()),
            ({1.5: nap.Ts(np.arange(10)), 1: nap.Ts(np.arange(10))},
             pytest.raises(ValueError, match="All keys must have integer value"))

        ]
    )
    def test_initialize_from_dict(self, test_dict, expectation):
        with expectation:
            nap.TsGroup(test_dict)

    @pytest.mark.parametrize(
        "tsgroup",
        [
            nap.TsGroup({"1": nap.Ts(np.arange(10)), "2": nap.Ts(np.arange(10))}),
            nap.TsGroup({"1": nap.Ts(np.arange(10)), 2: nap.Ts(np.arange(10))}),
            nap.TsGroup({-1: nap.Ts(np.arange(10)), 1: nap.Ts(np.arange(10))})

        ]
    )
    def test_metadata_len_match(self, tsgroup):
        assert len(tsgroup._metadata) == len(tsgroup)

    def test_create_ts_group_from_array(self):
        with warnings.catch_warnings(record=True) as w:
            nap.TsGroup({
                0: np.arange(0, 200),
                1: np.arange(0, 200, 0.5),
                2: np.arange(0, 300, 0.2),
                })
        assert str(w[0].message) == "Elements should not be passed as <class 'numpy.ndarray'>. Default time units is seconds when creating the Ts object."

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
            tmp = nap.TsGroup({
                0: nap.Ts(t=np.array([])),
                1: nap.Ts(t=np.array([])),
                2: nap.Ts(t=np.array([])),
                })
        assert str(e_info.value) == "Union of time supports is empty. Consider passing a time support as argument."

    def test_create_ts_group_with_bypass_check(self):
        tmp = {
            0: nap.Ts(t=np.arange(0, 100)),
            1: nap.Ts(t=np.arange(0, 200, 0.5), time_units="s"),
            2: nap.Ts(t=np.arange(0, 300, 0.2), time_units="s")
        }
        tsgroup = nap.TsGroup(tmp, time_support = nap.IntervalSet(0, 100), bypass_check=True)
        for i in tmp.keys():
            np.testing.assert_array_almost_equal(tmp[i].index, tsgroup[i].index)

        tmp = {
            0: nap.Ts(t=np.arange(0, 100)),
            1: nap.Ts(t=np.arange(0, 200, 0.5), time_units="s"),
            2: nap.Ts(t=np.arange(0, 300, 0.2), time_units="s")
        }
        tsgroup = nap.TsGroup(tmp, bypass_check=True)
        for i in tmp.keys():
            np.testing.assert_array_almost_equal(tmp[i].index, tsgroup[i].index)            

    def test_create_ts_group_with_metainfo(self, group):
        sr_info = pd.Series(index=[0, 1, 2], data=[0, 0, 0], name="sr")
        ar_info = np.ones(3) * 1
        tsgroup = nap.TsGroup(group, sr=sr_info, ar=ar_info)
        assert tsgroup._metadata.shape == (3, 3)
        np.testing.assert_array_almost_equal(tsgroup._metadata["sr"].values, sr_info.values)
        np.testing.assert_array_almost_equal(tsgroup._metadata["sr"].index.values, sr_info.index.values)
        np.testing.assert_array_almost_equal(tsgroup._metadata["ar"].values, ar_info)

    def test_add_metainfo(self, group):
        tsgroup = nap.TsGroup(group)
        df_info = pd.DataFrame(index=[0, 1, 2], data=[0, 0, 0], columns=["df"])
        sr_info = pd.Series(index=[0, 1, 2], data=[1, 1, 1], name="sr")
        ar_info = np.ones(3) * 3
        lt_info = [3,4,5]
        tu_info = (6,8,3)
        tsgroup.set_info(df_info, sr=sr_info, ar=ar_info, lt=lt_info, tu=tu_info)
        assert tsgroup._metadata.shape == (3, 6)
        pd.testing.assert_series_equal(tsgroup._metadata["df"], df_info["df"])
        pd.testing.assert_series_equal(tsgroup._metadata["sr"], sr_info)
        np.testing.assert_array_almost_equal(tsgroup._metadata["ar"].values, ar_info)
        np.testing.assert_array_almost_equal(tsgroup._metadata["lt"].values, lt_info)
        np.testing.assert_array_almost_equal(tsgroup._metadata["tu"].values, tu_info)

    def test_add_metainfo_raise_error(self, group):
        tsgroup = nap.TsGroup(group)
        df_info = pd.DataFrame(index=[4, 5, 6], data=[0, 0, 0], columns=["df"])

        with pytest.raises(RuntimeError) as e_info:
            tsgroup.set_info(df_info)
        assert str(e_info.value) == "Index are not equals"

        tsgroup = nap.TsGroup(group)
        sr_info = pd.Series(index=[0, 1, 2], data=[1, 1, 1])

        with pytest.raises(RuntimeError) as e_info:
            tsgroup.set_info(sr_info)
        assert str(e_info.value) == "Argument should be passed as keyword argument."

        tsgroup = nap.TsGroup(group)
        ar_info = np.ones(3) * 3

        with pytest.raises(RuntimeError) as e_info:
            tsgroup.set_info(ar_info)
        assert str(e_info.value) == "Argument should be passed as keyword argument."

    def test_add_metainfo_test_runtime_errors(self, group):
        tsgroup = nap.TsGroup(group)
        sr_info = pd.Series(index=[1, 2, 3], data=[1, 1, 1], name="sr")
        with pytest.raises(Exception) as e_info:
            tsgroup.set_info(sr=sr_info)
        assert str(e_info.value) == "Index are not equals for argument sr"
        df_info = pd.DataFrame(index=[1, 2, 3], data=[1, 1, 1], columns=["df"])
        with pytest.raises(Exception) as e_info:
            tsgroup.set_info(df_info)
        assert str(e_info.value) == "Index are not equals"

        sr_info = pd.Series(index=[1, 2, 3], data=[1, 1, 1], name="sr")
        with pytest.raises(Exception) as e_info:
            tsgroup.set_info(sr_info)
        assert str(e_info.value) == "Argument should be passed as keyword argument."

        ar_info = np.ones(4)
        with pytest.raises(Exception) as e_info:
            tsgroup.set_info(ar=ar_info)
        assert str(e_info.value) == "Array is not the same length."

    def test_keys(self, group):
        tsgroup = nap.TsGroup(group)
        assert tsgroup.keys() == [0, 1, 2]

    def test_rates_property(self, group):
        tsgroup = nap.TsGroup(group)
        pd.testing.assert_series_equal(tsgroup.rates, tsgroup._metadata['rate'])

    def test_items(self, group):
        tsgroup = nap.TsGroup(group)
        items = tsgroup.items()
        assert isinstance(items, list)
        for i,it in items:
            pd.testing.assert_series_equal(tsgroup[i].as_series(), it.as_series())

    def test_items(self, group):
        tsgroup = nap.TsGroup(group)
        values = tsgroup.values()
        assert isinstance(values, list)
        for i,it in enumerate(values):
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

    def test_get_info(self, group):
        tsgroup = nap.TsGroup(group)
        df_info = pd.DataFrame(index=[0, 1, 2], data=[0, 0, 0], columns=["df"])
        sr_info = pd.Series(index=[0, 1, 2], data=[1, 1, 1], name="sr")
        ar_info = np.ones(3) * 3
        tsgroup.set_info(df_info, sr=sr_info, ar=ar_info)
        pd.testing.assert_series_equal(tsgroup.get_info("df"), df_info["df"])
        pd.testing.assert_series_equal(tsgroup.get_info("sr"), sr_info)
        np.testing.assert_array_almost_equal(tsgroup.get_info("ar").values, ar_info)

    def test_get_rate(self, group):
        tsgroup = nap.TsGroup(group)
        rate = tsgroup._metadata["rate"]
        pd.testing.assert_series_equal(rate, tsgroup.get_info("rate"))
        pd.testing.assert_series_equal(rate, tsgroup.get_info("freq"))
        pd.testing.assert_series_equal(rate, tsgroup.get_info("frequency"))

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

    def test_value_from_with_restrict(self, group):
        tsgroup = nap.TsGroup(group)
        tsd = nap.Tsd(t=np.arange(0, 300, 0.1), d=np.arange(3000))
        ep = nap.IntervalSet(start=0, end=100)
        tsgroup2 = tsgroup.value_from(tsd, ep)
        np.testing.assert_array_almost_equal(tsgroup2[0].values, np.arange(0, 1010, 10))
        np.testing.assert_array_almost_equal(tsgroup2[1].values, np.arange(0, 1005, 5))
        np.testing.assert_array_almost_equal(tsgroup2[2].values, np.arange(0, 1002, 2))

    def test_count(self, group):
        ep = nap.IntervalSet(start=0, end=100)
        tsgroup = nap.TsGroup(group, time_support=ep)
        count = tsgroup.count(1.0)
        np.testing.assert_array_almost_equal(
            count.loc[0].values[0:-1].flatten(), np.ones(len(count) - 1)
        )
        np.testing.assert_array_almost_equal(
            count.loc[1].values[0:-1].flatten(), np.ones(len(count) - 1) * 2
        )
        np.testing.assert_array_almost_equal(
            count.loc[2].values[0:-1].flatten(), np.ones(len(count) - 1) * 5
        )

        count = tsgroup.count(1)
        np.testing.assert_array_almost_equal(
            count.loc[0].values[0:-1].flatten(), np.ones(len(count) - 1)
        )
        np.testing.assert_array_almost_equal(
            count.loc[1].values[0:-1].flatten(), np.ones(len(count) - 1) * 2
        )
        np.testing.assert_array_almost_equal(
            count.loc[2].values[0:-1].flatten(), np.ones(len(count) - 1) * 5
        )

        count = tsgroup.count()
        np.testing.assert_array_almost_equal(count.values, np.array([[101, 201, 501]]))

        count = tsgroup.count(1.0, dtype=np.int16)
        assert count.dtype == np.dtype(np.int16)

    def test_count_with_ep(self, group):
        ep = nap.IntervalSet(start=0, end=100)
        tsgroup = nap.TsGroup(group)
        count = tsgroup.count(1.0, ep)
        np.testing.assert_array_almost_equal(
            count.loc[0].values[0:-1].flatten(), np.ones(len(count) - 1)
        )
        np.testing.assert_array_almost_equal(
            count.loc[1].values[0:-1].flatten(), np.ones(len(count) - 1) * 2
        )
        np.testing.assert_array_almost_equal(
            count.loc[2].values[0:-1].flatten(), np.ones(len(count) - 1) * 5
        )
        count = tsgroup.count(bin_size=1.0, ep=ep)
        np.testing.assert_array_almost_equal(
            count.loc[0].values[0:-1].flatten(), np.ones(len(count) - 1)
        )
        np.testing.assert_array_almost_equal(
            count.loc[1].values[0:-1].flatten(), np.ones(len(count) - 1) * 2
        )
        np.testing.assert_array_almost_equal(
            count.loc[2].values[0:-1].flatten(), np.ones(len(count) - 1) * 5
        )
        count = tsgroup.count(ep=nap.IntervalSet(0, 50))
        np.testing.assert_array_almost_equal(count.values, np.array([[51, 101, 251]]))

    def test_count_time_units(self, group):
        ep = nap.IntervalSet(start=0, end=100)
        tsgroup = nap.TsGroup(group, time_support =ep)
        for b, tu in zip([1, 1e3, 1e6],['s', 'ms', 'us']):
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
            count = tsgroup.count(b, tu)
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
        with pytest.raises(ValueError):
            tsgroup.count(bin_size = {})

        with pytest.raises(ValueError):
            tsgroup.count(ep = {})

        with pytest.raises(ValueError):
            tsgroup.count(time_units = {})

    def test_get_interval(self, group):
        tsgroup = nap.TsGroup(group)
        tsgroup2 = tsgroup.get(10, 20)

        assert isinstance(tsgroup2, nap.TsGroup)

        assert all(map(lambda x: len(x[0]) == x[1], zip(tsgroup2.values(), [11, 21, 51])))

        for a, b in zip(tsgroup.values(), tsgroup2.values()):
            np.testing.assert_array_almost_equal(
                a.t[np.searchsorted(a.t, 10):np.searchsorted(a.t, 20, 'right')],
                b.t)

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

    def test_repr_(self, group):
        from tabulate import tabulate

        tsgroup = nap.TsGroup(group)
        tsgroup.set_info(abc = ['a']*len(tsgroup))
        tsgroup.set_info(bbb = [1]*len(tsgroup))
        tsgroup.set_info(ccc = [np.pi]*len(tsgroup))

        cols = tsgroup._metadata.columns.drop("rate")
        headers = ["Index", "rate"] + [c for c in cols]
        lines = []

        def round_if_float(x):
            if isinstance(x, float):
                return np.round(x, 5)
            else:
                return x

        for i in tsgroup.index:
            lines.append(
                [str(i), np.round(tsgroup._metadata.loc[i, "rate"], 5)]
                + [round_if_float(tsgroup._metadata.loc[i, c]) for c in cols]
            )
        assert tabulate(lines, headers=headers) == tsgroup.__repr__()

    def test_str_(self, group):
        tsgroup = nap.TsGroup(group)        
        assert tsgroup.__str__() == tsgroup.__repr__()
        
    def test_to_tsd(self, group):    
        t = []
        d = []
        group = {}
        for i in range(3):
            t.append(np.sort(np.random.rand(10)*100))
            d.append(np.ones(10)*i)
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

        alpha=np.random.randn(3)
        tsgroup.set_info(alpha=alpha)
        tsd2 = tsgroup.to_tsd("alpha")
        np.testing.assert_array_almost_equal(tsd2.index, times)
        np.testing.assert_array_almost_equal(tsd2.values, np.array([alpha[int(i)] for i in data]))

        tsd3 = tsgroup.to_tsd(alpha)
        np.testing.assert_array_almost_equal(tsd3.index, times)
        np.testing.assert_array_almost_equal(tsd3.values, np.array([alpha[int(i)] for i in data]))

        beta=pd.Series(index=np.arange(3), data=np.random.randn(3))        
        tsd4 = tsgroup.to_tsd(beta)
        np.testing.assert_array_almost_equal(tsd4.index, times)
        np.testing.assert_array_almost_equal(tsd4.values, np.array([beta[int(i)] for i in data]))

    def test_to_tsd_runtime_errors(self, group):

        tsgroup = nap.TsGroup(group)
        
        with pytest.raises(Exception) as e_info:
            tsgroup.to_tsd(pd.Series(index=np.arange(len(tsgroup)+1), data=np.arange(len(tsgroup)+1)))
        assert str(e_info.value) == "Index are not equals"

        with pytest.raises(Exception) as e_info:
            tsgroup.to_tsd(np.arange(len(tsgroup)+1))
        assert str(e_info.value) == "Values is not the same length."

        with pytest.raises(Exception) as e_info:
            tsgroup.to_tsd("error")
        assert str(e_info.value) == "Key error not in metadata of TsGroup"

        with pytest.raises(Exception) as e_info:
            tsgroup.to_tsd(dict)
        assert str(e_info.value) == """Unknown argument format. Must be pandas.Series, numpy.ndarray or a string from one of the following values : [rate]"""

        tsgroup.set_info(alpha=np.random.rand(len(tsgroup)))

        with pytest.raises(Exception) as e_info:
            tsgroup.to_tsd(dict)
        assert str(e_info.value) == """Unknown argument format. Must be pandas.Series, numpy.ndarray or a string from one of the following values : [rate, alpha]"""

    def test_save_npz(self, group):

        group = {
            0: nap.Tsd(t=np.arange(0, 20), d = np.random.rand(20)),
            1: nap.Tsd(t=np.arange(0, 20, 0.5), d=np.random.rand(40)),
            2: nap.Tsd(t=np.arange(0, 10, 0.2), d=np.random.rand(50))
        }

        tsgroup = nap.TsGroup(group, meta = np.arange(len(group), dtype=np.int64), meta2 = np.array(['a', 'b', 'c']))

        with pytest.raises(TypeError) as e:
            tsgroup.save(dict)

        with pytest.raises(RuntimeError) as e:
            tsgroup.save('./')
        assert str(e.value) == "Invalid filename input. {} is directory.".format(Path("./").resolve())

        fake_path = './fake/path'
        with pytest.raises(RuntimeError) as e:
            tsgroup.save(fake_path+'/file.npz')
        assert str(e.value) == "Path {} does not exist.".format(Path(fake_path).resolve())

        tsgroup.save("tsgroup.npz")
        assert "tsgroup.npz" in [f.name for f in Path('.').iterdir()]

        tsgroup.save("tsgroup2")
        assert "tsgroup2.npz" in [f.name for f in Path('.').iterdir()]

        file = np.load("tsgroup.npz")

        keys = list(file.keys())    
        for k in ['t', 'd', 'start', 'end', 'index', 'meta', 'meta2']:
            assert k in keys

        times = []
        index = []
        data = []
        for n in group.keys():
            times.append(group[n].index)
            index.append(np.ones(len(group[n]))*n)
            data.append(group[n].values)
        times = np.hstack(times)
        index = np.hstack(index)
        data = np.hstack(data)
        idx = np.argsort(times)
        times = times[idx]
        data = data[idx]
        index = index[idx]

        np.testing.assert_array_almost_equal(file['start'], tsgroup.time_support.start)
        np.testing.assert_array_almost_equal(file['end'], tsgroup.time_support.end)
        np.testing.assert_array_almost_equal(file['t'], times)
        np.testing.assert_array_almost_equal(file['d'], data)
        np.testing.assert_array_almost_equal(file['index'], index)
        np.testing.assert_array_almost_equal(file['meta'], np.arange(len(group), dtype=np.int64))
        assert np.all(file['meta2']==np.array(['a', 'b', 'c']))
        file.close()

        tsgroup3 = nap.TsGroup({
                    0: nap.Ts(t=np.arange(0, 20)),
                })
        tsgroup3.save("tsgroup3")

        with np.load("tsgroup3.npz") as file:
            assert 'd' not in list(file.keys())
            np.testing.assert_array_almost_equal(file['t'], tsgroup3[0].index)

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
            ([3, 4], pytest.raises(KeyError, match= r"Key \[3, 4\] not in group index.")),
            ([2, 3], pytest.raises(KeyError, match= r"Key \[3\] not in group index."))
        ]
    )
    def test_indexing_type(self, group, keys, expectation):
        ts_group = nap.TsGroup(group)
        with expectation:
            out = ts_group[keys]

    @pytest.mark.parametrize(
        "name, expectation",
        [
            ("a", does_not_raise()),
            ("ab", does_not_raise()),
            ("__1", does_not_raise()),
            (1, pytest.raises(ValueError, match="Metadata keys must be strings")),
            (1.1, pytest.raises(ValueError, match="Metadata keys must be strings")),
            (np.arange(1), pytest.raises(ValueError, match="Metadata keys must be strings")),
            (np.arange(2), pytest.raises(ValueError, match="Metadata keys must be strings"))
        ]
    )
    def test_setitem_metadata_key(self, group, name, expectation):
        group = nap.TsGroup(group)
        with expectation:
            group[name] = np.arange(len(group))

    @pytest.mark.parametrize(
        "val, expectation",
        [
            (np.arange(3), does_not_raise()),
            (pd.Series(range(3)), does_not_raise()),
            ([1, 2, 3], does_not_raise()),
            ((1, 2, 3), does_not_raise()),
            (1, pytest.raises(TypeError, match="Metadata columns provided must be")),
            (1.1, pytest.raises(TypeError, match="Metadata columns provided must be")),
            (np.arange(1), pytest.raises(RuntimeError, match="Array is not the same length")),
            (np.arange(2), pytest.raises(RuntimeError, match="Array is not the same length"))
        ]
    )
    def test_setitem_metadata_key(self, group, val, expectation):
        group = nap.TsGroup(group)
        with expectation:
            group["a"] = val

    def test_setitem_metadata_vals(self, group):
        group = nap.TsGroup(group)
        group["a"] = np.arange(len(group))
        assert all(group._metadata["a"] == np.arange(len(group)))

    def test_setitem_metadata_twice(self, group):
        group = nap.TsGroup(group)
        group["a"] = np.arange(len(group))
        group["a"] = np.arange(len(group)) + 10
        assert all(group._metadata["a"] == np.arange(len(group)) + 10)

    def test_prevent_overwriting_existing_methods(self, ts_group):
        with pytest.raises(ValueError, match=r"Invalid metadata name\(s\)"):
            ts_group["set_info"] = np.arange(2)

    def test_setitem_metadata_twice_fail(self, group):
        group = nap.TsGroup(group)
        group["a"] = np.arange(len(group))
        raised = False
        try:
            group["a"] = np.arange(1)
        except:
            raised = True
            # check no changes have been made
            assert all(group._metadata["a"] == np.arange(len(group)))

        if not raised:
            raise ValueError

    def test_getitem_ts_object(self, ts_group):
        assert isinstance(ts_group[1], nap.Ts)

    def test_getitem_metadata(self, ts_group):
        assert np.all(ts_group.meta == np.array([10, 11]))
        assert np.all(ts_group["meta"] == np.array([10, 11]))

    @pytest.mark.parametrize(
        "bool_idx",
        [
            [True, False],
            [False, True],
            [True, True],
            np.array([True, False], dtype=bool),
            np.array([False, True], dtype=bool),
            np.array([True, True], dtype=bool),
        ]
    )
    def test_getitem_bool_indexing(self, bool_idx, ts_group):
        out = ts_group[bool_idx]
        assert isinstance(out, nap.TsGroup)
        assert len(out) == sum(bool_idx)
        idx = np.where(bool_idx)[0]
        if len(idx) == 1:
            slc = slice(idx[0], idx[0]+1)
        else:
            slc = slice(0, 2)
        assert all(out.keys()[i] == ts_group.keys()[slc][i] for i in range(len(idx)))
        for key_i in np.where(bool_idx)[0]:
            key = ts_group.keys()[key_i]
            assert np.all(out[[key]].rates == ts_group.rates[[key]])
            assert np.all(out[[key]].meta == ts_group.meta[[key]])
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
            np.array([2, 1])
        ]
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
        assert np.all(ts_group.rates == np.array([10/9, 5/9]))

    def test_getitem_key_error(self, ts_group):
        with pytest.raises(KeyError, match="Key nonexistent not in group index."):
            _ = ts_group['nonexistent']

    def test_getitem_attribute_error(self, ts_group):
        with pytest.raises(AttributeError, match="'TsGroup' object has no attribute"):
            _ = ts_group.nonexistent_metadata

    @pytest.mark.parametrize(
        "bool_idx, expectation",
        [
            (np.ones((3,), dtype=bool), pytest.raises(IndexError, match="Boolean index length must be equal")),
            (np.ones((2, 1), dtype=bool), pytest.raises(IndexError, match="Only 1-dimensional boolean indices")),
            (np.array(True), pytest.raises(IndexError, match="Only 1-dimensional boolean indices"))
        ]
    )
    def test_getitem_boolean_fail(self, ts_group, bool_idx, expectation):
        with expectation:
            out = ts_group[bool_idx]

    def test_merge_complete(self, ts_group):
        with pytest.raises(TypeError,  match="Input at positions(.*)are not TsGroup!"):
            nap.TsGroup.merge_group(ts_group, str, dict)

        ts_group2 = nap.TsGroup(
            {
                3: nap.Ts(t=np.arange(15)),
                4: nap.Ts(t=np.arange(20)),
            },
            time_support=ts_group.time_support,
            meta=np.array([12, 13])
        )
        merged = ts_group.merge(ts_group2)
        assert len(merged) == 4
        assert np.all(merged.keys() == np.array([1, 2, 3, 4]))
        assert np.all(merged.meta == np.array([10, 11, 12, 13]))
        np.testing.assert_equal(merged.metadata_columns, ts_group.metadata_columns)

    @pytest.mark.parametrize(
            'col_name, ignore_metadata, expectation',
            [
                ('meta', False, does_not_raise()),
                ('meta', True,  does_not_raise()),
                ('wrong_name', False, pytest.raises(ValueError, match="TsGroup at position 2 has different metadata columns.*")),
                ('wrong_name', True,  does_not_raise())
                ]
                )
    def test_merge_metadata(self, ts_group, col_name, ignore_metadata, expectation):
        metadata = pd.DataFrame([12, 13], index=[3, 4], columns=[col_name])
        ts_group2 = nap.TsGroup(
            {
                3: nap.Ts(t=np.arange(15)),
                4: nap.Ts(t=np.arange(20)),
            },
            time_support=ts_group.time_support,
            **metadata
            )

        with expectation:
            merged = ts_group.merge(ts_group2, ignore_metadata=ignore_metadata)
        
        if ignore_metadata:
            assert merged.metadata_columns[0] == 'rate'
        elif col_name == 'meta':
            np.testing.assert_equal(merged.metadata_columns, ts_group.metadata_columns)

    @pytest.mark.parametrize(
        'index, reset_index, expectation',
        [
            (np.array([1, 2]), False, pytest.raises(ValueError, match="TsGroup at position 2 has overlapping keys.*")),
            (np.array([1, 2]), True, does_not_raise()),
            (np.array([3, 4]), False, does_not_raise()),
            (np.array([3, 4]), True, does_not_raise())
        ]
    )
    def test_merge_index(self, ts_group, index, reset_index, expectation):
        ts_group2 = nap.TsGroup(
            dict(zip(index, [nap.Ts(t=np.arange(15)), nap.Ts(t=np.arange(20))])),
            time_support=ts_group.time_support,
            meta=np.array([12, 13])
        )

        with expectation:
            merged = ts_group.merge(ts_group2, reset_index=reset_index)
        
        if reset_index:
            assert np.all(merged.keys() == np.arange(4))
        elif np.all(index == np.array([3, 4])):
            assert np.all(merged.keys() == np.array([1, 2, 3, 4]))

    @pytest.mark.parametrize(
        'time_support, reset_time_support, expectation',
        [
            (None, False, does_not_raise()),
            (None, True,  does_not_raise()),
            (nap.IntervalSet(start=0, end=1), False,
             pytest.raises(ValueError, match="TsGroup at position 2 has different time support.*")),
            (nap.IntervalSet(start=0, end=1), True,  does_not_raise())
        ]
    )
    def test_merge_time_support(self, ts_group, time_support, reset_time_support, expectation):
        if time_support is None:
            time_support = ts_group.time_support

        ts_group2 = nap.TsGroup(
            {
                3: nap.Ts(t=np.arange(15)),
                4: nap.Ts(t=np.arange(20)),
            },
            time_support=time_support,
            meta=np.array([12, 13])
        )

        with expectation:
            merged = ts_group.merge(ts_group2, reset_time_support=reset_time_support)
        
        if reset_time_support:
            np.testing.assert_array_almost_equal(
                ts_group.time_support.as_units("s").to_numpy(),
                merged.time_support.as_units("s").to_numpy()
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
    assert np.all(unpickled_obj._metadata == ts_group._metadata)

    # Ensure that metadata columns are the same
    assert np.all(unpickled_obj._metadata.columns == ts_group._metadata.columns)

    # Ensure that the Ts are the same
    assert all([np.all(ts_group[key].t == unpickled_obj[key].t) for key in unpickled_obj.keys()])

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
    ]
)
def test_count_dtype(dtype, expectation, ts_group, ts_group_one_group):
    with expectation:
        count = ts_group.count(bin_size=0.1, dtype=dtype)
        count_one = ts_group_one_group.count(bin_size=0.1, dtype=dtype)
        if dtype:
            assert np.issubdtype(count.dtype, dtype)
            assert np.issubdtype(count_one.dtype, dtype)
