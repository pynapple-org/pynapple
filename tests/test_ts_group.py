# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-03-30 11:14:41
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-09-18 17:05:05

"""Tests of ts group for `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest
from collections import UserDict
import warnings


@pytest.mark.parametrize(
    "group",
    [
        {
            0: nap.Ts(t=np.arange(0, 200)),
            1: nap.Ts(t=np.arange(0, 200, 0.5), time_units="s"),
            2: nap.Ts(t=np.arange(0, 300, 0.2), time_units="s"),
        }
    ],
)
class Test_Ts_Group_1:
    def test_create_ts_group(self, group):
        tsgroup = nap.TsGroup(group)
        assert isinstance(tsgroup, UserDict)
        assert len(tsgroup) == 3

    def test_create_ts_group_from_array(self, group):
        with warnings.catch_warnings(record=True) as w:
            nap.TsGroup({
                0: np.arange(0, 200),
                1: np.arange(0, 200, 0.5),
                2: np.arange(0, 300, 0.2),
                })
        assert str(w[0].message) == "Elements should not be passed as numpy array. Default time units is seconds when creating the Ts object."

    def test_create_ts_group_with_time_support(self, group):
        ep = nap.IntervalSet(start=0, end=100)
        tsgroup = nap.TsGroup(group, time_support=ep)
        pd.testing.assert_frame_equal(tsgroup.time_support, ep)
        first = [tsgroup[i].index[0] for i in tsgroup]
        last = [tsgroup[i].index[-1] for i in tsgroup]
        assert np.all(first >= ep.loc[0, "start"])
        assert np.all(last <= ep.loc[0, "end"])

    def test_create_ts_group_with_empty_time_support(self, group):
        with pytest.raises(RuntimeError) as e_info:
            tmp = nap.TsGroup({
                0: nap.Ts(t=np.array([])),
                1: nap.Ts(t=np.array([])),
                2: nap.Ts(t=np.array([])),
                })
        assert str(e_info.value) == "Union of time supports is empty. Consider passing a time support as argument."

    def test_create_ts_group_with_bypass_check(self, group):
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
        pd.testing.assert_series_equal(tsgroup._metadata["sr"], sr_info)
        np.testing.assert_array_almost_equal(tsgroup._metadata["ar"].values, ar_info)

    def test_add_metainfo(self, group):
        tsgroup = nap.TsGroup(group)
        df_info = pd.DataFrame(index=[0, 1, 2], data=[0, 0, 0], columns=["df"])
        sr_info = pd.Series(index=[0, 1, 2], data=[1, 1, 1], name="sr")
        ar_info = np.ones(3) * 3
        tsgroup.set_info(df_info, sr=sr_info, ar=ar_info)
        assert tsgroup._metadata.shape == (3, 4)
        pd.testing.assert_series_equal(tsgroup._metadata["df"], df_info["df"])
        pd.testing.assert_series_equal(tsgroup._metadata["sr"], sr_info)
        np.testing.assert_array_almost_equal(tsgroup._metadata["ar"].values, ar_info)

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
        assert str(e_info.value) == "Columns needs to be labelled for metadata"

        tsgroup = nap.TsGroup(group)
        ar_info = np.ones(3) * 3

        with pytest.raises(RuntimeError) as e_info:
            tsgroup.set_info(ar_info)
        assert str(e_info.value) == "Columns needs to be labelled for metadata"


    def test_add_metainfo_test_runtime_errors(self, group):
        tsgroup = nap.TsGroup(group)
        sr_info = pd.Series(index=[1, 2, 3], data=[1, 1, 1], name="sr")
        with pytest.raises(Exception) as e_info:
            tsgroup.set_info(sr=sr_info)
        assert str(e_info.value) == "Index are not equals"
        df_info = pd.DataFrame(index=[1, 2, 3], data=[1, 1, 1], columns=["df"])
        with pytest.raises(Exception) as e_info:
            tsgroup.set_info(df_info)
        assert str(e_info.value) == "Index are not equals"
        sr_info = pd.Series(index=[1, 2, 3], data=[1, 1, 1], name="sr")
        with pytest.raises(Exception) as e_info:
            tsgroup.set_info(sr_info)
        assert str(e_info.value) == "Columns needs to be labelled for metadata"
        ar_info = np.ones(4)
        with pytest.raises(Exception) as e_info:
            tsgroup.set_info(ar=ar_info)
        assert str(e_info.value) == "Array is not the same length."

    def test_non_mutability(self, group):
        tsgroup = nap.TsGroup(group)
        with pytest.raises(Exception) as e_info:
            tsgroup[4] = nap.Ts(t=np.arange(4))
        assert str(e_info.value) == "TsGroup object is not mutable."
        tsgroup = nap.TsGroup(group)
        with pytest.raises(Exception) as e_info:
            tsgroup[3] = nap.Ts(t=np.arange(4))
        assert str(e_info.value) == "TsGroup object is not mutable."

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
        assert np.all(first >= ep.loc[0, "start"])
        assert np.all(last <= ep.loc[0, "end"])

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

        cols = tsgroup._metadata.columns.drop("rate")
        headers = ["Index", "rate"] + [c for c in cols]
        lines = []

        for i in tsgroup.index:
            lines.append(
                [str(i), "%.2f" % tsgroup._metadata.loc[i, "rate"]]
                + [tsgroup._metadata.loc[i, c] for c in cols]
            )
        assert tabulate(lines, headers=headers) == tsgroup.__repr__()

    def test_str_(self, group):
        from tabulate import tabulate

        tsgroup = nap.TsGroup(group)

        cols = tsgroup._metadata.columns.drop("rate")
        headers = ["Index", "rate"] + [c for c in cols]
        lines = []

        for i in tsgroup.index:
            lines.append(
                [str(i), "%.2f" % tsgroup._metadata.loc[i, "rate"]]
                + [tsgroup._metadata.loc[i, c] for c in cols]
            )
        assert tabulate(lines, headers=headers) == tsgroup.__str__()
        
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

        import os

        group = {
            0: nap.Tsd(t=np.arange(0, 20), d = np.random.rand(20)),
            1: nap.Tsd(t=np.arange(0, 20, 0.5), d=np.random.rand(40)),
            2: nap.Tsd(t=np.arange(0, 10, 0.2), d=np.random.rand(50))
        }

        tsgroup = nap.TsGroup(group, meta = np.arange(len(group), dtype=np.int64), meta2 = np.array(['a', 'b', 'c']))

        with pytest.raises(RuntimeError) as e:
            tsgroup.save(dict)
        assert str(e.value) == "Invalid type; please provide filename as string"

        with pytest.raises(RuntimeError) as e:
            tsgroup.save('./')
        assert str(e.value) == "Invalid filename input. {} is directory.".format("./")

        fake_path = './fake/path'
        with pytest.raises(RuntimeError) as e:
            tsgroup.save(fake_path+'/file.npz')
        assert str(e.value) == "Path {} does not exist.".format(fake_path)

        tsgroup.save("tsgroup.npz")
        os.listdir('.')
        assert "tsgroup.npz" in os.listdir(".")

        tsgroup.save("tsgroup2")
        os.listdir('.')
        assert "tsgroup2.npz" in os.listdir(".")

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

        np.testing.assert_array_almost_equal(file['start'], tsgroup.time_support.start.values)
        np.testing.assert_array_almost_equal(file['end'], tsgroup.time_support.end.values)
        np.testing.assert_array_almost_equal(file['t'], times)
        np.testing.assert_array_almost_equal(file['d'], data)
        np.testing.assert_array_almost_equal(file['index'], index)
        np.testing.assert_array_almost_equal(file['meta'], np.arange(len(group), dtype=np.int64))
        assert np.all(file['meta2']==np.array(['a', 'b', 'c']))

        tsgroup3 = nap.TsGroup({
                    0: nap.Ts(t=np.arange(0, 20)),
                })
        tsgroup3.save("tsgroup3")
        file = np.load("tsgroup3.npz")

        assert 'd' not in list(file.keys())
        np.testing.assert_array_almost_equal(file['t'], tsgroup3[0].index)

        os.remove("tsgroup.npz")
        os.remove("tsgroup2.npz")
        os.remove("tsgroup3.npz")
