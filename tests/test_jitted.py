"""Tests of jitted core functions for `pynapple` package."""

import warnings

import numpy as np
import pandas as pd
import pytest

import pynapple as nap


def get_example_dataset(n=100):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        starts = np.sort(np.random.uniform(0, 1000, n))
        ep = nap.IntervalSet(start=starts, end=starts + np.random.uniform(1, 10, n))
        tsd = nap.Tsd(
            t=np.sort(np.random.uniform(0, 1000, n * 2)), d=np.random.rand(n * 2)
        )
        ts = nap.Ts(t=np.sort(np.random.uniform(0, 1000, n * 2)))
        tsdframe = nap.TsdFrame(
            t=np.sort(np.random.uniform(0, 1000, n * 2)), d=np.random.rand(n * 2, 3)
        )

    return (ep, ts, tsd, tsdframe)


def get_example_isets(n=100):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        starts = np.sort(np.random.uniform(0, 1000, n))

        ep1 = nap.IntervalSet(
            start=starts,
            end=starts + np.random.uniform(1, 10, n),
        )
        starts = np.sort(np.random.uniform(0, 1000, n))
        ep2 = nap.IntervalSet(
            start=starts,
            end=starts + np.random.uniform(1, 10, n),
        )

    return ep1, ep2


def restrict(ep, tsd):
    bins = ep.values.ravel()
    # Because yes there is no funtion with both bounds closed as an option
    ix = np.array(
        pd.cut(tsd.index, bins, labels=np.arange(len(bins) - 1, dtype=np.float64))
    )
    ix2 = np.array(
        pd.cut(
            tsd.index,
            bins,
            labels=np.arange(len(bins) - 1, dtype=np.float64),
            right=False,
        )
    )
    ix3 = np.vstack((ix, ix2)).T
    # ix[np.floor(ix / 2) * 2 != ix] = np.nan
    # ix = np.floor(ix/2)
    ix3[np.floor(ix3 / 2) * 2 != ix3] = np.nan
    ix3 = np.floor(ix3 / 2)
    ix3[np.isnan(ix3[:, 0]), 0] = ix3[np.isnan(ix3[:, 0]), 1]

    ix = ix3[:, 0]
    idx = ~np.isnan(ix)
    if not hasattr(tsd, "values"):
        return pd.Series(index=tsd.index[idx], dtype="object")
    else:
        return pd.Series(index=tsd.index[idx], data=tsd.values[idx])


def test_jitrestrict():
    for i in range(100):
        ep, ts, tsd, tsdframe = get_example_dataset()

        tsd2 = restrict(ep, tsd)
        ix = nap.core._jitted_functions.jitrestrict(tsd.index, ep.start, ep.end)
        tsd3 = pd.Series(index=tsd.index[ix], data=tsd.values[ix])
        np.testing.assert_array_almost_equal(tsd2.values, tsd3.values)
        np.testing.assert_array_almost_equal(tsd2.index.values, tsd3.index.values)


def test_jitrestrict_with_count():
    for i in range(100):
        ep, ts, tsd, tsdframe = get_example_dataset()

        tsd2 = restrict(ep, tsd)
        ix, count = nap.core._jitted_functions.jitrestrict_with_count(
            tsd.index, ep.start, ep.end
        )
        tsd3 = pd.Series(index=tsd.index[ix], data=tsd.values[ix])
        np.testing.assert_array_almost_equal(tsd2.values, tsd3.values)
        np.testing.assert_array_almost_equal(tsd2.index.values, tsd3.index.values)

        bins = ep.values.ravel()
        ix = np.array(
            pd.cut(tsd.index, bins, labels=np.arange(len(bins) - 1, dtype=np.float64))
        )
        ix2 = np.array(
            pd.cut(
                tsd.index,
                bins,
                labels=np.arange(len(bins) - 1, dtype=np.float64),
                right=False,
            )
        )
        ix3 = np.vstack((ix, ix2)).T
        ix3[np.floor(ix3 / 2) * 2 != ix3] = np.nan
        ix3 = np.floor(ix3 / 2)
        ix3[np.isnan(ix3[:, 0]), 0] = ix3[np.isnan(ix3[:, 0]), 1]
        ix = ix3[:, 0]
        count2 = np.array([np.sum(ix == j) for j in range(len(ep))])

        np.testing.assert_array_equal(count, count2)


def test_jitthreshold():
    for i in range(100):
        ep, ts, tsd, tsdframe = get_example_dataset()

        thr = np.random.rand()

        t, d, s, e = nap.core._jitted_functions.jitthreshold(
            tsd.index, tsd.values, ep.start, ep.end, thr
        )

        assert len(t) == np.sum(tsd.values > thr)
        assert len(d) == np.sum(tsd.values > thr)
        np.testing.assert_array_equal(d, tsd.values[tsd.values > thr])

        t, d, s, e = nap.core._jitted_functions.jitthreshold(
            tsd.index, tsd.values, ep.start, ep.end, thr, "below"
        )

        assert len(t) == np.sum(tsd.values < thr)
        assert len(d) == np.sum(tsd.values < thr)
        np.testing.assert_array_equal(d, tsd.values[tsd.values < thr])

        t, d, s, e = nap.core._jitted_functions.jitthreshold(
            tsd.index, tsd.values, ep.start, ep.end, thr, "aboveequal"
        )

        assert len(t) == np.sum(tsd.values >= thr)
        assert len(d) == np.sum(tsd.values >= thr)
        np.testing.assert_array_equal(d, tsd.values[tsd.values >= thr])

        t, d, s, e = nap.core._jitted_functions.jitthreshold(
            tsd.index, tsd.values, ep.start, ep.end, thr, "belowequal"
        )

        assert len(t) == np.sum(tsd.values <= thr)
        assert len(d) == np.sum(tsd.values <= thr)
        np.testing.assert_array_equal(d, tsd.values[tsd.values <= thr])

        # with warnings.catch_warnings(record=True) as w:
        #     new_ep = nap.IntervalSet(start=s, end=e)

        # new_tsd = restrict(new_ep, tsd)


def test_jitvalue_from():
    for i in range(100):
        ep, ts, tsd, tsdframe = get_example_dataset()

        t, d = nap.core._core_functions._value_from(
            ts.t, tsd.t, tsd.d, ep.start, ep.end
        )

        tsd3 = pd.Series(index=t, data=d)

        tsd2 = []
        for j in ep.index:
            ix = ts.restrict(ep[j]).index
            if len(ix):
                tsd2.append(
                    tsd.restrict(ep[j]).as_series().reindex(ix, method="nearest")
                )

        tsd2 = pd.concat(tsd2)

        np.testing.assert_array_almost_equal(tsd2.values, tsd3.values)
        np.testing.assert_array_almost_equal(tsd2.index.values, tsd3.index.values)


def test_jitcount():
    for i in range(10):
        ep, ts, tsd, tsdframe = get_example_dataset()

        time_array = ts.index
        starts = ep.start
        ends = ep.end
        bin_size = 1.0
        t, d = nap.core._jitted_functions.jitcount(
            time_array, starts, ends, bin_size, np.int64
        )
        tsd3 = nap.Tsd(t=t, d=d, time_support=ep)

        tsd2 = []
        for j in ep.index:
            bins = np.arange(ep[j, 0], ep[j, 1] + 1.0, 1.0)
            idx = np.digitize(ts.restrict(ep[j]).index, bins) - 1
            tmp = np.array([np.sum(idx == j) for j in range(len(bins) - 1)])
            tmp = nap.Tsd(t=bins[0:-1] + np.diff(bins) / 2, d=tmp)
            tmp = tmp.restrict(ep[j])

            tsd2.append(tmp.as_series())

        tsd2 = pd.concat(tsd2)

        np.testing.assert_array_almost_equal(tsd2.values, tsd3.values)
        np.testing.assert_array_almost_equal(tsd2.index.values, tsd3.index.values)


def test_jitbin():
    for i in range(10):
        ep, ts, tsd, tsdframe = get_example_dataset()

        time_array = tsd.index
        data_array = tsd.values
        starts = ep.start
        ends = ep.end
        bin_size = 1.0
        t, d = nap.core._jitted_functions.jitbin_array(
            time_array, data_array, starts, ends, bin_size
        )
        # tsd3 = nap.Tsd(t=t, d=d, time_support = ep)
        tsd3 = pd.Series(index=t, data=d)
        tsd3 = tsd3.fillna(0.0)

        tsd2 = []
        for j in ep.index:
            bins = np.arange(ep[j, 0], ep[j, 1] + 1.0, 1.0)
            aa = tsd.restrict(ep[j])
            tmp = np.zeros((len(bins) - 1))
            if len(aa):
                idx = np.digitize(aa.index, bins) - 1
                for k in np.unique(idx):
                    tmp[k] = np.mean(aa.values[idx == k])

            tmp = nap.Tsd(t=bins[0:-1] + np.diff(bins) / 2, d=tmp)
            tmp = tmp.restrict(ep[j])

            # pd.testing.assert_series_equal(tmp, tsd3.restrict(ep.loc[[j]]))

            tsd2.append(tmp.as_series())

        tsd2 = pd.concat(tsd2)
        # tsd2 = nap.Tsd(tsd2)
        tsd2 = tsd2.fillna(0.0)

        np.testing.assert_array_almost_equal(tsd2.values, tsd3.values)
        np.testing.assert_array_almost_equal(tsd2.index.values, tsd3.index.values)


def test_jitbin_array():
    for i in range(10):
        ep, ts, tsd, tsdframe = get_example_dataset()

        time_array = tsdframe.index
        data_array = tsdframe.values
        starts = ep.start
        ends = ep.end
        bin_size = 1.0
        t, d = nap.core._jitted_functions.jitbin_array(
            time_array, data_array, starts, ends, bin_size
        )
        tsd3 = pd.DataFrame(index=t, data=d)
        tsd3 = tsd3.fillna(0.0)
        # tsd3 = nap.TsdFrame(tsd3, time_support = ep)

        tsd2 = []
        for j in ep.index:
            bins = np.arange(ep[j, 0], ep[j, 1] + 1.0, 1.0)
            aa = tsdframe.restrict(ep[j])
            tmp = np.zeros((len(bins) - 1, tsdframe.shape[1]))
            if len(aa):
                idx = np.digitize(aa.index, bins) - 1
                for k in np.unique(idx):
                    tmp[k] = np.mean(aa.values[idx == k], 0)

            tmp = nap.TsdFrame(t=bins[0:-1] + np.diff(bins) / 2, d=tmp)
            tmp = tmp.restrict(ep[j])

            # pd.testing.assert_series_equal(tmp, tsd3.restrict(ep.loc[[j]]))

            tsd2.append(tmp.as_dataframe())

        tsd2 = pd.concat(tsd2)
        # tsd2 = nap.TsdFrame(tsd2)

        np.testing.assert_array_almost_equal(tsd3.values, tsd2.values)
        np.testing.assert_array_almost_equal(tsd3.index.values, tsd2.index.values)


def test_jitintersect():
    for i in range(10):
        ep1, ep2 = get_example_isets()

        # set label as interval index
        ep1.set_info(label1=np.arange(len(ep1)))
        ep2.set_info(label2=np.arange(len(ep2)))

        s, e, m = nap.core._jitted_functions.jitintersect(
            ep1.start, ep1.end, ep2.start, ep2.end
        )
        ep3 = nap.IntervalSet(
            s,
            e,
            metadata={
                "label1": ep1._metadata.loc[m[:, 0], "label1"]["label1"],
                "label2": ep2._metadata.loc[m[:, 1], "label2"]["label2"],
            },
        )

        i_sets = [ep1, ep2]
        n_sets = len(i_sets)

        time1 = [i_set["start"] for i_set in i_sets]
        time2 = [i_set["end"] for i_set in i_sets]
        time1.extend(time2)
        time = np.hstack(time1)

        start_end = np.hstack(
            (
                np.ones(len(time) // 2, dtype=np.int32),
                -1 * np.ones(len(time) // 2, dtype=np.int32),
            )
        )

        # stack labels to match up with start and end times
        label1 = np.hstack((ep1.label1, np.nan * np.ones(len(ep2))))
        label1 = np.hstack((label1, label1))
        label2 = np.hstack((np.nan * np.ones(len(ep1)), ep2.label2))
        label2 = np.hstack((label2, label2))

        df = pd.DataFrame(
            {"time": time, "start_end": start_end, "label1": label1, "label2": label2}
        )
        df.sort_values(by="time", inplace=True)
        df.reset_index(inplace=True, drop=True)
        # after sorting, fill NaN labels
        # will fill consecutive start/stop labels with same value, use both ffill and bfill to ensure there are no NaNs left
        # don't have to worry about values between stop and next start, as they will get ignored
        df = df.ffill().bfill()
        # cast to int to match original dtype
        df[["label1", "label2"]] = df[["label1", "label2"]].astype(int)
        df["cumsum"] = df["start_end"].cumsum()
        ix = (df["cumsum"] == n_sets).to_numpy().nonzero()[0]
        start = df["time"][ix]
        end = df["time"][ix + 1]
        # shouldn't matter if we grab label from start or end position
        label1 = df["label1"][ix].reset_index(drop=True)
        label2 = df["label2"][ix].reset_index(drop=True)

        ep4 = nap.IntervalSet(start, end, metadata={"label1": label1, "label2": label2})

        np.testing.assert_array_almost_equal(ep3, ep4)
        pd.testing.assert_frame_equal(ep3.metadata, ep4.metadata)


def test_jitunion():
    for i in range(10):
        ep1, ep2 = get_example_isets()

        s, e = nap.core._jitted_functions.jitunion(
            ep1.start, ep1.end, ep2.start, ep2.end
        )
        ep3 = nap.IntervalSet(s, e)

        i_sets = [ep1, ep2]
        time = np.hstack(
            [i_set["start"] for i_set in i_sets] + [i_set["end"] for i_set in i_sets]
        )

        start_end = np.hstack(
            (
                np.ones(len(time) // 2, dtype=np.int32),
                -1 * np.ones(len(time) // 2, dtype=np.int32),
            )
        )

        df = pd.DataFrame({"time": time, "start_end": start_end})
        df.sort_values(by="time", inplace=True)
        df.reset_index(inplace=True, drop=True)
        df["cumsum"] = df["start_end"].cumsum()
        ix_stop = (df["cumsum"] == 0).to_numpy().nonzero()[0]
        ix_start = np.hstack((0, ix_stop[:-1] + 1))
        start = df["time"][ix_start]
        stop = df["time"][ix_stop]

        ep4 = nap.IntervalSet(start, stop)

        np.testing.assert_array_almost_equal(ep3, ep4)


def test_jitdiff():
    for i in range(10):
        ep1, ep2 = get_example_isets()
        ep1.set_info(label1=np.arange(len(ep1)))

        s, e, m = nap.core._jitted_functions.jitdiff(
            ep1.start, ep1.end, ep2.start, ep2.end
        )
        ep3 = nap.IntervalSet(
            s, e, metadata={"label1": ep1._metadata.loc[m, "label1"]["label1"]}
        )

        i_sets = (ep1, ep2)
        time = np.hstack(
            [i_set["start"] for i_set in i_sets] + [i_set["end"] for i_set in i_sets]
        )
        label1 = np.hstack(
            (
                ep1.label1,
                np.nan * np.ones(len(ep2)),
                ep1.label1,
                np.nan * np.ones(len(ep2)),
            )
        )
        start_end1 = np.hstack(
            (
                np.ones(len(i_sets[0]), dtype=np.int32),
                -1 * np.ones(len(i_sets[0]), dtype=np.int32),
            )
        )
        start_end2 = np.hstack(
            (
                -1 * np.ones(len(i_sets[1]), dtype=np.int32),
                np.ones(len(i_sets[1]), dtype=np.int32),
            )
        )
        start_end = np.hstack((start_end1, start_end2))
        df = pd.DataFrame({"time": time, "start_end": start_end, "label1": label1})
        df.sort_values(by="time", inplace=True)
        df.reset_index(inplace=True, drop=True)
        df = df.ffill().bfill()
        df["label1"] = df["label1"].astype(int)
        df["cumsum"] = df["start_end"].cumsum()
        ix = (df["cumsum"] == 1).to_numpy().nonzero()[0]
        start = df["time"][ix].reset_index(drop=True)
        end = df["time"][ix + 1].reset_index(drop=True)
        label1 = df["label1"][ix].reset_index(drop=True)
        idx = start != end

        ep4 = nap.IntervalSet(
            start[idx],
            end[idx],
            metadata={"label1": label1[idx].reset_index(drop=True)},
        )

        np.testing.assert_array_almost_equal(ep3, ep4)
        pd.testing.assert_frame_equal(ep3.metadata, ep4.metadata)


def test_jitunion_isets():
    for i in range(10):
        ep1, ep2 = get_example_isets()
        ep3, ep4 = get_example_isets()

        i_sets = [ep1, ep2, ep3, ep4]

        ep6 = nap.core.ts_group._union_intervals(i_sets)

        time = np.hstack(
            [i_set["start"] for i_set in i_sets] + [i_set["end"] for i_set in i_sets]
        )

        start_end = np.hstack(
            (
                np.ones(len(time) // 2, dtype=np.int32),
                -1 * np.ones(len(time) // 2, dtype=np.int32),
            )
        )

        df = pd.DataFrame({"time": time, "start_end": start_end})
        df.sort_values(by="time", inplace=True)
        df.reset_index(inplace=True, drop=True)
        df["cumsum"] = df["start_end"].cumsum()
        ix_stop = (df["cumsum"] == 0).to_numpy().nonzero()[0]
        ix_start = np.hstack((0, ix_stop[:-1] + 1))
        start = df["time"][ix_start]
        stop = df["time"][ix_stop]

        ep5 = nap.IntervalSet(start, stop)

        np.testing.assert_array_almost_equal(ep5, ep6)


def test_jitin_interval():
    for i in range(10):
        ep, ts, tsd, tsdframe = get_example_dataset()

        inep = nap.core._jitted_functions.jitin_interval(tsd.index, ep.start, ep.end)
        inep[np.isnan(inep)] = -1

        bins = ep.values.ravel()
        ix = np.array(
            pd.cut(tsd.index, bins, labels=np.arange(len(bins) - 1, dtype=np.float64))
        )
        ix2 = np.array(
            pd.cut(
                tsd.index,
                bins,
                labels=np.arange(len(bins) - 1, dtype=np.float64),
                right=False,
            )
        )
        ix3 = np.vstack((ix, ix2)).T
        ix3[np.floor(ix3 / 2) * 2 != ix3] = np.nan
        ix3 = np.floor(ix3 / 2)
        ix3[np.isnan(ix3[:, 0]), 0] = ix3[np.isnan(ix3[:, 0]), 1]
        inep2 = ix3[:, 0]
        inep2[np.isnan(inep2)] = -1

        np.testing.assert_array_equal(inep, inep2)
