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
    # Because yes there is no function with both bounds closed as an option
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


def test_jitrestrict_empty_time_array():
    ep, ts, tsd, tsdframe = get_example_dataset()
    empty = np.array([], dtype=np.float64)
    ix = nap.core._jitted_functions.jitrestrict(empty, ep.start, ep.end)
    assert len(ix) == 0


def test_jitrestrict_empty_epochs():
    ep, ts, tsd, tsdframe = get_example_dataset()
    empty = np.array([], dtype=np.float64)
    ix = nap.core._jitted_functions.jitrestrict(tsd.index, empty, empty)
    assert len(ix) == 0


def test_jitrestrict_with_count_empty_time_array():
    ep, ts, tsd, tsdframe = get_example_dataset()
    empty = np.array([], dtype=np.float64)
    ix, count = nap.core._jitted_functions.jitrestrict_with_count(
        empty, ep.start, ep.end
    )
    assert len(ix) == 0
    assert len(count) == len(ep)
    np.testing.assert_array_equal(count, np.zeros(len(ep), dtype=np.int64))


def test_jitrestrict_with_count_empty_epochs():
    ep, ts, tsd, tsdframe = get_example_dataset()
    empty = np.array([], dtype=np.float64)
    ix, count = nap.core._jitted_functions.jitrestrict_with_count(
        tsd.index, empty, empty
    )
    assert len(ix) == 0
    assert len(count) == 0


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
            time_array,
            np.zeros(len(time_array), dtype=np.int64),
            starts,
            ends,
            bin_size,
            1,
            np.int64,
        )
        tsd3 = nap.Tsd(t=t, d=d[:, 0], time_support=ep)

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


# ── Edge-case tests ────────────────────────────────────────────────────────────


def test_jitin_interval_empty_time_array():
    starts = np.array([0.0])
    ends = np.array([10.0])
    data = nap.core._jitted_functions.jitin_interval(
        np.array([], dtype=np.float64), starts, ends
    )
    assert len(data) == 0


def test_jitin_interval_empty_epochs():
    time_array = np.array([1.0, 2.0, 3.0])
    data = nap.core._jitted_functions.jitin_interval(
        time_array, np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    )
    assert len(data) == 3
    assert np.all(np.isnan(data))


def test_jitremove_nan_empty():
    starts, ends = nap.core._jitted_functions.jitremove_nan(
        np.array([], dtype=np.float64), np.array([], dtype=np.bool_)
    )
    assert len(starts) == 0
    assert len(ends) == 0


def test_jitthreshold_single_element_above():
    time_array = np.array([5.0])
    data_array = np.array([1.0])
    starts = np.array([0.0])
    ends = np.array([10.0])
    t, d, s, e = nap.core._jitted_functions.jitthreshold(
        time_array, data_array, starts, ends, 0.5
    )
    assert len(t) == 1
    assert len(s) == 1
    assert len(e) == 1


def test_jitthreshold_single_element_below():
    time_array = np.array([5.0])
    data_array = np.array([0.0])
    starts = np.array([0.0])
    ends = np.array([10.0])
    t, d, s, e = nap.core._jitted_functions.jitthreshold(
        time_array, data_array, starts, ends, 0.5
    )
    assert len(t) == 0
    assert len(s) == 0
    assert len(e) == 0


def test_jitunion_isets_empty():
    s, e = nap.core._jitted_functions.jitunion_isets(
        np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    )
    assert len(s) == 0
    assert len(e) == 0


def _valuefrom_ranges(time_array, time_target, starts, ends, mode):
    return nap.core._jitted_functions.jitvaluefrom(
        time_array,
        time_target,
        np.searchsorted(time_array, starts, side="left"),
        np.searchsorted(time_array, ends, side="right"),
        np.searchsorted(time_target, starts, side="left"),
        np.searchsorted(time_target, ends, side="right"),
        mode,
    )


def test_jitvaluefrom_single_target_mode_before():
    # a single target in the epoch used to trigger an undefined nan_cond at mode=0
    time_array = np.array([1.0, 2.0])
    time_target = np.array([1.5])
    idx = _valuefrom_ranges(
        time_array, time_target, np.array([0.0]), np.array([3.0]), 0
    )
    assert idx[0] == -1  # target 1.5 is after timestamp 1.0 → no before-target
    assert idx[1] == 0  # target 1.5 is before timestamp 2.0 → target index 0


def test_jitvaluefrom_single_target_mode_after():
    # single target in the epoch, mode=2 (regression guard)
    time_array = np.array([1.0, 2.0])
    time_target = np.array([1.5])
    idx = _valuefrom_ranges(
        time_array, time_target, np.array([0.0]), np.array([3.0]), 2
    )
    assert idx[0] == 0  # target 1.5 is after timestamp 1.0 → target index 0
    assert idx[1] == -1  # target 1.5 is before timestamp 2.0 → no after-target


######################################
# Grouped (multi-unit) kernels
######################################


def get_grouped_dataset(seed, n_units=6, n_spikes=150, n_epochs=12):
    """Merged multi-unit array as a TsGroup stores it, plus each unit's own.

    Timestamps and epoch bounds lie on a shared 0.5 s grid, so units share
    timestamps and some timestamps fall exactly on epoch bounds. Unit 2 is empty.
    """
    rng = np.random.default_rng(seed)
    units = [np.unique(rng.integers(0, 400, n_spikes) * 0.5) for _ in range(n_units)]
    units[2] = np.array([])
    times = np.concatenate(units)
    unit_pos = np.repeat(np.arange(n_units), [len(u) for u in units])
    order = np.argsort(times, kind="stable")
    edges = np.sort(rng.choice(np.arange(410), 2 * n_epochs, replace=False)) * 0.5
    starts, ends = edges[::2], edges[1::2]
    return times[order], unit_pos[order], units, starts, ends


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("bin_size", [0.5, 1.0, 0.37, 3.0, 100.0])
@pytest.mark.parametrize("dtype", [np.int64, np.float32])
def test_jitcount_multi_unit(seed, bin_size, dtype):
    # counting merged units at once matches counting each unit alone
    jitcount = nap.core._jitted_functions.jitcount
    times, unit_pos, units, starts, ends = get_grouped_dataset(seed)
    t, cnt = jitcount(
        times, unit_pos, starts, ends, bin_size, len(units), np.dtype(dtype)
    )
    assert cnt.shape == (len(t), len(units))
    assert cnt.dtype == dtype
    for i, u in enumerate(units):
        t_ref, d_ref = jitcount(
            u,
            np.zeros(len(u), dtype=np.int64),
            starts,
            ends,
            bin_size,
            1,
            np.dtype(dtype),
        )
        np.testing.assert_array_equal(t, t_ref)
        np.testing.assert_array_equal(cnt[:, i], d_ref[:, 0])


@pytest.mark.parametrize("seed", range(5))
def test_jitcount_epochs(seed):
    times, unit_pos, units, starts, ends = get_grouped_dataset(seed)
    cnt = nap.core._jitted_functions.jitcount_epochs(
        times, unit_pos, starts, ends, len(units)
    )
    assert cnt.shape == (len(starts), len(units))
    for i, u in enumerate(units):
        _, ref = nap.core._jitted_functions.jitrestrict_with_count(u, starts, ends)
        np.testing.assert_array_equal(cnt[:, i], ref)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "align, alpha", [("start", 0.0), ("center", 0.5), ("end", 1.0)]
)
def test_jittimediff_grouped(seed, align, alpha):
    times, unit_pos, units, starts, ends = get_grouped_dataset(seed)
    new_t, new_d, offsets = nap.core._jitted_functions.jittimediff_grouped(
        times, unit_pos, starts, ends, len(units), alpha
    )
    assert offsets[0] == 0 and offsets[-1] == len(new_t) == len(new_d)
    ep = nap.IntervalSet(starts, ends)
    for i, u in enumerate(units):
        if len(u) == 0:
            # (the reference crashes on an empty Ts)
            assert offsets[i + 1] == offsets[i]
            continue
        ref = nap.Ts(u, time_support=ep).time_diff(align=align, epochs=ep)
        np.testing.assert_array_equal(new_t[offsets[i] : offsets[i + 1]], ref.t)
        np.testing.assert_array_equal(new_d[offsets[i] : offsets[i + 1]], ref.values)


@pytest.mark.parametrize("seed", range(5))
def test_jitgroup_by_unit(seed):
    _, unit_pos, units, _, _ = get_grouped_dataset(seed)
    order, offsets = nap.core._jitted_functions.jitgroup_by_unit(unit_pos, len(units))
    np.testing.assert_array_equal(order, np.argsort(unit_pos, kind="stable"))
    np.testing.assert_array_equal(np.diff(offsets), [len(u) for u in units])


@pytest.mark.parametrize("mode", ["closest", "before", "after"])
def test_value_from_extra_arrays(mode):
    from pynapple.core._core_functions import _value_from

    ep, ts, tsd, _ = get_example_dataset()
    labels = np.arange(len(ts.t))
    t, d, sliced, none = _value_from(
        ts.t, tsd.t, tsd.d, ep.start, ep.end, labels, None, mode=mode
    )
    t_ref, d_ref = _value_from(ts.t, tsd.t, tsd.d, ep.start, ep.end, mode=mode)
    np.testing.assert_array_equal(t, t_ref)
    np.testing.assert_array_equal(d, d_ref)
    # the extra array is sliced the same way as the timestamps
    np.testing.assert_array_equal(ts.t[sliced], t)
    assert none is None


@pytest.mark.parametrize("seed", range(5))
def test_jitcount_clusters(seed):
    from pynapple.core._core_functions import _count_clusters

    _, unit_pos, units, _, _ = get_grouped_dataset(seed)
    # non-contiguous keys with gaps and an offset: unit i has key 3 * i + 7
    index = 3 * np.arange(len(units)) + 7
    clusters = index[unit_pos]
    counts = nap.core._jitted_functions.jitcount_clusters(
        clusters, index[0], index[-1] - index[0] + 1
    )
    np.testing.assert_array_equal(counts[index - index[0]], [len(u) for u in units])
    np.testing.assert_array_equal(
        _count_clusters(clusters, index), [len(u) for u in units]
    )


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("fragmented", [True, False])
@pytest.mark.parametrize("with_data", [True, False])
def test_restrict_arrays(seed, fragmented, with_data):
    from pynapple.core._core_functions import _restrict_arrays

    times, unit_pos, units, starts, ends = get_grouped_dataset(seed)
    if not fragmented:  # few intervals: searchsorted path rather than merge scan
        starts, ends = starts[:1], ends[:1]
    data = np.stack([times, -times], axis=1) if with_data else None
    t, u, d = _restrict_arrays(times, starts, ends, unit_pos, data)
    for i, ui in enumerate(units):
        ref = ui[nap.core._jitted_functions.jitrestrict(ui, starts, ends)]
        np.testing.assert_array_equal(t[u == i], ref)
    if with_data:
        np.testing.assert_array_equal(d, np.stack([t, -t], axis=1))
    else:
        assert d is None


def test_grouped_kernels_empty():
    jf = nap.core._jitted_functions
    empty_t = np.array([], dtype=np.float64)
    empty_u = np.array([], dtype=np.int64)
    starts, ends = np.array([0.0, 10.0]), np.array([5.0, 15.0])
    no_ep = np.array([], dtype=np.float64)

    t, cnt = jf.jitcount(empty_t, empty_u, starts, ends, 1.0, 3, np.int64)
    assert cnt.shape == (len(t), 3) and cnt.sum() == 0
    t, cnt = jf.jitcount(np.array([1.0]), np.array([0]), no_ep, no_ep, 1.0, 3, np.int64)
    assert len(t) == 0 and cnt.shape == (0, 3)

    cnt = jf.jitcount_epochs(empty_t, empty_u, starts, ends, 3)
    np.testing.assert_array_equal(cnt, np.zeros((2, 3)))

    new_t, new_d, offsets = jf.jittimediff_grouped(
        empty_t, empty_u, starts, ends, 3, 0.5
    )
    assert len(new_t) == len(new_d) == 0
    np.testing.assert_array_equal(offsets, np.zeros(4))

    order, offsets = jf.jitgroup_by_unit(empty_u, 3)
    assert len(order) == 0
    np.testing.assert_array_equal(offsets, np.zeros(4))

    np.testing.assert_array_equal(jf.jitcount_clusters(empty_u, 0, 3), np.zeros(3))
    from pynapple.core._core_functions import _count_clusters

    assert len(_count_clusters(empty_u, empty_u)) == 0
