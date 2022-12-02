#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-03-30 11:15:02
# @Last Modified by:   gviejo
# @Last Modified time: 2022-11-28 22:39:56

"""Tests for IntervalSet of `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest
import warnings


def test_create_iset():
    start = [0, 10, 16, 25]
    end = [5, 15, 20, 40]
    ep = nap.IntervalSet(start=start, end=end)
    assert isinstance(ep, pd.DataFrame)
    np.testing.assert_array_almost_equal(start, ep.start.values)
    np.testing.assert_array_almost_equal(end, ep.end.values)


def test_create_iset_from_scalars():
    ep = nap.IntervalSet(start=0, end=10)
    np.testing.assert_approx_equal(ep.start[0], 0)
    np.testing.assert_approx_equal(ep.end[0], 10)


def test_create_iset_from_df():
    df = pd.DataFrame(data=[[16, 100]], columns=["start", "end"])
    ep = nap.IntervalSet(df)
    np.testing.assert_array_almost_equal(df.start.values, ep.start.values)
    np.testing.assert_array_almost_equal(df.end.values, ep.end.values)


def test_create_iset_from_s():
    start = np.array([0, 10, 16, 25])
    end = np.array([5, 15, 20, 40])
    ep = nap.IntervalSet(start=start, end=end, time_units="s")
    np.testing.assert_array_almost_equal(start, ep.start.values)
    np.testing.assert_array_almost_equal(end, ep.end.values)


def test_create_iset_from_ms():
    start = np.array([0, 10, 16, 25])
    end = np.array([5, 15, 20, 40])
    ep = nap.IntervalSet(start=start, end=end, time_units="ms")
    np.testing.assert_array_almost_equal(start * 1e-3, ep.start.values)
    np.testing.assert_array_almost_equal(end * 1e-3, ep.end.values)


def test_create_iset_from_us():
    start = np.array([0, 10, 16, 25])
    end = np.array([5, 15, 20, 40])
    ep = nap.IntervalSet(start=start, end=end, time_units="us")
    np.testing.assert_array_almost_equal(start * 1e-6, ep.start.values)
    np.testing.assert_array_almost_equal(end * 1e-6, ep.end.values)


def test_timespan():
    start = [0, 10, 16, 25]
    end = [5, 15, 20, 40]
    ep = nap.IntervalSet(start=start, end=end)
    ep2 = ep.time_span()
    assert len(ep2) == 1
    np.testing.assert_array_almost_equal(np.array([0]), ep2.start.values)
    np.testing.assert_array_almost_equal(np.array([40]), ep2.end.values)


def test_tot_length():
    start = np.array([0, 10, 16, 25])
    end = np.array([5, 15, 20, 40])
    ep = nap.IntervalSet(start=start, end=end)
    tot_l = np.sum(end - start)
    np.testing.assert_approx_equal(tot_l, ep.tot_length())
    np.testing.assert_approx_equal(tot_l * 1e3, ep.tot_length("ms"))
    np.testing.assert_approx_equal(tot_l * 1e6, ep.tot_length("us"))


def test_as_units():
    ep = nap.IntervalSet(start=0, end=100)
    pd.testing.assert_frame_equal(ep, ep.as_units("s"))
    pd.testing.assert_frame_equal(ep * 1e3, ep.as_units("ms"))
    tmp = ep * 1e6
    np.testing.assert_array_almost_equal(tmp.values, ep.as_units("us").values)


def test_intersect():
    ep = nap.IntervalSet(start=[0, 30], end=[10, 70])
    ep2 = nap.IntervalSet(start=40, end=100)
    ep3 = nap.IntervalSet(start=40, end=70)
    pd.testing.assert_frame_equal(ep.intersect(ep2), ep3)
    pd.testing.assert_frame_equal(ep2.intersect(ep), ep3)


def test_union():
    ep = nap.IntervalSet(start=[0, 30], end=[10, 70])
    ep2 = nap.IntervalSet(start=40, end=100)
    ep3 = nap.IntervalSet(start=[0, 30], end=[10, 100])
    pd.testing.assert_frame_equal(ep.union(ep2), ep3)
    pd.testing.assert_frame_equal(ep2.union(ep), ep3)


def test_set_diff():
    ep = nap.IntervalSet(start=[0, 30], end=[10, 70])
    ep2 = nap.IntervalSet(start=40, end=100)
    ep3 = nap.IntervalSet(start=[0, 30], end=[10, 40])
    pd.testing.assert_frame_equal(ep.set_diff(ep2), ep3)
    ep4 = nap.IntervalSet(start=[70], end=[100])
    pd.testing.assert_frame_equal(ep2.set_diff(ep), ep4)


def test_in_interval():
    ep = nap.IntervalSet(start=[0, 30], end=[10, 70])
    tsd = nap.Ts(t=np.array([5, 20, 50, 100]))    
    tmp = ep.in_interval(tsd)
    np.testing.assert_array_almost_equal(
        tmp, np.array([0.0, np.nan, 1.0, np.nan])
    )

def test_drop_short_intervals():
    ep = nap.IntervalSet(start=np.array([0, 10, 16, 25]), end=np.array([5, 15, 20, 40]))
    ep2 = nap.IntervalSet(start=25, end=40)
    pd.testing.assert_frame_equal(ep.drop_short_intervals(5.0), ep2)
    pd.testing.assert_frame_equal(
        ep.drop_short_intervals(5.0 * 1e3, time_units="ms"), ep2
    )
    pd.testing.assert_frame_equal(
        ep.drop_short_intervals(5.0 * 1e6, time_units="us"), ep2
    )


def test_drop_long_intervals():
    ep = nap.IntervalSet(start=np.array([0, 10, 16, 25]), end=np.array([5, 15, 20, 40]))
    ep2 = nap.IntervalSet(start=16, end=20)
    pd.testing.assert_frame_equal(ep.drop_long_intervals(5.0), ep2)
    pd.testing.assert_frame_equal(
        ep.drop_long_intervals(5.0 * 1e3, time_units="ms"), ep2
    )
    pd.testing.assert_frame_equal(
        ep.drop_long_intervals(5.0 * 1e6, time_units="us"), ep2
    )


def test_merge_close_intervals():
    ep = nap.IntervalSet(start=np.array([0, 10, 16]), end=np.array([5, 15, 20]))
    ep2 = nap.IntervalSet(start=np.array([0, 10]), end=np.array([5, 20]))
    pd.testing.assert_frame_equal(ep.merge_close_intervals(4.0), ep2)
    pd.testing.assert_frame_equal(ep.merge_close_intervals(4.0, time_units="s"), ep2)
    pd.testing.assert_frame_equal(
        ep.merge_close_intervals(4.0 * 1e3, time_units="ms"), ep2
    )
    pd.testing.assert_frame_equal(
        ep.merge_close_intervals(4.0 * 1e6, time_units="us"), ep2
    )

def test_merge_close_intervals_empty():
    ep = nap.IntervalSet(start=np.array([]), end=np.array([]))
    ep = ep.merge_close_intervals(1)
    assert len(ep) == 0


def test_jitfix_iset():
    starts = np.array([0, 10, 16])
    ends = np.array([5, 15, 20])

    ep, to_warn = nap.core.interval_set.jitfix_iset(starts, ends)
    np.testing.assert_array_almost_equal(starts, ep[:,0])
    np.testing.assert_array_almost_equal(ends, ep[:,1])
    np.testing.assert_array_almost_equal(to_warn, np.zeros(4))

def test_jitfix_iset_error0():
    start = np.around(np.array([0, 10, 15], dtype=np.float64), 9)
    end = np.around(np.array([10, 15, 20], dtype=np.float64), 9)

    ep, to_warn = nap.core.interval_set.jitfix_iset(start, end)

    end[1:] -= 1e-6

    np.testing.assert_array_almost_equal(start, ep[:,0])
    np.testing.assert_array_almost_equal(end, ep[:,1])
    np.testing.assert_array_equal(to_warn, np.array([True, False, False, False]))

    with warnings.catch_warnings(record=True) as w:
        nap.IntervalSet(start=start, end=end)
    assert str(w[0].message) == "Some starts and ends are equal. Removing 1 microsecond!"

def test_jitfix_iset_error1():
    """
    Some ends precede the relative start. Dropping them!
    """
    start = np.around(np.array([0, 15, 16], dtype=np.float64), 9)
    end = np.around(np.array([5, 10, 20], dtype=np.float64), 9)

    ep, to_warn = nap.core.interval_set.jitfix_iset(start, end)

    np.testing.assert_array_almost_equal(start[[0,2]], ep[:,0])
    np.testing.assert_array_almost_equal(end[[0,2]], ep[:,1])
    np.testing.assert_array_equal(to_warn, np.array([False, True, False, False]))

    with warnings.catch_warnings(record=True) as w:
        nap.IntervalSet(start=start, end=end)
    assert str(w[0].message) == "Some ends precede the relative start. Dropping them!"

def test_jitfix_iset_error2():
    """
    Some starts precede the previous end. Joining them!
    """
    start = np.around(np.array([0, 10, 16], dtype=np.float64), 9)
    end = np.around(np.array([11, 15, 20], dtype=np.float64), 9)

    ep, to_warn = nap.core.interval_set.jitfix_iset(start, end)

    np.testing.assert_array_almost_equal(start[[0,2]], ep[:,0])
    np.testing.assert_array_almost_equal(end[[1,2]], ep[:,1])
    np.testing.assert_array_equal(to_warn, np.array([False, False, True, False]))

    with warnings.catch_warnings(record=True) as w:
        nap.IntervalSet(start=start, end=end)
    assert str(w[0].message) == "Some starts precede the previous end. Joining them!"

def test_jitfix_iset_error3():
    """
    Some epochs have no duration
    """
    start = np.around(np.array([0, 15, 16], dtype=np.float64), 9)
    end = np.around(np.array([5, 15, 20], dtype=np.float64), 9)

    ep, to_warn = nap.core.interval_set.jitfix_iset(start, end)

    np.testing.assert_array_almost_equal(start[[0,2]], ep[:,0])
    np.testing.assert_array_almost_equal(end[[0,2]], ep[:,1])
    np.testing.assert_array_equal(to_warn, np.array([False, False, False, True]))

    with warnings.catch_warnings(record=True) as w:
        nap.IntervalSet(start=start, end=end)
    assert str(w[0].message) == "Some epochs have no duration"

@pytest.mark.filterwarnings("ignore")
def test_raise_warning():
    start = np.around(np.array([0, 15, 16], dtype=np.float64), 9)
    end = np.around(np.array([5, 15, 20], dtype=np.float64), 9)
    ep = nap.IntervalSet(start=start,end=end)

def test_iset_wrong_columns():
    df = pd.DataFrame(data=[[16, 100]], columns=["start", "endssss"])
    
    with pytest.raises(Exception) as e_info:
        nap.IntervalSet(df)
    assert str(e_info.value) == "wrong columns name"

def test_iset_diff_length():
    with pytest.raises(Exception) as e_info:
        nap.IntervalSet(start=np.array([0, 10, 16]), end=np.array([5, 15, 20, 40]))
    assert str(e_info.value) == "Starts end ends are not of the same length"
    
@pytest.mark.filterwarnings("ignore")
def test_raise_warning():
    start = np.around(np.array([0, 15, 16], dtype=np.float64), 9)
    end = np.around(np.array([5, 15, 20], dtype=np.float64), 9)
    ep = nap.IntervalSet(start=start,end=end)

@pytest.mark.filterwarnings("ignore")
def test_sort_starts():
    start = np.around(np.array([10, 0, 16], dtype=np.float64), 9)
    end = np.around(np.array([5, 15, 20], dtype=np.float64), 9)
    ep = nap.IntervalSet(start=start,end=end)
    np.testing.assert_array_almost_equal(np.sort(start), ep.values[:,0])

@pytest.mark.filterwarnings("ignore")
def test_sort_ends():
    start = np.around(np.array([0, 10, 16], dtype=np.float64), 9)
    end = np.around(np.array([15, 5, 20], dtype=np.float64), 9)
    ep = nap.IntervalSet(start=start,end=end)
    np.testing.assert_array_almost_equal(np.sort(end), ep.values[:,1])

def test_repr_():
    start = np.around(np.array([0, 10, 16], dtype=np.float64), 9)
    end = np.around(np.array([5, 15, 20], dtype=np.float64), 9)
    ep = nap.IntervalSet(start=start,end=end)

    assert pd.DataFrame(ep).__repr__() == ep.__repr__()

def test_str_():
    start = np.around(np.array([0, 10, 16], dtype=np.float64), 9)
    end = np.around(np.array([5, 15, 20], dtype=np.float64), 9)
    ep = nap.IntervalSet(start=start,end=end)

    assert pd.DataFrame(ep).__str__() == ep.__str__()





