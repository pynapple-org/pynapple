#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-03-30 11:15:02
# @Last Modified by:   gviejo
# @Last Modified time: 2024-02-21 21:39:07

"""Tests for IntervalSet of `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest
import warnings
from .mock import MockArray


def test_create_iset():
    start = [0, 10, 16, 25]
    end = [5, 15, 20, 40]
    ep = nap.IntervalSet(start=start, end=end)
    assert isinstance(ep, nap.core.interval_set.IntervalSet)
    np.testing.assert_array_almost_equal(start, ep.start)
    np.testing.assert_array_almost_equal(end, ep.end)

def test_iset_properties():
    start = [0, 10, 16, 25]
    end = [5, 15, 20, 40]
    ep = nap.IntervalSet(start=start, end=end)    
    assert isinstance(ep.starts, nap.Ts)
    assert isinstance(ep.ends, nap.Ts)
    np.testing.assert_array_almost_equal(np.array(start), ep.starts.index)
    np.testing.assert_array_almost_equal(np.array(end), ep.ends.index)

    assert ep.shape == ep.values.shape
    assert ep.ndim == ep.values.ndim
    assert ep.size == ep.values.size

def test_iset_centers():
    start = np.array([0, 10, 16, 25])
    end = np.array([5, 15, 20, 40])
    ep = nap.IntervalSet(start=start, end=end)

    center_ts = ep.get_intervals_center()
    assert isinstance(center_ts, nap.Ts)
    np.testing.assert_array_almost_equal(center_ts.index, start + (end-start)/2)

    alpha = np.random.rand()
    center_ts = ep.get_intervals_center(alpha)
    assert isinstance(center_ts, nap.Ts)
    np.testing.assert_array_almost_equal(center_ts.index, start + (end-start)*alpha)

    with pytest.raises(RuntimeError):
        ep.get_intervals_center({})

def test_create_iset_from_scalars():
    ep = nap.IntervalSet(start=0, end=10)
    np.testing.assert_approx_equal(ep.start[0], 0)
    np.testing.assert_approx_equal(ep.end[0], 10)


def test_create_iset_from_df():
    df = pd.DataFrame(data=[[16, 100]], columns=["start", "end"])
    ep = nap.IntervalSet(df)
    np.testing.assert_array_almost_equal(df.start.values, ep.start)
    np.testing.assert_array_almost_equal(df.end.values, ep.end)

def test_create_iset_from_mock_array():
    start = np.array([0, 200])
    end = np.array([100, 300])

    with warnings.catch_warnings(record=True) as w:
        ep = nap.IntervalSet(MockArray(start), MockArray(end))

    assert str(w[0].message) == "Converting 'start' to numpy.array. The provided array was of type 'MockArray'."
    assert str(w[1].message) == "Converting 'end' to numpy.array. The provided array was of type 'MockArray'."
    
    np.testing.assert_array_almost_equal(ep.start, start)
    np.testing.assert_array_almost_equal(ep.end, end)

def test_create_iset_from_unknown_format():    
    with pytest.raises(RuntimeError) as e:
        nap.IntervalSet(start="abc", end=[1, 2])
    assert str(e.value) == "Unknown format for start. Accepted formats are numpy.ndarray, list, tuple or any array-like objects."
    with pytest.raises(RuntimeError) as e:
        nap.IntervalSet(start=[1,2], end="abc")
    assert str(e.value) == "Unknown format for end. Accepted formats are numpy.ndarray, list, tuple or any array-like objects."

def test_create_iset_from_s():
    start = np.array([0, 10, 16, 25])
    end = np.array([5, 15, 20, 40])
    ep = nap.IntervalSet(start=start, end=end, time_units="s")
    np.testing.assert_array_almost_equal(start, ep.start)
    np.testing.assert_array_almost_equal(end, ep.end)


def test_create_iset_from_ms():
    start = np.array([0, 10, 16, 25])
    end = np.array([5, 15, 20, 40])
    ep = nap.IntervalSet(start=start, end=end, time_units="ms")
    np.testing.assert_array_almost_equal(start * 1e-3, ep.start)
    np.testing.assert_array_almost_equal(end * 1e-3, ep.end)


def test_create_iset_from_us():
    start = np.array([0, 10, 16, 25])
    end = np.array([5, 15, 20, 40])
    ep = nap.IntervalSet(start=start, end=end, time_units="us")
    np.testing.assert_array_almost_equal(start * 1e-6, ep.start)
    np.testing.assert_array_almost_equal(end * 1e-6, ep.end)

def test_modify_iset():
    start = np.around(np.array([0, 10, 16], dtype=np.float64), 9)
    end = np.around(np.array([5, 15, 20], dtype=np.float64), 9)
    ep = nap.IntervalSet(start=start,end=end)

    with pytest.raises(RuntimeError) as e:
        ep[0,0] = 1
    assert str(e.value) == "IntervalSet is immutable. Starts and ends have been already sorted."

def test_get_iset():
    start = np.array([0, 10, 16], dtype=np.float64)
    end = np.array([5, 15, 20], dtype=np.float64)
    ep = nap.IntervalSet(start=start,end=end)

    assert isinstance(ep['start'], np.ndarray)
    assert isinstance(ep['end'], np.ndarray)
    np.testing.assert_array_almost_equal(ep['start'], start)
    np.testing.assert_array_almost_equal(ep['end'], end)

    with pytest.raises(IndexError) as e:
        ep['a']
    assert str(e.value) == "Unknown string argument. Should be 'start' or 'end'"

    # Get a new IntervalSet
    ep2 = ep[0]
    assert isinstance(ep2, nap.IntervalSet)
    np.testing.assert_array_almost_equal(ep2, np.array([[0., 5.]]))

    ep2 = ep[0:2]
    assert isinstance(ep2, nap.IntervalSet)
    np.testing.assert_array_almost_equal(ep2, ep.values[0:2])

    ep2 = ep[[0,2]]
    assert isinstance(ep2, nap.IntervalSet)
    np.testing.assert_array_almost_equal(ep2, ep.values[[0,2]])

    ep2 = ep[0:2,:]
    assert isinstance(ep2, nap.IntervalSet)
    np.testing.assert_array_almost_equal(ep2, ep.values[0:2])

    ep2 = ep[0:2,0:2]
    assert isinstance(ep2, nap.IntervalSet)
    np.testing.assert_array_almost_equal(ep2, ep.values[0:2])

    ep2 = ep[:,0]    
    np.testing.assert_array_almost_equal(ep2, ep.start)
    ep2 = ep[:,1]
    np.testing.assert_array_almost_equal(ep2, ep.end)

    with pytest.raises(IndexError) as e:
        ep[:,0,3]
    assert str(e.value) == "too many indices for IntervalSet: IntervalSet is 2-dimensional"

def test_iset_loc():
    start = np.array([0, 10, 16], dtype=np.float64)
    end = np.array([5, 15, 20], dtype=np.float64)
    ep = nap.IntervalSet(start=start,end=end)

    np.testing.assert_array_almost_equal(ep.loc[0], ep.values[0])
    assert isinstance(ep.loc[[0]], nap.IntervalSet)
    np.testing.assert_array_almost_equal(ep.loc[[0]], ep[0])
    np.testing.assert_array_almost_equal(ep.loc['start'], start)
    np.testing.assert_array_almost_equal(ep.loc['end'], end)


def test_array_ufunc():
    start = np.array([0, 10, 16], dtype=np.float64)
    end = np.array([5, 15, 20], dtype=np.float64)
    ep = nap.IntervalSet(start=start,end=end)    

    with warnings.catch_warnings(record=True) as w:
        out = np.exp(ep)
    assert str(w[0].message) == "Converting IntervalSet to numpy.array"
    np.testing.assert_array_almost_equal(out, np.exp(ep.values))

    with warnings.catch_warnings(record=True) as w:
        out = ep*2
    assert str(w[0].message) == "Converting IntervalSet to numpy.array"
    np.testing.assert_array_almost_equal(out, ep.values*2)

    with warnings.catch_warnings(record=True) as w:
        out = ep + ep
    assert str(w[0].message) == "Converting IntervalSet to numpy.array"
    np.testing.assert_array_almost_equal(out, ep.values*2)

    # test warning
    from contextlib import nullcontext as does_not_raise
    nap.config.nap_config.suppress_conversion_warnings = True
    with does_not_raise():
        np.exp(ep)

    nap.config.nap_config.suppress_conversion_warnings = False

def test_array_func():
    start = np.array([0, 10, 16], dtype=np.float64)
    end = np.array([5, 15, 20], dtype=np.float64)
    ep = nap.IntervalSet(start=start,end=end)

    with warnings.catch_warnings(record=True) as w:
        out = np.vstack((ep, ep))
    assert str(w[0].message) == "Converting IntervalSet to numpy.array"
    np.testing.assert_array_almost_equal(out, np.vstack((ep.values, ep.values)))

    with warnings.catch_warnings(record=True) as w:
        out = np.ravel(ep)
    assert str(w[0].message) == "Converting IntervalSet to numpy.array"
    np.testing.assert_array_almost_equal(out, np.ravel(ep.values))

    # test warning
    from contextlib import nullcontext as does_not_raise
    nap.config.nap_config.suppress_conversion_warnings = True
    with does_not_raise():
        out = np.ravel(ep)

    nap.config.nap_config.suppress_conversion_warnings = False

def test_timespan():
    start = [0, 10, 16, 25]
    end = [5, 15, 20, 40]
    ep = nap.IntervalSet(start=start, end=end)
    ep2 = ep.time_span()
    assert len(ep2) == 1
    np.testing.assert_array_almost_equal(np.array([0]), ep2.start)
    np.testing.assert_array_almost_equal(np.array([40]), ep2.end)


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
    df = pd.DataFrame(data=np.array([[0.0, 100.0]]), columns=["start", "end"])
    pd.testing.assert_frame_equal(df, ep.as_units("s"))
    pd.testing.assert_frame_equal(df * 1e3, ep.as_units("ms"))
    tmp = df * 1e6
    np.testing.assert_array_almost_equal(tmp.values, ep.as_units("us").values)


def test_intersect():
    ep = nap.IntervalSet(start=[0, 30], end=[10, 70])
    ep2 = nap.IntervalSet(start=40, end=100)
    ep3 = nap.IntervalSet(start=40, end=70)
    np.testing.assert_array_almost_equal(ep.intersect(ep2), ep3)
    np.testing.assert_array_almost_equal(ep2.intersect(ep), ep3)


def test_union():
    ep = nap.IntervalSet(start=[0, 30], end=[10, 70])
    ep2 = nap.IntervalSet(start=40, end=100)
    ep3 = nap.IntervalSet(start=[0, 30], end=[10, 100])
    np.testing.assert_array_almost_equal(ep.union(ep2), ep3)
    np.testing.assert_array_almost_equal(ep2.union(ep), ep3)


def test_set_diff():
    ep = nap.IntervalSet(start=[0, 30], end=[10, 70])
    ep2 = nap.IntervalSet(start=40, end=100)
    ep3 = nap.IntervalSet(start=[0, 30], end=[10, 40])
    np.testing.assert_array_almost_equal(ep.set_diff(ep2), ep3)
    ep4 = nap.IntervalSet(start=[70], end=[100])
    np.testing.assert_array_almost_equal(ep2.set_diff(ep), ep4)


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
    np.testing.assert_array_almost_equal(ep.drop_short_intervals(5.0), ep2)
    np.testing.assert_array_almost_equal(
        ep.drop_short_intervals(5.0 * 1e3, time_units="ms"), ep2
    )
    np.testing.assert_array_almost_equal(
        ep.drop_short_intervals(5.0 * 1e6, time_units="us"), ep2
    )


def test_drop_long_intervals():
    ep = nap.IntervalSet(start=np.array([0, 10, 16, 25]), end=np.array([5, 15, 20, 40]))
    ep2 = nap.IntervalSet(start=16, end=20)
    np.testing.assert_array_almost_equal(ep.drop_long_intervals(5.0), ep2)
    np.testing.assert_array_almost_equal(
        ep.drop_long_intervals(5.0 * 1e3, time_units="ms"), ep2
    )
    np.testing.assert_array_almost_equal(
        ep.drop_long_intervals(5.0 * 1e6, time_units="us"), ep2
    )


def test_merge_close_intervals():
    ep = nap.IntervalSet(start=np.array([0, 10, 16]), end=np.array([5, 15, 20]))
    ep2 = nap.IntervalSet(start=np.array([0, 10]), end=np.array([5, 20]))
    np.testing.assert_array_almost_equal(ep.merge_close_intervals(4.0), ep2)
    np.testing.assert_array_almost_equal(ep.merge_close_intervals(4.0, time_units="s"), ep2)
    np.testing.assert_array_almost_equal(
        ep.merge_close_intervals(4.0 * 1e3, time_units="ms"), ep2
    )
    np.testing.assert_array_almost_equal(
        ep.merge_close_intervals(4.0 * 1e6, time_units="us"), ep2
    )

def test_merge_close_intervals_empty():
    ep = nap.IntervalSet(start=np.array([]), end=np.array([]))
    ep = ep.merge_close_intervals(1)
    assert len(ep) == 0


def test_jitfix_iset():
    starts = np.array([0, 10, 16])
    ends = np.array([5, 15, 20])

    ep, to_warn = nap.core.utils._jitfix_iset(starts, ends)
    np.testing.assert_array_almost_equal(starts, ep[:,0])
    np.testing.assert_array_almost_equal(ends, ep[:,1])
    np.testing.assert_array_almost_equal(to_warn, np.zeros(4))

def test_jitfix_iset_error0():
    start = np.around(np.array([0, 10, 15], dtype=np.float64), 9)
    end = np.around(np.array([10, 15, 20], dtype=np.float64), 9)

    ep, to_warn = nap.core.utils._jitfix_iset(start, end)

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

    ep, to_warn = nap.core.utils._jitfix_iset(start, end)

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

    ep, to_warn = nap.core.utils._jitfix_iset(start, end)

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

    ep, to_warn = nap.core.utils._jitfix_iset(start, end)

    np.testing.assert_array_almost_equal(start[[0,2]], ep[:,0])
    np.testing.assert_array_almost_equal(end[[0,2]], ep[:,1])
    np.testing.assert_array_equal(to_warn, np.array([False, False, False, True]))

    with warnings.catch_warnings(record=True) as w:
        nap.IntervalSet(start=start, end=end)
    assert str(w[0].message) == "Some epochs have no duration"

def test_raise_warning():
    start = np.around(np.array([0, 15, 16], dtype=np.float64), 9)
    end = np.around(np.array([5, 15, 20], dtype=np.float64), 9)
    with pytest.warns(UserWarning, match=r"Some epochs have no duration"):
        nap.IntervalSet(start=start,end=end)

def test_iset_wrong_columns():
    df = pd.DataFrame(data=[[16, 100]], columns=["start", "endssss"])
    
    with pytest.raises(Exception) as e_info:
        nap.IntervalSet(df)

def test_iset_diff_length():
    with pytest.raises(Exception) as e_info:
        nap.IntervalSet(start=np.array([0, 10, 16]), end=np.array([5, 15, 20, 40]))
    assert str(e_info.value) == "Starts end ends are not of the same length"
    

def test_sort_starts():
    start = np.around(np.array([10, 0, 16], dtype=np.float64), 9)
    end = np.around(np.array([5, 15, 20], dtype=np.float64), 9)
    with pytest.warns(UserWarning, match=r"start is not sorted. Sorting it."):
        ep = nap.IntervalSet(start=start,end=end)
    np.testing.assert_array_almost_equal(np.sort(start), ep.values[:,0])

def test_sort_ends():
    start = np.around(np.array([0, 10, 16], dtype=np.float64), 9)
    end = np.around(np.array([15, 5, 20], dtype=np.float64), 9)
    with pytest.warns(UserWarning, match=r"end is not sorted. Sorting it."):
        ep = nap.IntervalSet(start=start,end=end)
    np.testing.assert_array_almost_equal(np.sort(end), ep.values[:,1])

def test_repr_():
    start = np.around(np.array([0, 10, 16], dtype=np.float64), 9)
    end = np.around(np.array([5, 15, 20], dtype=np.float64), 9)
    ep = nap.IntervalSet(start=start,end=end)
    assert isinstance(ep.__repr__(), str)

def test_str_():
    start = np.around(np.array([0, 10, 16], dtype=np.float64), 9)
    end = np.around(np.array([5, 15, 20], dtype=np.float64), 9)
    ep = nap.IntervalSet(start=start,end=end)
    assert isinstance(ep.__str__(), str)

def test_save_npz():
    import os

    start = np.around(np.array([0, 10, 16], dtype=np.float64), 9)
    end = np.around(np.array([5, 15, 20], dtype=np.float64), 9)
    ep = nap.IntervalSet(start=start,end=end)

    with pytest.raises(RuntimeError) as e:
        ep.save(dict)
    assert str(e.value) == "Invalid type; please provide filename as string"

    with pytest.raises(RuntimeError) as e:
        ep.save('./')
    assert str(e.value) == "Invalid filename input. {} is directory.".format("./")

    fake_path = './fake/path'
    with pytest.raises(RuntimeError) as e:
        ep.save(fake_path+'/file.npz')
    assert str(e.value) == "Path {} does not exist.".format(fake_path)

    ep.save("ep.npz")
    os.listdir('.')
    assert "ep.npz" in os.listdir(".")

    ep.save("ep2")
    os.listdir('.')
    assert "ep2.npz" in os.listdir(".")

    file = np.load("ep.npz")

    keys = list(file.keys())    
    assert 'start' in keys
    assert 'end' in keys

    np.testing.assert_array_almost_equal(file['start'], start)
    np.testing.assert_array_almost_equal(file['end'], end)

    # Cleaning    
    os.remove("ep.npz")
    os.remove("ep2.npz")    







