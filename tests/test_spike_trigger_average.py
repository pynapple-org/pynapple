#!/usr/bin/env python

"""Tests of spike trigger average for `pynapple` package."""

import numpy as np
import pandas as pd
import pytest

import pynapple as nap

# from matplotlib.pyplot import *


def test_compute_spike_trigger_average_tsd():
    ep = nap.IntervalSet(0, 100)
    d = np.zeros(int(101 / 0.01))
    t1 = np.arange(1, 100)
    x = np.arange(100, 10000, 100)
    d[x] = 1.0
    feature = nap.Tsd(t=np.arange(0, 101, 0.01), d=d, time_support=ep)
    spikes = nap.TsGroup(
        {0: nap.Ts(t1), 1: nap.Ts(t1 - 0.1), 2: nap.Ts(t1 + 0.2)}, time_support=ep
    )

    sta = nap.compute_event_trigger_average(spikes, feature, 0.2, (0.6, 0.6), ep)

    output = np.zeros((7, 3))
    output[3, 0] = 0.05
    output[4, 1] = 0.05
    output[2, 2] = 0.05

    assert isinstance(sta, nap.TsdFrame)
    assert sta.shape == output.shape
    np.testing.assert_array_almost_equal(sta.values, output)

    sta = nap.compute_event_trigger_average(spikes, feature, 0.2, (0.6, 0.6))
    assert isinstance(sta, nap.TsdFrame)
    assert sta.shape == output.shape
    np.testing.assert_array_almost_equal(sta.values, output)


def test_compute_spike_trigger_average_tsdframe():
    ep = nap.IntervalSet(0, 100)
    d = np.zeros((int(101 / 0.01), 1))
    x = np.arange(100, 10000, 100)
    d[x] = 1.0
    feature = nap.TsdFrame(t=np.arange(0, 101, 0.01), d=d, time_support=ep)
    t1 = np.arange(1, 100)
    spikes = nap.TsGroup(
        {0: nap.Ts(t1), 1: nap.Ts(t1 - 0.1), 2: nap.Ts(t1 + 0.2)}, time_support=ep
    )

    sta = nap.compute_event_trigger_average(spikes, feature, 0.2, (0.6, 0.6), ep)

    output = np.zeros((7, 3))
    output[3, 0] = 0.05
    output[4, 1] = 0.05
    output[2, 2] = 0.05

    assert isinstance(sta, nap.TsdTensor)
    assert sta.shape == (*output.shape, 1)
    np.testing.assert_array_almost_equal(sta.values, np.expand_dims(output, 2))


def test_compute_spike_trigger_average_tsdtensor():
    ep = nap.IntervalSet(0, 100)
    d = np.zeros((int(101 / 0.01), 1, 1))
    x = np.arange(100, 10000, 100)
    d[x] = 1.0
    feature = nap.TsdTensor(t=np.arange(0, 101, 0.01), d=d, time_support=ep)
    t1 = np.arange(1, 100)
    spikes = nap.TsGroup(
        {0: nap.Ts(t1), 1: nap.Ts(t1 - 0.1), 2: nap.Ts(t1 + 0.2)}, time_support=ep
    )

    sta = nap.compute_event_trigger_average(spikes, feature, 0.2, (0.6, 0.6), ep)

    output = np.zeros((7, 3, 1, 1))
    output[3, 0] = 0.05
    output[4, 1] = 0.05
    output[2, 2] = 0.05

    assert isinstance(sta, nap.TsdTensor)
    assert sta.shape == output.shape
    np.testing.assert_array_almost_equal(sta.values, output)


def test_compute_spike_trigger_average_random_feature():
    ep = nap.IntervalSet(0, 100)
    feature = nap.Tsd(
        t=np.arange(0, 100, 0.001), d=np.random.randn(100000), time_support=ep
    )
    t1 = np.sort(np.random.uniform(0, 100, 1000))
    spikes = nap.TsGroup({0: nap.Ts(t1)}, time_support=ep)

    group = spikes
    binsize = 0.1
    windowsize = (1.0, 1.0)

    sta = nap.compute_event_trigger_average(spikes, feature, binsize, windowsize, ep)

    start, end = windowsize
    idx1 = -np.arange(0, start + binsize, binsize)[::-1][:-1]
    idx2 = np.arange(0, end + binsize, binsize)[1:]
    time_idx = np.hstack((idx1, np.zeros(1), idx2))
    count = group.count(binsize, ep)
    tmp = feature.bin_average(binsize, ep)
    from scipy.linalg import hankel

    # Build the Hankel matrix
    n_p = len(idx1)
    n_f = len(idx2)
    pad_tmp = np.pad(tmp.values, (n_p, n_f))
    offset_tmp = hankel(pad_tmp, pad_tmp[-(n_p + n_f + 1) :])[0 : len(tmp)]
    sta2 = np.dot(offset_tmp.T, count.values)
    sta2 = sta2 / np.sum(count.values, 0)

    np.testing.assert_array_almost_equal(sta.values, sta2)


def test_compute_spike_trigger_average_add_nan():
    ep = nap.IntervalSet(0, 110)
    d = np.zeros(int(110 / 0.01))
    x = np.arange(100, 10000, 100)
    d[x] = 1.0
    d[-1001:] = np.nan
    feature = nap.Tsd(t=np.arange(0, 110, 0.01), d=d, time_support=ep)
    t1 = np.arange(1, 100)

    spikes = nap.TsGroup(
        {0: nap.Ts(t1), 1: nap.Ts(t1 - 0.1), 2: nap.Ts(t1 + 0.2)}, time_support=ep
    )

    sta = nap.compute_event_trigger_average(spikes, feature, 0.2, (0.6, 0.6), ep)

    output = np.zeros((7, 3))
    output[3, 0] = 0.05
    output[4, 1] = 0.05
    output[2, 2] = 0.05

    assert isinstance(sta, nap.TsdFrame)
    assert sta.shape == output.shape
    np.testing.assert_array_almost_equal(sta.values, output)


def test_compute_spike_trigger_average_raise_error():
    ep = nap.IntervalSet(0, 101)
    d = np.zeros(int(101 / 0.01))
    x = np.arange(100, 10000, 100) + 1
    d[x] = 1.0
    feature = nap.Tsd(t=np.arange(0, 101, 0.01), d=d, time_support=ep)
    t1 = np.arange(1, 101) + 0.01
    spikes = nap.TsGroup(
        {0: nap.Ts(t1), 1: nap.Ts(t1 - 0.1), 2: nap.Ts(t1 + 0.2)}, time_support=ep
    )

    with pytest.raises(TypeError) as e_info:
        nap.compute_event_trigger_average(feature, feature, 0.1, (0.5, 0.5), ep)
    assert (
        str(e_info.value)
        == "Invalid type. Parameter group must be of type ['TsGroup']."
    )

    with pytest.raises(TypeError) as e_info:
        nap.compute_event_trigger_average(spikes, np.array(10), 0.1, (0.5, 0.5), ep)
    assert (
        str(e_info.value)
        == "Invalid type. Parameter feature must be of type ['Tsd', 'TsdFrame', 'TsdTensor']."
    )

    with pytest.raises(TypeError) as e_info:
        nap.compute_event_trigger_average(spikes, feature, "0.1", (0.5, 0.5), ep)
    assert (
        str(e_info.value)
        == "Invalid type. Parameter binsize must be of type ['Number']."
    )

    with pytest.raises(TypeError) as e_info:
        nap.compute_event_trigger_average(
            spikes, feature, 0.1, (0.5, 0.5), ep, time_unit=1
        )
    assert (
        str(e_info.value)
        == "Invalid type. Parameter time_unit must be of type ['str']."
    )

    with pytest.raises(RuntimeError) as e_info:
        nap.compute_event_trigger_average(
            spikes, feature, 0.1, (0.5, 0.5), ep, time_unit="a"
        )
    assert str(e_info.value) == "time_unit should be 's', 'ms' or 'us'"

    with pytest.raises(RuntimeError) as e_info:
        nap.compute_event_trigger_average(spikes, feature, 0.1, (0.5, 0.5, 0.5), ep)
    assert (
        str(e_info.value)
        == "windowsize should be a tuple of 2 numbers or a single number."
    )

    with pytest.raises(RuntimeError) as e_info:
        nap.compute_event_trigger_average(spikes, feature, 0.1, ("a", "b"), ep)
    assert (
        str(e_info.value)
        == "windowsize should be a tuple of 2 numbers or a single number."
    )

    with pytest.raises(TypeError) as e_info:
        nap.compute_event_trigger_average(spikes, feature, 0.1, (0.5, 0.5), [1, 2, 3])
    assert (
        str(e_info.value)
        == "Invalid type. Parameter ep must be of type ['IntervalSet']."
    )


def test_compute_spike_trigger_average_time_unit():
    ep = nap.IntervalSet(0, 100)
    t = np.arange(0, 101, 0.01)
    d = np.zeros(int(101 / 0.01))
    t1 = np.arange(1, 100)
    for i in range(len(t1)):
        d[t == t1[i]] = 1.0
    feature = nap.Tsd(t=t, d=d, time_support=ep)

    spikes = nap.TsGroup(
        {0: nap.Ts(t1), 1: nap.Ts(t1 - 0.1), 2: nap.Ts(t1 + 0.2)}, time_support=ep
    )

    sta = nap.compute_event_trigger_average(spikes, feature, 0.2, (0.6, 0.6), ep)

    output = np.zeros((7, 3))
    output[3, 0] = 0.05
    output[4, 1] = 0.05
    output[2, 2] = 0.05

    binsize = 0.2
    windowsize = np.array([0.6, 0.6])

    for tu, fa in zip(["s", "ms", "us"], [1, 1e3, 1e6]):
        sta = nap.compute_event_trigger_average(
            spikes, feature, binsize * fa, tuple(windowsize * fa), ep, time_unit=tu
        )
        assert isinstance(sta, nap.TsdFrame)
        assert sta.shape == output.shape
        np.testing.assert_array_almost_equal(sta.values, output)


@pytest.mark.filterwarnings("ignore")
def test_compute_spike_trigger_average_no_windows():
    ep = nap.IntervalSet(0, 100)
    feature = pd.Series(index=np.arange(0, 101, 0.01), data=np.zeros(int(101 / 0.01)))
    t1 = np.arange(1, 100)
    feature.loc[t1] = 1.0
    spikes = nap.TsGroup(
        {0: nap.Ts(t1), 1: nap.Ts(t1 - 0.1), 2: nap.Ts(t1 + 0.2)}, time_support=ep
    )

    feature = nap.Tsd(feature, time_support=ep)

    sta = nap.compute_event_trigger_average(spikes, feature, 0.2, ep=ep)

    output = np.zeros((1, 3))
    output[0, 0] = 0.05

    assert isinstance(sta, nap.TsdFrame)
    assert sta.shape == output.shape
    np.testing.assert_array_almost_equal(sta.values, output)


def test_compute_spike_trigger_average_multiple_epochs():
    ep = nap.IntervalSet(start=[0, 200], end=[100, 300])
    feature = nap.Tsd(
        t=np.hstack((np.arange(0, 100, 0.001), np.arange(200, 300, 0.001))),
        d=np.hstack((np.random.randn(100000), np.random.randn(100000))),
        time_support=ep,
    )
    t1 = np.hstack(
        (
            np.sort(np.random.uniform(0, 100, 1000)),
            np.sort(np.random.uniform(200, 300, 1000)),
        )
    )
    spikes = nap.TsGroup({0: nap.Ts(t1)}, time_support=ep)

    group = spikes
    binsize = 0.1
    windowsize = (1.0, 1.0)

    sta = nap.compute_event_trigger_average(spikes, feature, binsize, windowsize, ep)

    start, end = windowsize
    idx1 = -np.arange(0, start + binsize, binsize)[::-1][:-1]
    idx2 = np.arange(0, end + binsize, binsize)[1:]
    time_idx = np.hstack((idx1, np.zeros(1), idx2))
    from scipy.linalg import hankel

    n_p = len(idx1)
    n_f = len(idx2)

    sta2 = []
    for i in range(2):
        count = group.count(binsize, ep[i])
        tmp = feature.bin_average(binsize, ep[i])

        # Build the Hankel matrix
        pad_tmp = np.pad(tmp.values, (n_p, n_f))
        offset_tmp = hankel(pad_tmp, pad_tmp[-(n_p + n_f + 1) :])[0 : len(tmp)]
        stai = np.dot(offset_tmp.T, count.values)
        stai = stai / np.sum(count.values, 0)
        sta2.append(stai)

    sta2 = np.hstack(sta2).mean(1)

    np.testing.assert_array_almost_equal(sta.values[:, 0], sta2)
