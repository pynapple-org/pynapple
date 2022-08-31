# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-08-29 17:27:02
# @Last Modified by:   gviejo
# @Last Modified time: 2022-08-30 20:38:22
#!/usr/bin/env python

"""Tests of spike trigger average for `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest


def test_compute_spike_trigger_average():
    ep = nap.IntervalSet(0, 101)
    feature = nap.Tsd(
        t=np.arange(0, 101, 0.01), d=np.zeros(int(101 / 0.01)), time_support=ep
    )
    t1 = np.arange(1, 101)
    feature.loc[t1] = 1.0
    spikes = nap.TsGroup(
        {0: nap.Ts(t1), 1: nap.Ts(t1 - 0.05), 2: nap.Ts(t1 + 0.25)}, time_support=ep
    )

    sta = nap.compute_event_trigger_average(spikes, feature, 0.1, (0.5, 0.5), ep)

    output = np.zeros((int((0.5 / 00.1) * 2 + 1), 3))
    output[int((0.5 / 0.1)), 0] = 0.1
    output[int((0.5 / 0.1) + 1), 1] = 0.1
    output[int((0.5 / 0.1) - 2), 2] = 0.1

    assert isinstance(sta, nap.TsdFrame)
    assert sta.shape == output.shape
    np.testing.assert_array_almost_equal(sta, output)


def test_compute_spike_trigger_average_raise_error():
    ep = nap.IntervalSet(0, 101)
    feature = nap.Tsd(
        t=np.arange(0, 101, 0.01), d=np.zeros(int(101 / 0.01)), time_support=ep
    )
    t1 = np.arange(1, 101) + 0.01
    feature.loc[t1] = 1.0

    with pytest.raises(Exception) as e_info:
        nap.compute_event_trigger_average(feature, feature, 0.1, (0.5, 0.5), ep)
    assert str(e_info.value) == "Unknown format for group"


def test_compute_spike_trigger_average_time_units():
    ep = nap.IntervalSet(0, 101)
    feature = nap.Tsd(
        t=np.arange(0, 101, 0.01), d=np.zeros(int(101 / 0.01)), time_support=ep
    )
    t1 = np.arange(1, 101)
    feature.loc[t1] = 1.0
    spikes = nap.TsGroup(
        {0: nap.Ts(t1), 1: nap.Ts(t1 - 0.05), 2: nap.Ts(t1 + 0.25)}, time_support=ep
    )

    sta = nap.compute_event_trigger_average(spikes, feature, 0.1, (0.5, 0.5), ep)

    output = np.zeros((int((0.5 / 00.1) * 2 + 1), 3))
    output[int((0.5 / 0.1)), 0] = 0.1
    output[int((0.5 / 0.1) + 1), 1] = 0.1
    output[int((0.5 / 0.1) - 2), 2] = 0.1

    binsize = 0.1
    windowsize = np.array([0.5, 0.5])

    for tu, fa in zip(["s", "ms", "us"], [1, 1e3, 1e6]):
        sta = nap.compute_event_trigger_average(
            spikes, feature, binsize * fa, tuple(windowsize * fa), ep, time_units=tu
        )
        assert isinstance(sta, nap.TsdFrame)
        assert sta.shape == output.shape
        np.testing.assert_array_almost_equal(sta.values, output)


def test_compute_spike_trigger_average_multiple_epochs():
    ep = nap.IntervalSet(0, 101)
    feature = nap.Tsd(
        t=np.arange(0, 101, 0.01), d=np.zeros(int(101 / 0.01)), time_support=ep
    )
    t1 = np.arange(1, 101)
    feature.loc[t1] = 1.0
    spikes = nap.TsGroup({0: nap.Ts(t1 + 0.25)}, time_support=ep)

    ep2 = nap.IntervalSet(start=[0, 40], end=[11, 61])

    sta = nap.compute_event_trigger_average(spikes, feature, 0.1, (0.5, 0.5), ep2)

    output = np.zeros(int((0.5 / 00.1) * 2 + 1))
    output[int((0.5 / 0.1) - 2)] = 0.1

    assert isinstance(sta, nap.TsdFrame)
    np.testing.assert_array_almost_equal(sta.values.flatten(), output)
