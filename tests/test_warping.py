import re

import numpy as np
import pytest

import pynapple as nap


############################################################
# Test for warping
############################################################
def get_input():
    return nap.Tsd(t=np.arange(10), d=np.arange(10))


def get_ep():
    return nap.IntervalSet(0, 10)


@pytest.mark.parametrize(
    "input, ep, binsize, align, padding_value, time_unit, expectation",
    [
        (
            {},
            get_ep(),
            1,
            "start",
            np.nan,
            "s",
            "Invalid type. Parameter input must be of type ['Ts', 'Tsd', 'TsdFrame', 'TsdTensor', 'TsGroup'].",
        ),
        (
            get_input(),
            {},
            1,
            "start",
            np.nan,
            "s",
            "Invalid type. Parameter ep must be of type ['IntervalSet'].",
        ),
        (
            get_input(),
            get_ep(),
            "a",
            "start",
            np.nan,
            "s",
            "Invalid type. Parameter binsize must be of type ['Number'].",
        ),
        (
            get_input(),
            get_ep(),
            1,
            1,
            np.nan,
            "s",
            "Invalid type. Parameter align must be of type ['str'].",
        ),
        (
            get_input(),
            get_ep(),
            1,
            "start",
            {},
            "s",
            "Invalid type. Parameter padding_value must be of type ['Number'].",
        ),
        (
            get_input(),
            get_ep(),
            1,
            "start",
            np.nan,
            1,
            "Invalid type. Parameter time_unit must be of type ['str'].",
        ),
    ],
)
def test_build_tensor_type_error(
    input, ep, binsize, align, padding_value, time_unit, expectation
):
    with pytest.raises(TypeError, match=re.escape(expectation)):
        nap.build_tensor(
            input=input,
            ep=ep,
            binsize=binsize,
            align=align,
            padding_value=padding_value,
            time_unit=time_unit,
        )


def test_build_tensor_runtime_error():
    group = nap.TsGroup(
        {
            0: nap.Ts(t=np.arange(0, 100)),
            1: nap.Ts(t=np.arange(0, 100, 0.5), time_units="s"),
            2: nap.Ts(t=np.arange(0, 100, 0.2), time_units="s"),
        }
    )
    ep = nap.IntervalSet(
        start=np.arange(0, 100, 20), end=np.arange(0, 100, 20) + np.arange(0, 10, 2)
    )

    with pytest.raises(
        RuntimeError, match=r"When input is a TsGroup, binsize should be specified"
    ):
        nap.build_tensor(group, ep)

    with pytest.raises(RuntimeError, match=r"time_unit should be 's', 'ms' or 'us'"):
        nap.build_tensor(group, ep, 1, time_unit="a")

    with pytest.raises(RuntimeError, match=r"align should be 'start' or 'end'"):
        nap.build_tensor(group, ep, 1, align="a")


def test_build_tensor_with_group():
    group = nap.TsGroup(
        {
            0: nap.Ts(t=np.arange(0, 100)),
            1: nap.Ts(t=np.arange(0, 100, 0.5), time_units="s"),
            2: nap.Ts(t=np.arange(0, 100, 0.2), time_units="s"),
        }
    )
    ep = nap.IntervalSet(
        start=np.arange(0, 100, 20), end=np.arange(0, 100, 20) + np.arange(0, 10, 2)
    )

    expected = np.ones((len(group), len(ep), 8)) * np.nan
    for i, k in zip(range(len(ep)), range(2, 10, 2)):
        expected[:, i, 0:k] = 1
    for i, k in zip(range(len(group)), [1, 2, 5]):
        expected[i] *= k

    tensor = nap.build_tensor(group, ep, binsize=1)
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(group, ep, binsize=1, align="start")
    np.testing.assert_array_almost_equal(tensor, expected)

    tensor = nap.build_tensor(group, ep, binsize=1, align="end")
    np.testing.assert_array_almost_equal(tensor, np.flip(expected, axis=2))
