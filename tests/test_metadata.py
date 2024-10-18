"""Tests for metadata in IntervalSet, TsdFrame, and TsGroup"""

import pickle
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from contextlib import nullcontext as does_not_raise

import pynapple as nap


#####################
# IntervalSet tests #
#####################
@pytest.fixture
def iset():
    start = np.array([0, 10, 16, 25])
    end = np.array([5, 15, 20, 40])
    return nap.IntervalSet(start=start, end=end)


def test_create_iset_with_metadata():
    start = np.array([0, 10, 16, 25])
    end = np.array([5, 15, 20, 40])
    sr_info = pd.Series(index=[0, 1, 2, 3], data=[0, 0, 0, 0], name="sr")
    ar_info = np.ones(4)
    ep = nap.IntervalSet(start=start, end=end, sr=sr_info, ar=ar_info)
    assert ep._metadata.shape == (4, 2)
    np.testing.assert_array_almost_equal(ep._metadata["sr"].values, sr_info.values)
    np.testing.assert_array_almost_equal(
        ep._metadata["sr"].index.values, sr_info.index.values
    )
    np.testing.assert_array_almost_equal(ep._metadata["ar"].values, ar_info)

    # test adding metadata with single interval
    start = 0
    end = 10
    label = ["a", "b"]
    ep = nap.IntervalSet(start=start, end=end, label=label)
    assert ep._metadata["label"][0] == label


@pytest.mark.parametrize(
    "obj",
    [
        nap.IntervalSet(start=np.array([0, 10, 16, 25]), end=np.array([5, 15, 20, 40])),
        nap.TsdFrame(
            t=np.arange(100),
            d=np.random.rand(100, 4),
            time_units="s",
        ),
        nap.TsGroup(
            {
                0: nap.Ts(t=np.arange(0, 200)),
                1: nap.Ts(t=np.arange(0, 200, 0.5), time_units="s"),
                2: nap.Ts(t=np.arange(0, 300, 0.2), time_units="s"),
                3: nap.Ts(t=np.arange(0, 400, 1), time_units="s"),
            }
        ),
    ],
)
class Test_Metadata:
    @pytest.mark.parametrize(
        "info",
        [
            # pd.Series
            pd.Series(index=[0, 1, 2, 3], data=[1, 1, 1, 1], name="label"),
            # np.ndarray
            np.ones(4) * 2,
            # list
            [3, 3, 3, 3],
            # tuple
            (4, 4, 4, 4),
        ],
    )
    def test_set_metadata(self, obj, info):
        obj.set_info(label=info)

        # verify shape of metadata
        if isinstance(obj, nap.TsGroup):
            assert obj._metadata.shape == (4, 2)
        else:
            assert obj._metadata.shape == (4, 1)

        # verify value in private metadata
        if isinstance(info, pd.Series):
            pd.testing.assert_series_equal(obj._metadata["label"], info)
        else:
            np.testing.assert_array_almost_equal(obj._metadata["label"].values, info)

        # verify public retrieval of metadata
        pd.testing.assert_series_equal(obj.get_info("label"), obj._metadata["label"])
        pd.testing.assert_series_equal(obj.label, obj._metadata["label"])
        pd.testing.assert_series_equal(obj["label"], obj._metadata["label"])

    @pytest.mark.parametrize(
        "info", [pd.DataFrame(index=[0, 1, 2, 3], data=[0, 0, 0, 0], columns=["label"])]
    )
    def test_set_metadata_df(self, obj, info):
        # add metadata with `set_info`
        obj.set_info(info)

        # verify shape and value in private metadata
        if isinstance(obj, nap.TsGroup):
            assert obj._metadata.shape == (4, 2)
            pd.testing.assert_series_equal(obj._metadata["label"], info["label"])
        else:
            assert obj._metadata.shape == (4, 1)
            pd.testing.assert_frame_equal(obj._metadata, info)

        # verify public retrieval of metadata
        pd.testing.assert_series_equal(obj.get_info("label"), obj._metadata["label"])
        pd.testing.assert_series_equal(obj.label, obj._metadata["label"])
        pd.testing.assert_series_equal(obj["label"], obj._metadata["label"])

    @pytest.mark.parametrize(
        "info",
        [
            # pd.Series
            pd.Series(index=[0, 1, 2, 3], data=[1, 1, 1, 1], name="label"),
            # np.ndarray
            np.ones(4) * 2,
            # list
            [3, 3, 3, 3],
            # tuple
            (4, 4, 4, 4),
        ],
    )
    def test_add_metadata_key(self, obj, info):
        # add metadata as key
        obj["label"] = info

        # verify shape of metadata
        if isinstance(obj, nap.TsGroup):
            assert obj._metadata.shape == (4, 2)
        else:
            assert obj._metadata.shape == (4, 1)

        # verify value in private metadata
        if isinstance(info, pd.Series):
            pd.testing.assert_series_equal(obj._metadata["label"], info)
        else:
            np.testing.assert_array_almost_equal(obj._metadata["label"].values, info)

        # verify public retrieval of metadata
        pd.testing.assert_series_equal(obj.get_info("label"), obj._metadata["label"])
        pd.testing.assert_series_equal(obj.label, obj._metadata["label"])
        pd.testing.assert_series_equal(obj["label"], obj._metadata["label"])

    @pytest.mark.parametrize(
        "args, kwargs, expected",
        [
            (
                # invalid names as integers
                [pd.DataFrame(data=np.random.randint(0, 5, size=(3, 3)))],
                {},
                pytest.raises(TypeError, match="Invalid metadata type"),
            ),
            (
                # invalid names as strings starting with a number
                [
                    pd.DataFrame(
                        columns=["1"],
                        data=np.ones((4, 1)),
                    )
                ],
                {},
                pytest.raises(ValueError, match="Invalid metadata name"),
            ),
            (
                # invalid names with spaces
                [
                    pd.DataFrame(
                        columns=["l 1"],
                        data=np.ones((4, 1)),
                    )
                ],
                {},
                pytest.raises(ValueError, match="Invalid metadata name"),
            ),
            (
                # invalid names with periods
                [
                    pd.DataFrame(
                        columns=["l.1"],
                        data=np.ones((4, 1)),
                    )
                ],
                {},
                pytest.raises(ValueError, match="Invalid metadata name"),
            ),
            (
                # invalid names with dashes
                [
                    pd.DataFrame(
                        columns=["l-1"],
                        data=np.ones((4, 1)),
                    )
                ],
                {},
                pytest.raises(ValueError, match="Invalid metadata name"),
            ),
            (
                # name that overlaps with existing attribute
                [],
                {"__dir__": np.zeros(4)},
                pytest.raises(ValueError, match="Invalid metadata name"),
            ),
            (
                # metadata with wrong length
                [],
                {"label": np.zeros(5)},
                pytest.raises(RuntimeError, match="Array is not the same length."),
            ),
        ],
    )
    def test_add_metadata_error(self, obj, args, kwargs, expected):
        with expected:
            obj.set_info(*args, **kwargs)

    def test_add_metadata_key_error(self, obj):
        # type specific key errors
        info = np.ones(4)
        if isinstance(obj, nap.IntervalSet):
            with pytest.raises(RuntimeError, match="IntervalSet is immutable"):
                obj[0] = info
            with pytest.raises(RuntimeError, match="IntervalSet is immutable"):
                obj["start"] = info
            with pytest.raises(RuntimeError, match="IntervalSet is immutable"):
                obj["end"] = info

        elif isinstance(obj, nap.TsGroup):
            # currently obj[0] does not raise an error for TsdFrame
            with pytest.raises(TypeError, match="Metadata keys must be strings!"):
                obj[0] = info

    @pytest.mark.parametrize("label, val", [([1, 1, 2, 2], 2)])
    def test_metadata_slicing(self, obj, label, val):
        # add label
        obj.set_info(label=label, extra=[0, 1, 2, 3])

        # test slicing
        obj2 = obj[obj.label == val]
        assert isinstance(obj2, type(obj))
        assert np.all(obj2.label == val)
        if isinstance(obj, nap.IntervalSet):
            # interval set slicing resets index
            pd.testing.assert_frame_equal(
                obj2._metadata, obj._metadata[obj.label == val].reset_index(drop=True)
            )
        else:
            # other types do not reset index
            pd.testing.assert_frame_equal(
                obj2._metadata, obj._metadata[obj.label == val]
            )

        # type specific checks
        if isinstance(obj, nap.IntervalSet):
            # slicing will update rows
            np.testing.assert_array_almost_equal(
                obj2.values, obj.values[obj.label == val]
            )
            # number of columns should be the same
            assert np.all(obj2.columns == obj.columns)

        elif isinstance(obj, nap.TsdFrame):
            # slicing will update columns
            np.testing.assert_array_almost_equal(
                obj2.columns, obj.columns[obj.label == val]
            )
            np.testing.assert_array_almost_equal(obj2.metadata_index, obj2.columns)
            # number of rows should be the same
            assert len(obj2) == len(obj)

        elif isinstance(obj, nap.TsGroup):
            # slicing will update keys
            np.testing.assert_array_almost_equal(
                obj2.index, obj.index[obj.label == val]
            )
            # length of values should be the same
            assert np.all(
                len(values1) == len(values2)
                for values1, values2 in zip(obj.values(), obj2.values())
            )

        # metadata columns should be the same
        assert np.all(obj2.metadata_columns == obj.metadata_columns)
