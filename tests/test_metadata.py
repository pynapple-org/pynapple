"""Tests for metadata in IntervalSet, TsdFrame, and TsGroup"""

import inspect
import pickle
import warnings
from contextlib import nullcontext as does_not_raise
from numbers import Number
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pynapple as nap


#################
## IntervalSet ##
#################
@pytest.fixture
def iset_meta():
    start = np.array([0, 10, 16, 25])
    end = np.array([5, 15, 20, 40])
    metadata = {"label": ["a", "b", "c", "d"], "info": np.arange(4)}
    return nap.IntervalSet(start=start, end=end, metadata=metadata)


@pytest.fixture
def label_meta():
    return {"label": [1, 2, 3, 4]}


def test_create_iset_with_metadata():
    start = np.array([0, 10, 16, 25])
    end = np.array([5, 15, 20, 40])
    sr_info = pd.Series(index=[0, 1, 2, 3], data=[0, 0, 0, 0], name="sr")
    ar_info = np.ones(4)
    lt_info = [2, 2, 2, 2]
    tu_info = (3, 3, 3, 3)
    metadata = {
        "sr": sr_info,
        "ar": ar_info,
        "lt": lt_info,
        "tu": tu_info,
    }
    ep = nap.IntervalSet(start=start, end=end, metadata=metadata)
    assert ep._metadata.shape == (4, 4)
    np.testing.assert_array_almost_equal(ep._metadata["sr"].values, sr_info.values)
    np.testing.assert_array_almost_equal(
        ep._metadata["sr"].index.values, sr_info.index.values
    )
    np.testing.assert_array_almost_equal(ep._metadata["ar"].values, ar_info)

    # test adding metadata with single interval
    start = 0
    end = 10
    label = [["a", "b"]]
    metadata = {"label": label}
    ep = nap.IntervalSet(start=start, end=end, metadata=metadata)
    assert ep._metadata["label"][0] == label[0]


@pytest.mark.parametrize(
    "start, end",
    [
        # start time not sorted
        (
            np.array([10, 5, 16, 25]),
            np.array([5, 15, 20, 40]),
        ),
        # end time not sorted
        (
            np.array([5, 10, 16, 25]),
            np.array([15, 5, 20, 40]),
        ),
        # overlapping intervals
        (
            np.array([0, 5, 16, 25]),
            np.array([10, 15, 20, 40]),
        ),
    ],
)
def test_create_iset_with_metadata_warn_drop(start, end, label_meta):
    with warnings.catch_warnings(record=True) as w:
        ep = nap.IntervalSet(start=start, end=end, metadata=label_meta)
    assert "dropping metadata" in str(w[-1].message)
    assert len(ep.metadata_columns) == 0


@pytest.mark.parametrize(
    "start, end",
    [
        # start and end are equal
        (
            np.array([0, 5, 16, 25]),
            np.array([5, 15, 20, 40]),
        ),
    ],
)
def test_create_iset_with_metadata_warn_keep(start, end, label_meta):
    with warnings.catch_warnings(record=True) as w:
        ep = nap.IntervalSet(start=start, end=end, metadata=label_meta)
    assert "dropping metadata" not in str(w[-1].message)
    assert len(ep.metadata_columns) == 1


def test_create_iset_from_df_with_metadata():
    df = pd.DataFrame(data=[[16, 100, "a"]], columns=["start", "end", "label"])
    ep = nap.IntervalSet(df)
    np.testing.assert_array_almost_equal(df.start.values, ep.start)
    np.testing.assert_array_almost_equal(df.end.values, ep.end)


@pytest.mark.parametrize(
    "df, expected",
    [
        # dataframe is sorted and metadata is kept
        (
            pd.DataFrame(
                {
                    "start": [25.0, 0.0, 10.0, 16.0],
                    "end": [40.0, 5.0, 15.0, 20.0],
                    "label": np.arange(4),
                }
            ),
            ["DataFrame is not sorted by start times"],
        ),
        (
            # dataframe is sorted and and metadata is dropped
            pd.DataFrame(
                {
                    "start": [25, 0, 10, 16],
                    "end": [40, 20, 15, 20],
                    "label": np.arange(4),
                }
            ),
            ["DataFrame is not sorted by start times", "dropping metadata"],
        ),
    ],
)
def test_create_iset_from_df_with_metadata_sort(df, expected):
    with warnings.catch_warnings(record=True) as w:
        ep = nap.IntervalSet(df)
    for e in expected:
        assert np.any([e in str(w.message) for w in w])
    if "dropping metadata" not in expected:
        pd.testing.assert_frame_equal(
            ep.as_dataframe(), df.sort_values("start").reset_index(drop=True)
        )


@pytest.mark.parametrize(
    "index",
    [
        0,
        slice(0, 2),
        [0, 2],
        (slice(0, 2), slice(None)),
        (slice(0, 2), slice(0, 2)),
        (slice(None), ["start", "end"]),
        (0, slice(None)),
    ],
)
def test_get_iset_with_metadata(iset_meta, index):
    assert isinstance(iset_meta[index], nap.IntervalSet)


@pytest.mark.parametrize(
    "index, expected",
    [
        ((slice(None), 0), "start"),
        ((slice(None), 1), "end"),
        ((slice(None), "end"), "end"),
        ((slice(None), "label"), "label"),
        ((slice(None), ["end", "label"]), ["end", "label"]),
        ((0, [0, 1]), ([0], ["start", "end"])),
        ((0, slice(None)), ([0], slice(None))),
        (([1, 2], slice(None)), ([1, 2], slice(None))),
        (([1, 2], ["end", "label"]), ([1, 2], ["end", "label"])),
    ],
)
def test_slice_iset_with_metadata(iset_meta, index, expected):
    if isinstance(expected, str):
        if (expected == "start") or (expected == "end"):
            # start and end returned as array
            np.testing.assert_array_almost_equal(
                iset_meta[index], iset_meta.as_dataframe()[expected].values
            )
        else:
            # metadata returned as series
            pd.testing.assert_series_equal(
                iset_meta[index], iset_meta.as_dataframe()[expected]
            )
    elif isinstance(expected, list):
        pd.testing.assert_frame_equal(
            iset_meta[index], iset_meta.as_dataframe()[expected]
        )
    elif isinstance(expected, tuple):
        try:
            # index reset when IntervalSet is returned
            pd.testing.assert_frame_equal(
                iset_meta[index].as_dataframe(),
                iset_meta.as_dataframe().loc[expected].reset_index(drop=True),
            )
        except AttributeError:
            # index not reset when DataFrame is returned
            pd.testing.assert_frame_equal(
                iset_meta[index], iset_meta.as_dataframe().loc[expected]
            )


@pytest.mark.parametrize(
    "index, expected",
    [
        (
            (slice(None), pd.Series(index=[0, 1, 2, 3], data=[0, 0, 0, 0])),
            pytest.raises(
                IndexError,
                match="unknown type <class 'pandas.core.series.Series'> for index 2",
            ),
        ),
        (
            (slice(None), [pd.Series(index=[0, 1, 2, 3], data=[0, 0, 0, 0])]),
            pytest.raises(
                IndexError,
                match="unknown index",
            ),
        ),
        (
            pd.DataFrame(index=[0, 1, 2, 3], data=[0, 0, 0, 0]),
            pytest.raises(
                IndexError,
                match="unknown type <class 'pandas.core.frame.DataFrame'> for index",
            ),
        ),
        (
            (slice(None), 2),
            pytest.raises(
                IndexError,
                match="index 2 is out of bounds for axis 1 with size 2",
            ),
        ),
        (
            (slice(None), slice(1, 3)),
            pytest.raises(
                IndexError,
                match="index slice\\(1, 3, None\\) out of bounds for IntervalSet axis 1 with size 2",
            ),
        ),
        (
            (slice(None), [0, 3]),
            pytest.raises(
                IndexError,
                match="index \\[0, 3\\] out of bounds for IntervalSet axis 1 with size 2",
            ),
        ),
    ],
)
def test_get_iset_with_metadata_errors(iset_meta, index, expected):
    with expected:
        iset_meta[index]


def test_as_dataframe_metadata():
    ep = nap.IntervalSet(start=0, end=100, metadata={"m1": 0, "m2": 1})
    df = pd.DataFrame(
        data=np.array([[0.0, 100.0, 0, 1]]),
        columns=["start", "end", "m1", "m2"],
        dtype=np.float64,
    )
    np.testing.assert_array_almost_equal(df.values, ep.as_dataframe().values)


def test_intersect_metadata():
    ep = nap.IntervalSet(start=[0, 50], end=[30, 70], metadata={"m1": [0, 1]})
    ep2 = nap.IntervalSet(start=20, end=60, metadata={"m2": 2})
    ep3 = nap.IntervalSet(
        start=[20, 50], end=[30, 60], metadata={"m1": [0, 1], "m2": [2, 2]}
    )
    np.testing.assert_array_almost_equal(ep.intersect(ep2).values, ep3.values)
    np.testing.assert_array_almost_equal(ep2.intersect(ep).values, ep3.values)
    pd.testing.assert_series_equal(
        ep.intersect(ep2)._metadata["m1"], ep3._metadata["m1"]
    )
    pd.testing.assert_series_equal(
        ep.intersect(ep2)._metadata["m2"], ep3._metadata["m2"]
    )
    pd.testing.assert_series_equal(
        ep2.intersect(ep)._metadata["m1"], ep3._metadata["m1"]
    )
    pd.testing.assert_series_equal(
        ep2.intersect(ep)._metadata["m2"], ep3._metadata["m2"]
    )


def test_set_diff_metadata():
    ep = nap.IntervalSet(start=[0, 60], end=[50, 80], metadata={"m1": [0, 1]})
    ep2 = nap.IntervalSet(start=[20, 40], end=[30, 70], metadata={"m2": [2, 3]})
    ep3 = nap.IntervalSet(
        start=[0, 30, 70], end=[20, 40, 80], metadata={"m1": [0, 0, 1]}
    )
    np.testing.assert_array_almost_equal(ep.set_diff(ep2).values, ep3.values)
    pd.testing.assert_series_equal(
        ep.set_diff(ep2)._metadata["m1"], ep3._metadata["m1"]
    )
    ep4 = nap.IntervalSet(start=50, end=60, metadata={"m2": [3]})
    np.testing.assert_array_almost_equal(ep2.set_diff(ep).values, ep4.values)
    pd.testing.assert_series_equal(
        ep2.set_diff(ep)._metadata["m2"], ep4._metadata["m2"]
    )


def test_drop_short_intervals_metadata(iset_meta):
    iset_dropped = iset_meta.drop_short_intervals(5)
    print(iset_dropped)
    assert np.all(iset_dropped.metadata_columns == iset_meta.metadata_columns)
    assert len(iset_dropped._metadata) == 1  # one interval left
    assert iset_dropped.metadata_index == 0  # index reset to 0
    # label of remaining interval should be "d"
    assert iset_dropped._metadata["label"][0] == "d"


def test_drop_long_intervals_metadata(iset_meta):
    iset_dropped = iset_meta.drop_long_intervals(5)
    print(iset_dropped)
    assert np.all(iset_dropped.metadata_columns == iset_meta.metadata_columns)
    assert len(iset_dropped._metadata) == 1  # one interval left
    assert iset_dropped.metadata_index == 0  # index reset to 0
    # label of remaining interval should be "c"
    assert iset_dropped._metadata["label"][0] == "c"


def test_split_metadata(iset_meta):
    iset_split = iset_meta.split(1)
    for i, iset in enumerate(iset_meta):
        # check number of labels in each split
        iset_i = iset_split[iset_split.info == i]
        assert len(iset_i) == (iset.end - iset.start)
        # check first start and last end
        start_end = iset_i.values[[0, -1]].ravel()[[0, -1]]
        np.testing.assert_array_almost_equal(start_end, iset.values[0])


def test_drop_metadata_warnings(iset_meta):
    with pytest.warns(UserWarning, match="metadata incompatible"):
        iset_meta.merge_close_intervals(1)
    with pytest.warns(UserWarning, match="metadata incompatible"):
        iset_meta.union(iset_meta)
    with pytest.warns(UserWarning, match="metadata incompatible"):
        iset_meta.time_span()


@pytest.mark.parametrize(
    "name, set_exp, set_attr_exp, set_key_exp, get_attr_exp, get_key_exp",
    [
        # existing attribute and key
        (
            "start",
            # warn with set_info
            pytest.warns(UserWarning, match="overlaps with an existing"),
            # error with setattr
            pytest.raises(AttributeError, match="IntervalSet is immutable"),
            # error with setitem
            pytest.raises(RuntimeError, match="IntervalSet is immutable"),
            # attr should not match metadata
            pytest.raises(AssertionError),
            # key should not match metadata
            pytest.raises(AssertionError),
        ),
        # existing attribute and key
        (
            "end",
            # warn with set_info
            pytest.warns(UserWarning, match="overlaps with an existing"),
            # error with setattr
            pytest.raises(AttributeError, match="IntervalSet is immutable"),
            # error with setitem
            pytest.raises(RuntimeError, match="IntervalSet is immutable"),
            # attr should not match metadata
            pytest.raises(AssertionError),
            # key should not match metadata
            pytest.raises(AssertionError),
        ),
        # existing attribute
        (
            "values",
            # warn with set_info
            pytest.warns(UserWarning, match="overlaps with an existing"),
            # error with setattr
            pytest.raises(AttributeError, match="IntervalSet is immutable"),
            # warn with setitem
            pytest.warns(UserWarning, match="overlaps with an existing"),
            # attr should not match metadata
            pytest.raises(AssertionError),
            # key should match metadata
            does_not_raise(),
        ),
        # existing metdata
        (
            "label",
            # no warning with set_info
            does_not_raise(),
            # no warning with setattr
            does_not_raise(),
            # no warning with setitem
            does_not_raise(),
            # attr should match metadata
            does_not_raise(),
            # key should match metadata
            does_not_raise(),
        ),
    ],
)
def test_iset_metadata_overlapping_names(
    iset_meta, name, set_exp, set_attr_exp, set_key_exp, get_attr_exp, get_key_exp
):
    assert hasattr(iset_meta, name)

    # warning when set
    with set_exp:
        iset_meta.set_info({name: np.ones(4)})
    # error when set as attribute
    with set_attr_exp:
        setattr(iset_meta, name, np.ones(4))
    # error when set as key
    with set_key_exp:
        iset_meta[name] = np.ones(4)
    # retrieve with get_info
    np.testing.assert_array_almost_equal(iset_meta.get_info(name), np.ones(4))
    # make sure it doesn't access metadata if its an existing attribute or key
    with get_attr_exp:
        np.testing.assert_array_almost_equal(getattr(iset_meta, name), np.ones(4))
    # make sure it doesn't access metadata if its an existing key
    with get_key_exp:
        np.testing.assert_array_almost_equal(iset_meta[name], np.ones(4))


##############
## TsdFrame ##
##############
@pytest.fixture
def tsdframe_meta():
    return nap.TsdFrame(
        t=np.arange(100),
        d=np.random.rand(100, 4),
        time_units="s",
        columns=["a", "b", "c", "d"],
        metadata={"l1": np.arange(4), "l2": ["x", "x", "y", "y"]},
    )


def test_tsdframe_metadata_slicing(tsdframe_meta):
    # test slicing obj[obj.mcol == mval] and obj[:, obj.mcol == mval], and that they produce the same results
    if len(tsdframe_meta.metadata_columns):
        for mcol in tsdframe_meta.metadata_columns:
            mval = tsdframe_meta._metadata[mcol].iloc[0]
            fcols = tsdframe_meta._metadata[tsdframe_meta._metadata[mcol] == mval].index
            assert isinstance(tsdframe_meta[tsdframe_meta[mcol] == mval], nap.TsdFrame)
            assert np.all(tsdframe_meta[tsdframe_meta[mcol] == mval].columns == fcols)
            assert np.all(
                tsdframe_meta[tsdframe_meta[mcol] == mval].metadata_index == fcols
            )
            assert isinstance(
                tsdframe_meta[:, tsdframe_meta[mcol] == mval], nap.TsdFrame
            )
            np.testing.assert_array_almost_equal(
                tsdframe_meta[tsdframe_meta[mcol] == mval].values,
                tsdframe_meta[:, tsdframe_meta[mcol] == mval].values,
            )


@pytest.mark.parametrize(
    "name, attr_exp, set_exp, set_attr_exp, set_key_exp, get_exp, get_attr_exp, get_key_exp",
    [
        # existing data column
        (
            "a",
            # not attribute
            pytest.raises(AssertionError),
            # error with set_info
            pytest.raises(ValueError, match="Invalid metadata name"),
            # error with setattr
            pytest.raises(ValueError, match="Invalid metadata name"),
            # shape mismatch with setitem
            pytest.raises(ValueError),
            # assertion error with get_info
            pytest.raises(AssertionError),
            # attribute should raise error
            pytest.raises(AttributeError),
            # key should not match metadata
            pytest.raises(AssertionError),
        ),
        (
            "columns",
            # attribute exists
            does_not_raise(),
            # warn with set_info
            pytest.warns(UserWarning, match="overlaps with an existing"),
            # cannot be set as attribute
            pytest.raises(AttributeError, match="Cannot set attribute"),
            # warn when set as key
            pytest.warns(UserWarning, match="overlaps with an existing"),
            # no error with get_info
            does_not_raise(),
            # attribute should not match metadata
            pytest.raises(TypeError),
            # key should match metadata
            does_not_raise(),
        ),
        # existing metdata
        (
            "l1",
            # attribute exists
            does_not_raise(),
            # no warning with set_info
            does_not_raise(),
            # no warning with setattr
            does_not_raise(),
            # no warning with setitem
            does_not_raise(),
            # no error with get_info
            does_not_raise(),
            # attr should match metadata
            does_not_raise(),
            # key should match metadata
            does_not_raise(),
        ),
    ],
)
def test_tsdframe_metadata_overlapping_names(
    tsdframe_meta,
    name,
    attr_exp,
    set_exp,
    set_attr_exp,
    get_exp,
    set_key_exp,
    get_attr_exp,
    get_key_exp,
):
    with attr_exp:
        assert hasattr(tsdframe_meta, name)
    # warning when set
    with set_exp:
        # warnings.simplefilter("error")
        tsdframe_meta.set_info({name: np.ones(4)})
    # error when set as attribute
    with set_attr_exp:
        setattr(tsdframe_meta, name, np.ones(4))
    # error when set as key
    with set_key_exp:
        tsdframe_meta[name] = np.ones(4)
    # retrieve with get_info
    with get_exp:
        np.testing.assert_array_almost_equal(tsdframe_meta.get_info(name), np.ones(4))
    # make sure it doesn't access metadata if its an existing attribute or key
    with get_attr_exp:
        np.testing.assert_array_almost_equal(getattr(tsdframe_meta, name), np.ones(4))
    # make sure it doesn't access metadata if its an existing key
    with get_key_exp:
        np.testing.assert_array_almost_equal(tsdframe_meta[name], np.ones(4))


#############
## TsGroup ##
#############
@pytest.fixture
def tsgroup_meta():
    return nap.TsGroup(
        {
            0: nap.Ts(t=np.arange(0, 200)),
            1: nap.Ts(t=np.arange(0, 200, 0.5), time_units="s"),
            2: nap.Ts(t=np.arange(0, 300, 0.2), time_units="s"),
            3: nap.Ts(t=np.arange(0, 400, 1), time_units="s"),
        },
        metadata={"label": [1, 2, 3, 4]},
    )


@pytest.mark.parametrize(
    "name, set_exp, set_attr_exp, set_key_exp, get_exp, get_attr_exp, get_key_exp",
    [
        # pre-computed rate metadata
        (
            "rate",
            # error with set_info
            pytest.raises(ValueError, match="Invalid metadata name"),
            # error with setattr
            pytest.raises(AttributeError, match="Cannot set attribute"),
            # error with setitem
            pytest.raises(ValueError, match="Invalid metadata name"),
            # value mismatch with get_info
            pytest.raises(AssertionError),
            # value mismatch with getattr
            pytest.raises(AssertionError),
            # value mismatch with getitem
            pytest.raises(AssertionError),
        ),
        # 'rates' attribute
        (
            "rates",
            # warning with set_info
            pytest.warns(UserWarning, match="overlaps with an existing"),
            # error with setattr
            pytest.raises(AttributeError, match="Cannot set attribute"),
            # warn with setitem
            pytest.warns(UserWarning, match="overlaps with an existing"),
            # no error with get_info
            does_not_raise(),
            # get attribute is not metadata
            pytest.raises(AssertionError),
            # get key is metadata
            does_not_raise(),
        ),
        # existing metdata
        (
            "label",
            # no warning with set_info
            does_not_raise(),
            # no warning with setattr
            does_not_raise(),
            # no warning with setitem
            does_not_raise(),
            # no error with get_info
            does_not_raise(),
            # attr should match metadata
            does_not_raise(),
            # key should match metadata
            does_not_raise(),
        ),
    ],
)
def test_tsgroup_metadata_overlapping_names(
    tsgroup_meta,
    name,
    set_exp,
    set_attr_exp,
    set_key_exp,
    get_exp,
    get_attr_exp,
    get_key_exp,
):
    assert hasattr(tsgroup_meta, name)

    # warning when set
    with set_exp:
        tsgroup_meta.set_info({name: np.ones(4)})
    # error when set as attribute
    with set_attr_exp:
        setattr(tsgroup_meta, name, np.ones(4))
    # error when set as key
    with set_key_exp:
        tsgroup_meta[name] = np.ones(4)
    # retrieve with get_info
    with get_exp:
        np.testing.assert_array_almost_equal(tsgroup_meta.get_info(name), np.ones(4))
    # make sure it doesn't access metadata if its an existing attribute or key
    with get_attr_exp:
        np.testing.assert_array_almost_equal(getattr(tsgroup_meta, name), np.ones(4))
    # make sure it doesn't access metadata if its an existing key
    with get_key_exp:
        np.testing.assert_array_almost_equal(tsgroup_meta[name], np.ones(4))


def test_tsgroup_metadata_future_warnings():
    with pytest.warns(FutureWarning, match="may be unsupported"):
        tsgroup = nap.TsGroup(
            {
                0: nap.Ts(t=np.arange(0, 200)),
                1: nap.Ts(t=np.arange(0, 200, 0.5), time_units="s"),
                2: nap.Ts(t=np.arange(0, 300, 0.2), time_units="s"),
                3: nap.Ts(t=np.arange(0, 400, 1), time_units="s"),
            },
            label=[1, 2, 3, 4],
        )


##################
## Shared tests ##
##################
@pytest.fixture
def clear_metadata(obj):
    if isinstance(obj, nap.TsGroup):
        columns = [col for col in obj.metadata_columns if col != "rate"]
    else:
        columns = obj.metadata_columns
    obj._metadata.drop(columns=columns, inplace=True)
    return obj


@pytest.mark.parametrize(
    "obj",
    [
        # IntervalSet length 4
        nap.IntervalSet(start=np.array([0, 10, 16, 25]), end=np.array([5, 15, 20, 40])),
        # IntervalSet length 1
        nap.IntervalSet(start=0, end=5),
        # TsdFrame with 4 columns
        nap.TsdFrame(
            t=np.arange(100),
            d=np.random.rand(100, 4),
            time_units="s",
        ),
        # TsdFrame with 1 column
        nap.TsdFrame(
            t=np.arange(100),
            d=np.random.rand(100, 1),
            time_units="s",
        ),
        # TsGroup length 4
        nap.TsGroup(
            {
                0: nap.Ts(t=np.arange(0, 200)),
                1: nap.Ts(t=np.arange(0, 200, 0.5), time_units="s"),
                2: nap.Ts(t=np.arange(0, 300, 0.2), time_units="s"),
                3: nap.Ts(t=np.arange(0, 400, 1), time_units="s"),
            }
        ),
        # TsGroup length 1
        nap.TsGroup(
            {
                0: nap.Ts(t=np.arange(0, 200)),
            }
        ),
    ],
)
@pytest.mark.usefixtures("clear_metadata")
class Test_Metadata:

    @pytest.fixture
    def obj_len(self, obj):
        if isinstance(obj, nap.TsdFrame):
            return len(obj.columns)
        else:
            return len(obj)

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
    class Test_Add_Metadata:

        def test_add_metadata(self, obj, info, obj_len):
            obj.set_info(label=info[:obj_len])

            # verify shape of metadata
            if isinstance(obj, nap.TsGroup):
                assert obj._metadata.shape == (obj_len, 2)
            else:
                assert obj._metadata.shape == (obj_len, 1)

            # verify value in private metadata
            if isinstance(info, pd.Series):
                pd.testing.assert_series_equal(obj._metadata["label"], info[:obj_len])
            else:
                np.testing.assert_array_almost_equal(
                    obj._metadata["label"].values, info[:obj_len]
                )

            # verify public retrieval of metadata
            pd.testing.assert_series_equal(
                obj.get_info("label"), obj._metadata["label"]
            )
            pd.testing.assert_series_equal(obj.label, obj._metadata["label"])
            pd.testing.assert_series_equal(obj["label"], obj._metadata["label"])

        def test_add_metadata_key(self, obj, info, obj_len):
            # add metadata as key
            obj["label"] = info[:obj_len]

            # verify shape of metadata
            if isinstance(obj, nap.TsGroup):
                assert obj._metadata.shape == (obj_len, 2)
            else:
                assert obj._metadata.shape == (obj_len, 1)

            # verify value in private metadata
            if isinstance(info, pd.Series):
                pd.testing.assert_series_equal(obj._metadata["label"], info[:obj_len])
            else:
                np.testing.assert_array_almost_equal(
                    obj._metadata["label"].values, info[:obj_len]
                )

            # verify public retrieval of metadata
            pd.testing.assert_series_equal(
                obj.get_info("label"), obj._metadata["label"]
            )
            pd.testing.assert_series_equal(obj.label, obj._metadata["label"])
            pd.testing.assert_series_equal(obj["label"], obj._metadata["label"])

        def test_add_metadata_attr(self, obj, info, obj_len):
            # add metadata as attribute
            obj.label = info[:obj_len]

            # verify shape of metadata
            if isinstance(obj, nap.TsGroup):
                assert obj._metadata.shape == (obj_len, 2)
            else:
                assert obj._metadata.shape == (obj_len, 1)

            # verify value in private metadata
            if isinstance(info, pd.Series):
                pd.testing.assert_series_equal(obj._metadata["label"], info[:obj_len])
            else:
                np.testing.assert_array_almost_equal(
                    obj._metadata["label"].values, info[:obj_len]
                )

            # verify public retrieval of metadata
            pd.testing.assert_series_equal(
                obj.get_info("label"), obj._metadata["label"]
            )
            pd.testing.assert_series_equal(obj.label, obj._metadata["label"])
            pd.testing.assert_series_equal(obj["label"], obj._metadata["label"])

    def test_add_metadata_many(self, obj, obj_len):
        l1 = [1] * obj_len
        l2 = [2] * obj_len
        l3 = [3] * obj_len
        # add with set_info kwargs
        obj.set_info(l1=l1, l2=l2, l3=l3)

        # verify shape and value in private metadata
        if isinstance(obj, nap.TsGroup):
            assert obj._metadata.shape == (obj_len, 4), print(obj, obj._metadata)
        else:
            assert obj._metadata.shape == (obj_len, 3), print(obj._metadata)

        [
            np.testing.assert_array_almost_equal(obj._metadata[col].values, label)
            for col, label in zip(["l1", "l2", "l3"], [l1, l2, l3])
        ]

    @pytest.mark.parametrize(
        "info", [pd.DataFrame(index=[0, 1, 2, 3], data=[0, 0, 0, 0], columns=["label"])]
    )
    def test_add_metadata_df(self, obj, info, obj_len):
        # get proper length of metadata
        info = info.iloc[:obj_len]

        # add metadata with `set_info`
        obj.set_info(info)

        # verify shape and value in private metadata
        if isinstance(obj, nap.TsGroup):
            assert obj._metadata.shape == (obj_len, 2)
            pd.testing.assert_series_equal(obj._metadata["label"], info["label"])
        else:
            assert obj._metadata.shape == (obj_len, 1)
            pd.testing.assert_frame_equal(obj._metadata, info)

        # verify public retrieval of metadata
        pd.testing.assert_series_equal(obj.get_info("label"), obj._metadata["label"])
        pd.testing.assert_series_equal(obj.label, obj._metadata["label"])
        pd.testing.assert_series_equal(obj["label"], obj._metadata["label"])

    @pytest.mark.parametrize(
        "args, kwargs, expected",
        [
            (
                # invalid names as integers
                [pd.DataFrame(data=np.random.randint(0, 5, size=(4, 3)))],
                {},
                pytest.raises(TypeError, match="Invalid metadata type"),
            ),
            (
                # invalid names as strings starting with a number
                [
                    {"1": np.ones(4)},
                ],
                {},
                pytest.warns(UserWarning, match="starts with a number"),
            ),
            (
                # invalid names with spaces
                [
                    {"l 1": np.ones(4)},
                ],
                {},
                pytest.warns(UserWarning, match="contains a special character"),
            ),
            (
                # invalid names with periods
                [
                    {"1.1": np.ones(4)},
                ],
                {},
                pytest.warns(UserWarning, match="contains a special character"),
            ),
            (
                # metadata with wrong length
                [],
                {"label": np.zeros(100)},
                pytest.raises(
                    ValueError,
                    match="input array length 100 does not match",
                ),
            ),
        ],
    )
    def test_add_metadata_error(self, obj, obj_len, args, kwargs, expected):
        # trim to appropriate length
        if len(args):
            if isinstance(args[0], pd.DataFrame):
                metadata = args[0].iloc[:obj_len]
            elif isinstance(args[0], dict):
                metadata = {k: v[:obj_len] for k, v in args[0].items()}
        else:
            metadata = None
        with expected:
            obj.set_info(metadata, **kwargs)

    def test_add_metadata_key_error(self, obj, obj_len):
        # type specific key errors
        info = np.ones(obj_len)
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

    def test_overwrite_metadata(self, obj, obj_len):
        # add metadata
        obj.set_info(label=[1] * obj_len)
        assert np.all(obj.label == 1)

        obj.set_info(label=[2] * obj_len)
        assert np.all(obj.label == 2)

        obj["label"] = [3] * obj_len
        assert np.all(obj.label == 3)

        obj.label = [4] * obj_len
        assert np.all(obj.label == 4)

    # test naming overlap of shared attributes
    @pytest.mark.parametrize(
        "name",
        [
            "set_info",
            "_metadata",
            "_class_attributes",
        ],
    )
    def test_metadata_overlapping_names(self, obj, obj_len, name):
        values = np.ones(obj_len)

        # set some metadata to force assertion error with "_metadata" case
        obj.set_info(label=values)

        # assert attribute exists
        assert hasattr(obj, name)

        # warning when set
        with pytest.warns(UserWarning, match="overlaps with an existing"):
            obj.set_info({name: values})
        # error when set as attribute
        with pytest.raises(AttributeError, match="Cannot set attribute"):
            setattr(obj, name, values)
        # warning when set as key
        with pytest.warns(UserWarning, match="overlaps with an existing"):
            obj[name] = values
        # retrieve with get_info
        np.testing.assert_array_almost_equal(obj.get_info(name), values)
        # make sure it doesn't access metadata if its an existing attribute or key
        with pytest.raises((AssertionError, ValueError, TypeError)):
            np.testing.assert_array_almost_equal(getattr(obj, name), values)
        # access metadata as key
        np.testing.assert_array_almost_equal(obj[name], values)

    # test metadata that can only be accessed as key
    @pytest.mark.parametrize(
        "name",
        [
            "l.1",
            "l 1",
            "0",
        ],
    )
    def test_metadata_nonattribute_names(self, obj, obj_len, name):
        values = np.ones(obj_len)

        # set some metadata to force assertion error with "_metadata" case
        with pytest.warns(UserWarning, match="cannot be accessed as an attribute"):
            obj.set_info({name: values})

        # make sure it can be accessed with get_info
        np.testing.assert_array_almost_equal(obj.get_info(name), values)
        # make sure it can be accessed as key
        np.testing.assert_array_almost_equal(obj[name], values)

    @pytest.mark.parametrize("label, val", [([1, 1, 2, 2], 2)])
    def test_metadata_slicing(self, obj, label, val, obj_len):
        # slicing not relevant for length 1 objects
        if obj_len > 1:
            # add label
            obj.set_info(label=label, extra=[0, 1, 2, 3])

            # test slicing
            obj2 = obj[obj.label == val]
            assert isinstance(obj2, type(obj))
            assert np.all(obj2.label == val)
            if isinstance(obj, nap.IntervalSet):
                # interval set slicing resets index
                pd.testing.assert_frame_equal(
                    obj2._metadata,
                    obj._metadata[obj.label == val].reset_index(drop=True),
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

    def test_metadata_index_columns(self, obj, obj_len):
        # add metadata
        obj.set_info(one=[1] * obj_len, two=[2] * obj_len, three=[3] * obj_len)

        # test metadata columns
        assert np.all(obj["one"] == 1)
        assert np.all(obj[["two", "three"]] == obj._metadata[["two", "three"]])

    def test_save_and_load_npz(self, obj, obj_len):
        obj.set_info(label1=[1] * obj_len, label2=[2] * obj_len)

        obj.save("obj.npz")
        file = np.load("obj.npz", allow_pickle=True)

        # only test that metadata is saved correctly
        assert "_metadata" in file.keys()
        metadata = pd.DataFrame.from_dict(file["_metadata"].item())
        for k in ["label1", "label2"]:
            assert k in metadata.columns
            pd.testing.assert_series_equal(obj._metadata[k], metadata[k])

        # test pynapple loading
        obj2 = nap.load_file("obj.npz")
        assert isinstance(obj2, type(obj))
        pd.testing.assert_frame_equal(obj2._metadata, obj._metadata)

        # cleaning
        Path("obj.npz").unlink()


# test double inheritance
def get_defined_members(cls):
    """
    Get all methods and attributes explicitly defined in a class (excluding inherited ones),
    without relying on `__dir__` overrides.
    """
    # Fetch the class's dictionary directly to avoid `__dir__` overrides
    cls_dict = cls.__dict__

    # Use inspect to identify which are functions, properties, or other attributes
    return {
        name
        for name, obj in cls_dict.items()
        if not name.startswith("__")  # Ignore dunder methods
        and (
            inspect.isfunction(obj)
            or isinstance(obj, property)
            or not inspect.isroutine(obj)
        )
    }


def test_no_conflict_between_intervalset_and_metadatamixin():
    from pynapple.core import IntervalSet
    from pynapple.core.metadata_class import _MetadataMixin  # Adjust import as needed

    iset_members = get_defined_members(IntervalSet)
    metadatamixin_members = get_defined_members(_MetadataMixin)

    # Check for any overlapping names between IntervalSet and _MetadataMixin
    conflicting_members = iset_members.intersection(metadatamixin_members)

    # set_info and get_info will conflict
    assert len(conflicting_members) == 2, (
        f"Conflict detected! The following methods/attributes are "
        f"overwritten in IntervalSet: {conflicting_members}"
    )


def test_no_conflict_between_tsdframe_and_metadatamixin():
    from pynapple.core import TsdFrame
    from pynapple.core.metadata_class import _MetadataMixin  # Adjust import as needed

    tsdframe_members = get_defined_members(TsdFrame)
    metadatamixin_members = get_defined_members(_MetadataMixin)

    # Check for any overlapping names between TsdFrame and _MetadataMixin
    conflicting_members = tsdframe_members.intersection(metadatamixin_members)

    # set_info and get_info will conflict
    assert len(conflicting_members) == 2, (
        f"Conflict detected! The following methods/attributes are "
        f"overwritten in TsdFrame: {conflicting_members}"
    )


def test_no_conflict_between_tsgroup_and_metadatamixin():
    from pynapple.core import TsGroup
    from pynapple.core.metadata_class import _MetadataMixin  # Adjust import as needed

    tsgroup_members = get_defined_members(TsGroup)
    metadatamixin_members = get_defined_members(_MetadataMixin)

    # Check for any overlapping names between TsdFrame and _MetadataMixin
    conflicting_members = tsgroup_members.intersection(metadatamixin_members)

    # set_info and get_info will conflict
    assert len(conflicting_members) == 2, (
        f"Conflict detected! The following methods/attributes are "
        f"overwritten in TsGroup: {conflicting_members}"
    )
