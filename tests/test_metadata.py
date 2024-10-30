"""Tests for metadata in IntervalSet, TsdFrame, and TsGroup"""

from numbers import Number
import inspect


import pickle
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from contextlib import nullcontext as does_not_raise
import warnings

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
    label = ["a", "b"]
    metadata = {"label": label}
    ep = nap.IntervalSet(start=start, end=end, metadata=metadata)
    assert ep._metadata["label"][0] == label


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
    ep4 = nap.IntervalSet(start=50, end=60, metadata={"m2": 3})
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
    "args, kwargs, expected",
    [
        (
            # invalid metadata names that are the same as column names
            [
                pd.DataFrame(
                    index=["a", "b", "c"],
                    columns=["a", "b", "c"],
                    data=np.random.randint(0, 5, size=(3, 3)),
                )
            ],
            {},
            pytest.raises(ValueError, match="Invalid metadata name"),
        ),
    ],
)
def test_tsdframe_add_metadata_error(tsdframe_meta, args, kwargs, expected):
    with expected:
        tsdframe_meta.set_info(*args, **kwargs)


#############
## TsGroup ##
#############


##################
## Shared tests ##
##################
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
    class Test_Add_Metadata:
        def test_add_metadata(self, obj, info):
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
                np.testing.assert_array_almost_equal(
                    obj._metadata["label"].values, info
                )

            # verify public retrieval of metadata
            pd.testing.assert_series_equal(
                obj.get_info("label"), obj._metadata["label"]
            )
            pd.testing.assert_series_equal(obj.label, obj._metadata["label"])
            pd.testing.assert_series_equal(obj["label"], obj._metadata["label"])

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
                np.testing.assert_array_almost_equal(
                    obj._metadata["label"].values, info
                )

            # verify public retrieval of metadata
            pd.testing.assert_series_equal(
                obj.get_info("label"), obj._metadata["label"]
            )
            pd.testing.assert_series_equal(obj.label, obj._metadata["label"])
            pd.testing.assert_series_equal(obj["label"], obj._metadata["label"])

        def test_add_metadata_attr(self, obj, info):
            # add metadata as attribute
            obj.label = info

            # verify shape of metadata
            if isinstance(obj, nap.TsGroup):
                assert obj._metadata.shape == (4, 2)
            else:
                assert obj._metadata.shape == (4, 1)

            # verify value in private metadata
            if isinstance(info, pd.Series):
                pd.testing.assert_series_equal(obj._metadata["label"], info)
            else:
                np.testing.assert_array_almost_equal(
                    obj._metadata["label"].values, info
                )

            # verify public retrieval of metadata
            pd.testing.assert_series_equal(
                obj.get_info("label"), obj._metadata["label"]
            )
            pd.testing.assert_series_equal(obj.label, obj._metadata["label"])
            pd.testing.assert_series_equal(obj["label"], obj._metadata["label"])

    @pytest.mark.parametrize(
        "info", [pd.DataFrame(index=[0, 1, 2, 3], data=[0, 0, 0, 0], columns=["label"])]
    )
    def test_add_metadata_df(self, obj, info):
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
                pytest.raises(
                    ValueError,
                    match="input array length 5 does not match metadata length 4.",
                ),
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

    def test_overwrite_metadata(self, obj):
        # add metadata
        obj.set_info(label=[1, 1, 1, 1])
        assert np.all(obj.label == 1)

        obj.set_info(label=[2, 2, 2, 2])
        assert np.all(obj.label == 2)

        obj["label"] = [3, 3, 3, 3]
        assert np.all(obj.label == 3)

        obj.label = [4, 4, 4, 4]
        assert np.all(obj.label == 4)

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

    def test_metadata_index_columns(self, obj):
        # add metadata
        obj.set_info(one=[1, 1, 1, 1], two=[2, 2, 2, 2], three=[3, 3, 3, 3])

        # test metadata columns
        assert np.all(obj["one"] == 1)
        assert np.all(obj[["two", "three"]] == obj._metadata[["two", "three"]])

    def test_save_and_load_npz(self, obj):
        obj.set_info(label1=[1, 1, 2, 2], label2=[0, 1, 2, 3])

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


def test_no_conflict_between_intervalset_and_metadatabase():
    from pynapple.core import IntervalSet
    from pynapple.core.metadata_class import _MetadataBase  # Adjust import as needed

    iset_members = get_defined_members(IntervalSet)
    metadatabase_members = get_defined_members(_MetadataBase)

    # Check for any overlapping names between IntervalSet and _MetadataBase
    conflicting_members = iset_members.intersection(metadatabase_members)

    assert len(conflicting_members) == 0, (
        f"Conflict detected! The following methods/attributes are "
        f"overwritten in TsdFrame: {conflicting_members}"
    )


def test_no_conflict_between_tsdframe_and_metadatabase():
    from pynapple.core import TsdFrame
    from pynapple.core.metadata_class import _MetadataBase  # Adjust import as needed

    tsdframe_members = get_defined_members(TsdFrame)
    metadatabase_members = get_defined_members(_MetadataBase)

    # Check for any overlapping names between TsdFrame and _MetadataBase
    conflicting_members = tsdframe_members.intersection(metadatabase_members)

    assert len(conflicting_members) == 0, (
        f"Conflict detected! The following methods/attributes are "
        f"overwritten in TsdFrame: {conflicting_members}"
    )


def test_no_conflict_between_tsgroup_and_metadatabase():
    from pynapple.core import TsGroup
    from pynapple.core.metadata_class import _MetadataBase  # Adjust import as needed

    tsgroup_members = get_defined_members(TsGroup)
    metadatabase_members = get_defined_members(_MetadataBase)

    # Check for any overlapping names between TsdFrame and _MetadataBase
    conflicting_members = tsgroup_members.intersection(metadatabase_members)

    assert len(conflicting_members) == 0, (
        f"Conflict detected! The following methods/attributes are "
        f"overwritten in TsdFrame: {conflicting_members}"
    )
