"""Newaxis indexing must track the time axis independently of its length."""

import numpy as np
import pytest

import pynapple as nap


@pytest.fixture(params=[0, 1, 3], ids=["empty", "singleton", "multiple"])
def n_times(request):
    return request.param


@pytest.fixture(
    params=[(nap.Tsd, ()), (nap.TsdFrame, (2,)), (nap.TsdTensor, (2, 3))],
    ids=["Tsd", "TsdFrame", "TsdTensor"],
)
def series(request, n_times):
    cls, tail = request.param
    times = 10 + 10 * np.arange(n_times, dtype=float)
    data = np.arange(n_times * int(np.prod(tail)), dtype=np.int16).reshape(
        (n_times,) + tail
    )
    support = nap.IntervalSet(0, 50)
    kwargs = {"metadata": {"label": ["left", "right"]}} if cls is nap.TsdFrame else {}
    obj = cls(t=times, d=data, time_support=support, **kwargs)
    yield obj
    np.testing.assert_array_equal(obj.values, data)
    np.testing.assert_array_equal(obj.t, times)
    np.testing.assert_array_equal(obj.time_support.values, support.values)
    if cls is nap.TsdFrame:
        assert obj.get_info("label").tolist() == ["left", "right"]


def assert_native_result(result, expected):
    assert type(result) is type(expected)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


def assert_time_result(result, expected, times, support):
    cls = {1: nap.Tsd, 2: nap.TsdFrame}.get(expected.ndim, nap.TsdTensor)
    assert isinstance(result, cls)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result.values, expected)
    np.testing.assert_array_equal(result.t, times)
    np.testing.assert_array_equal(result.time_support.values, support.values)


@pytest.mark.parametrize(
    "key", [None, (None,), (None, Ellipsis), (None, Ellipsis, None)]
)
def test_leading_newaxis_returns_native_array(series, key):
    # The inserted axis is not time, even when there is only one timestamp.
    assert_native_result(series[key], series.values[key])


@pytest.mark.parametrize(
    "key, time_key",
    [
        ((slice(None), None), slice(None)),
        ((Ellipsis, None), slice(None)),
        ((slice(0, 1), Ellipsis, None), slice(0, 1)),
        ((slice(0, 0), None), slice(0, 0)),
    ],
)
def test_newaxis_after_time_preserves_timestamps(series, key, time_key):
    assert_time_result(
        series[key], series.values[key], series.t[time_key], series.time_support
    )


@pytest.mark.parametrize("leading", [False, True])
@pytest.mark.parametrize("kind", ["boolean", "integer", "boolean_tsd"])
def test_time_selection_with_newaxis(series, leading, kind):
    mask = np.arange(len(series)) % 2 == 0
    native_selector = np.flatnonzero(mask) if kind == "integer" else mask
    selector = native_selector
    if kind == "boolean_tsd":
        selector = nap.Tsd(t=series.t, d=mask, time_support=series.time_support)
    key = (None, selector) if leading else (selector, None)
    native_key = (None, native_selector) if leading else (native_selector, None)
    expected = series.values[native_key]
    result = series[key]
    if leading:
        assert_native_result(result, expected)
    else:
        assert_time_result(
            result, expected, series.t[native_selector], series.time_support
        )


@pytest.mark.parametrize("key", [(0, None), (0, Ellipsis, None)])
def test_integer_time_selection_does_not_create_a_new_time_axis(series, key):
    if len(series) == 0:
        with pytest.raises(IndexError):
            series[key]
    else:
        # Integer indexing removes time. A new length-one axis cannot replace it.
        assert_native_result(series[key], series.values[key])


def test_empty_list_with_newaxis(series):
    key = ([], None)
    assert_time_result(
        series[key], series.values[key], series.t[:0], series.time_support
    )


def test_reverse_time_slice_with_newaxis(series):
    key = (slice(None, None, -1), None)
    assert_time_result(
        series[key], series.values[key], series.t[::-1], series.time_support
    )


@pytest.mark.parametrize("n_columns", [1, 2])
@pytest.mark.parametrize("position", ["before", "after", "ellipsis"])
def test_new_column_axis_drops_original_column_metadata(n_times, n_columns, position):
    data = np.arange(n_times * n_columns).reshape(n_times, n_columns)
    labels = ["left", "right"][:n_columns]
    frame = nap.TsdFrame(
        t=10 + 10 * np.arange(n_times),
        d=data,
        time_support=nap.IntervalSet(0, 50),
        metadata={"label": labels},
    )
    column = n_columns - 1
    key = {
        "before": (slice(None), None, column),
        "after": (slice(None), column, None),
        "ellipsis": (Ellipsis, None, column),
    }[position]
    result = frame[key]
    assert_time_result(result, frame.values[key], frame.t, frame.time_support)
    assert isinstance(result, nap.TsdFrame)
    assert result.metadata_columns == []
    np.testing.assert_array_equal(result.columns, [0])
    # Identical old/new column counts must not cause metadata to be reattached.
    assert frame.get_info("label").tolist() == labels


@pytest.mark.parametrize(
    "key, row, col",
    [
        (Ellipsis, slice(None), slice(None)),
        ((Ellipsis,), slice(None), slice(None)),
        ((), slice(None), slice(None)),
        ((slice(None), Ellipsis), slice(None), slice(None)),
        ((Ellipsis, [1, 0]), slice(None), [1, 0]),
        (([2, 0], Ellipsis), [2, 0], slice(None)),
        ((slice(None), Ellipsis, []), slice(None), []),
        (([], Ellipsis), [], slice(None)),
        (([False, True, False], Ellipsis), [False, True, False], slice(None)),
        ((Ellipsis, [True, False]), slice(None), [True, False]),
    ],
)
def test_ellipsis_preserves_existing_frame_column_metadata(key, row, col):
    frame = nap.TsdFrame(
        t=[10, 20, 30],
        d=np.arange(6).reshape(3, 2),
        columns=["a", "b"],
        time_support=nap.IntervalSet(0, 50),
        metadata={"label": ["left", "right"]},
    )
    result = frame[key]
    assert_time_result(result, frame.values[key], frame.t[row], frame.time_support)
    np.testing.assert_array_equal(result.columns, frame.columns[col])
    assert result.metadata_columns == frame.metadata_columns
    np.testing.assert_array_equal(
        result.get_info("label"), frame.get_info("label").iloc[col]
    )


def test_original_issue_newaxis_examples():
    frame = nap.TsdFrame(t=np.arange(10), d=np.ones((10, 2)))
    assert_time_result(
        frame[:, None], frame.values[:, None], frame.t, frame.time_support
    )
    tensor = nap.TsdTensor(t=np.arange(10), d=np.arange(20).reshape(10, 1, 2))
    assert_native_result(tensor[None], tensor.values[None])


def test_ellipsis_preserves_tuple_column_labels_and_metadata():
    frame = nap.TsdFrame(
        t=[10, 20, 30],
        d=np.arange(6).reshape(3, 2),
        columns=[("ca1", 1), ("ca1", 2)],
        metadata={"quality": [0.1, 0.2]},
    )
    result = frame[..., [1, 0]]
    assert_time_result(result, frame.values[..., [1, 0]], frame.t, frame.time_support)
    assert list(result.columns) == [("ca1", 2), ("ca1", 1)]
    np.testing.assert_array_equal(result.get_info("quality"), [0.2, 0.1])
    # Selecting one tuple-labelled column still returns a scalar time series.
    assert_time_result(frame[..., 1], frame.values[..., 1], frame.t, frame.time_support)


def test_newaxis_preserves_native_invalid_index_errors(series):
    with pytest.raises(IndexError):
        series[None, ..., ...]
    with pytest.raises(IndexError):
        series[(slice(None),) * (series.ndim + 1) + (None,)]


def test_newaxis_rejects_non_boolean_time_series_selector():
    data = nap.Tsd(t=[10, 20], d=[1.0, 2.0])
    selector = nap.Tsd(t=data.t, d=[0, 1])
    with pytest.raises(ValueError, match="indices must be boolean"):
        data[selector, None]


@pytest.mark.parametrize(
    "cls, shape",
    [(nap.Tsd, (3,)), (nap.TsdFrame, (1, 2)), (nap.TsdTensor, (3, 1, 2))],
)
@pytest.mark.parametrize(
    "key",
    [
        True,
        False,
        (True,),
        (False,),
        np.bool_(True),
        (np.bool_(False),),
        np.array(True),
        (np.array(False),),
    ],
)
def test_boolean_scalar_indexing_returns_native_array(cls, shape, key):
    data = np.arange(np.prod(shape)).reshape(shape)
    obj = cls(
        t=10 + 10 * np.arange(shape[0]), d=data, time_support=nap.IntervalSet(0, 50)
    )
    # A Boolean scalar adds a selection axis before time; it does not select a row.
    assert_native_result(obj[key], obj.values[key])
    np.testing.assert_array_equal(obj.values, data)


@pytest.mark.parametrize(
    "cls, shape", [(nap.TsdFrame, (3, 2)), (nap.TsdTensor, (3, 1, 2))]
)
@pytest.mark.parametrize("as_tuple", [False, True])
@pytest.mark.parametrize("select_any", [False, True])
@pytest.mark.parametrize("as_list", [False, True])
def test_full_data_boolean_mask_returns_native_array(
    cls, shape, as_tuple, select_any, as_list
):
    data = np.arange(np.prod(shape)).reshape(shape)
    obj = cls(
        t=10 + 10 * np.arange(shape[0]), d=data, time_support=nap.IntervalSet(0, 50)
    )
    mask = (data % 2 == 0) if select_any else np.zeros(shape, dtype=bool)
    if as_list:
        mask = mask.tolist()
    key = (mask,) if as_tuple else mask
    # Masking all dimensions flattens values and removes the original time axis.
    assert_native_result(obj[key], obj.values[key])
    np.testing.assert_array_equal(obj.values, data)


@pytest.mark.parametrize("count", [0, 3])
@pytest.mark.parametrize("key", [[], np.array([], dtype=int), np.array([], dtype=bool)])
def test_empty_time_selector_preserves_frame_columns_and_metadata(count, key):
    frame = nap.TsdFrame(
        t=10 + 10 * np.arange(count),
        d=np.arange(count * 2).reshape(count, 2),
        columns=["left", "right"],
        time_support=nap.IntervalSet(0, 50),
        metadata={"quality": [0.1, 0.2]},
    )
    result = frame[key]
    assert_time_result(result, frame.values[key], frame.t[:0], frame.time_support)
    np.testing.assert_array_equal(result.columns, frame.columns)
    assert result.metadata_columns == frame.metadata_columns
    np.testing.assert_array_equal(result.get_info("quality"), [0.1, 0.2])


def test_empty_boolean_time_series_selector_preserves_frame_columns():
    support = nap.IntervalSet(0, 50)
    frame = nap.TsdFrame(
        t=[], d=np.empty((0, 2)), columns=["a", "b"], time_support=support
    )
    mask = nap.Tsd(t=[], d=np.array([], dtype=bool), time_support=support)
    result = frame[mask]
    assert_time_result(result, frame.values[mask.values], frame.t, support)
    np.testing.assert_array_equal(result.columns, frame.columns)


def test_boolean_frame_mask_of_tensor_returns_native_array():
    tensor = nap.TsdTensor(t=[10, 20, 30], d=np.arange(12).reshape(3, 2, 2))
    mask = nap.TsdFrame(t=tensor.t, d=np.array([[True, False]] * 3))
    assert_native_result(tensor[mask], tensor.values[mask.values])


@pytest.mark.parametrize("count", [0, 1, 3])
def test_empty_boolean_column_selector_preserves_empty_metadata(count):
    frame = nap.TsdFrame(
        t=10 + 10 * np.arange(count),
        d=np.arange(count * 2).reshape(count, 2),
        columns=["a", "b"],
        time_support=nap.IntervalSet(0, 50),
        metadata={"quality": [0.1, 0.2]},
    )
    key = (slice(None), np.array([], dtype=bool))
    result = frame[key]
    assert_time_result(result, frame.values[key], frame.t, frame.time_support)
    assert len(result.columns) == 0
    assert result.metadata_columns == ["quality"]
    assert len(result.get_info("quality")) == 0


@pytest.mark.parametrize("key", [[[True, False], [True]], [[0, 1], [2]]])
@pytest.mark.parametrize("as_tuple", [False, True])
def test_ragged_sequence_index_preserves_native_error(key, as_tuple):
    frame = nap.TsdFrame(t=[10, 20, 30], d=np.arange(6).reshape(3, 2))
    key = (key,) if as_tuple else key
    with pytest.raises((IndexError, TypeError, ValueError)) as native_error:
        frame.values[key]
    with pytest.raises(type(native_error.value)):
        frame[key]
