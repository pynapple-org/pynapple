"""Tests of LazyTsGroup.

A lazy group must give the same results as a regular TsGroup with the same
spikes, with each source and each storage (numpy, h5py, zarr). It must read
only the spikes that an operation needs, and it must not keep them.
"""

import pickle
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

import pynapple as nap
from pynapple.core.lazy_ts_group import (
    LazyTsGroup,
    _RaggedArraySource,
    _SortedArraySource,
    _windows,
)


def make_spikes():
    """Make spikes with the difficult cases:

    - keys in an unsorted order
    - an empty unit
    - a unit with one spike outside the spans of the other units
    - equal times in two units
    """
    rng = np.random.default_rng(0)
    spikes = {
        7: np.sort(rng.uniform(0, 10, 200)),
        2: np.sort(rng.uniform(5, 20, 300)),
        4: np.sort(np.concatenate([rng.uniform(1, 3, 50), [7.0, 7.0]])),
        9: np.array([]),
        3: np.array([50.0]),
    }
    spikes[2][10] = 7.0
    spikes[2] = np.sort(spikes[2])
    return spikes


SPIKES = make_spikes()
QUALITY = {k: f"q{i}" for i, k in enumerate(SPIKES)}
N_TOTAL = sum(len(t) for t in SPIKES.values())

EP = nap.IntervalSet(start=[2, 6], end=[4, 8])
TSD = nap.Tsd(t=np.linspace(0, 20, 500), d=np.arange(500.0))
EVENT = nap.Ts(t=np.linspace(1, 19, 30))


def make_eager(spikes):
    with warnings.catch_warnings():
        # empty and single-spike units give warnings about their time support
        warnings.simplefilter("ignore")
        return nap.TsGroup(
            {k: nap.Ts(t) for k, t in spikes.items()},
            metadata={"quality": [QUALITY[k] for k in spikes]},
        )


#################################
# Storage and sources
#################################


class CountingArray:
    """Array wrapper that counts the values that each read returns."""

    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.dtype = array.dtype
        self.n_read = 0

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        out = np.asarray(self.array[index])
        self.n_read += out.size
        return out


@pytest.fixture(params=["numpy", "h5py", "zarr"])
def store(request, tmp_path):
    """A function that writes an array to the storage and gives the stored
    array back."""
    if request.param == "numpy":
        yield lambda name, array: np.asarray(array)
        return

    if request.param == "h5py":
        files = []

        def store_h5py(name, array):
            # One file for each array: HDF5 cannot write a file that is open
            # for a read.
            path = tmp_path / f"{name}.h5"
            with h5py.File(path, "w") as f:
                f.create_dataset(name, data=np.asarray(array))
            files.append(h5py.File(path, "r"))
            return files[-1][name]

        yield store_h5py
        for f in files:
            f.close()
        return

    zarr = pytest.importorskip("zarr")

    def store_zarr(name, array):
        array = np.asarray(array)
        path = str(tmp_path / f"{name}.zarr")
        z = zarr.open_array(
            path, mode="w", shape=array.shape, dtype=array.dtype, chunks=(64,)
        )
        z[:] = array
        return zarr.open_array(path, mode="r")

    yield store_zarr


def ragged_source(spikes, store):
    """A ragged source: the units in the order of `spikes`, one after the
    other."""
    keys = list(spikes)
    flat = np.concatenate([spikes[k] for k in keys])
    index = np.cumsum([len(spikes[k]) for k in keys])
    return _RaggedArraySource(store("ragged_array", flat), index, keys)


def sorted_source(spikes, store, chunk_size=64):
    """A sorted source: all the spikes sorted by time, then by key. The small
    chunks make the scans of ``clusters`` read several chunks."""
    keys = sorted(spikes)
    times = np.concatenate([spikes[k] for k in keys])
    clusters = np.repeat(keys, [len(spikes[k]) for k in keys])
    order = np.argsort(times, kind="stable")
    return _SortedArraySource(
        store("times", times[order]),
        store("clusters", clusters[order]),
        keys=list(spikes),
        chunk_size=chunk_size,
    )


SOURCES = {"ragged": ragged_source, "sorted": sorted_source}


@pytest.fixture(params=list(SOURCES))
def make_lazy(request, store):
    """A function that makes a lazy group from a dict of spikes."""

    def make(spikes, wrap=None):
        stored = store if wrap is None else (lambda n, a: wrap(n, store(n, a)))
        source = SOURCES[request.param](spikes, stored)
        metadata = {"quality": [QUALITY[k] for k in source.keys]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LazyTsGroup(source, metadata=metadata)

    return make


#################################
# Same results as a regular TsGroup
#################################


def assert_same(a, b):
    """Check that two results of an operation are equal."""
    if isinstance(a, Exception) or isinstance(b, Exception):
        assert type(a) is type(b), (a, b)
        assert str(a) == str(b)
        return
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, nap.TsGroup):
        assert a == b
        np.testing.assert_array_equal(a.rates, b.rates)
    elif isinstance(a, nap.Ts):
        np.testing.assert_array_equal(a.t, b.t)
        np.testing.assert_array_equal(a.time_support.values, b.time_support.values)
    elif isinstance(a, (nap.Tsd, nap.TsdFrame, nap.TsdTensor)):
        np.testing.assert_array_equal(a.t, b.t)
        np.testing.assert_array_equal(a.values, b.values)
        np.testing.assert_array_equal(a.time_support.values, b.time_support.values)
    elif isinstance(a, dict):
        assert list(a) == list(b)
        for k in a:
            assert_same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    elif isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b)
    elif hasattr(a, "values") and hasattr(a, "coords"):  # xarray
        np.testing.assert_array_equal(a.values, b.values)
    else:
        np.testing.assert_array_equal(a, b)


def run(operation, group):
    """Run the operation, and give its exception as the result if it fails."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return operation(group)
    except Exception as err:
        return err


def other_group(g):
    return nap.TsGroup({100: nap.Ts(t=[1.0, 2.0])}, time_support=g.time_support)


OPERATIONS = {
    # metadata only
    "index": lambda g: g.index,
    "keys": lambda g: g.keys(),
    "len": lambda g: len(g),
    "contains": lambda g: (7 in g, 8 in g),
    "rates": lambda g: g.rates,
    "time_support": lambda g: g.time_support.values,
    "metadata": lambda g: g.quality.values,
    "repr": lambda g: repr(g),
    "repr(data)": lambda g: repr(g.data),
    # selection of units
    "g[k]": lambda g: g[7],
    "g[empty unit]": lambda g: g[9],
    "g[keys]": lambda g: g[[2, 4]],
    "g[bool]": lambda g: g[g.rates > 5],
    "g[missing key]": lambda g: g[[2, 8]],
    "getby_threshold": lambda g: g.getby_threshold("rate", 5),
    "getby_category": lambda g: g.getby_category("quality"),
    "values": lambda g: g.values(),
    "items": lambda g: g.items(),
    "dict(data)": lambda g: dict(g.data),
    # selection of time
    "restrict": lambda g: g.restrict(EP),
    "restrict(empty)": lambda g: g.restrict(nap.IntervalSet(100, 200)),
    "restrict(bad)": lambda g: g.restrict([0, 1]),
    "get(start, end)": lambda g: g.get(3, 5),
    "get(ms)": lambda g: g.get(3000, 5000, "ms"),
    "get(t)": lambda g: g.get(7.0),
    "get(bad)": lambda g: g.get("a", 5),
    "count(bin, ep)": lambda g: g.count(0.5, EP),
    "count(ep)": lambda g: g.count(ep=EP),
    "count(bad ep)": lambda g: g.count(0.5, [0, 1]),
    "value_from(ep)": lambda g: g.value_from(TSD, EP),
    "value_from": lambda g: g.value_from(TSD),
    "time_diff(epochs)": lambda g: g.time_diff(epochs=EP),
    "trial_count": lambda g: g.trial_count(EP, 0.5),
    # operations on all the spikes
    "count(bin)": lambda g: g.count(0.5),
    "time_diff": lambda g: g.time_diff(),
    "to_tsd": lambda g: g.to_tsd(),
    "to_tsd(metadata)": lambda g: g.to_tsd("rate"),
    "subsample": lambda g: g.subsample(0.5, seed=1),
    "copy": lambda g: g.copy(),
    "pickle": lambda g: pickle.loads(pickle.dumps(g)),
    "merge": lambda g: g.merge(other_group(g), ignore_metadata=True),
    "merge_group": lambda g: nap.TsGroup.merge_group(
        g, other_group(g), ignore_metadata=True
    ),
    # process functions that read the merged arrays directly
    "tuning_curves": lambda g: nap.compute_tuning_curves(g, TSD, bins=5),
    "tuning_curves(epochs)": lambda g: nap.compute_tuning_curves(
        g, TSD, bins=5, epochs=EP
    ),
    "autocorrelogram": lambda g: nap.compute_autocorrelogram(g, 0.1, 1.0),
    "crosscorrelogram(ep)": lambda g: nap.compute_crosscorrelogram(g, 0.1, 1.0, ep=EP),
    "eventcorrelogram": lambda g: nap.compute_eventcorrelogram(g, EVENT, 0.1, 1.0),
}


@pytest.mark.parametrize("name", list(OPERATIONS))
def test_same_as_eager(make_lazy, name):
    operation = OPERATIONS[name]
    lazy = make_lazy(SPIKES)
    assert type(lazy) is LazyTsGroup
    assert_same(run(operation, lazy), run(operation, make_eager(SPIKES)))


def test_equal_to_eager(make_lazy):
    lazy = make_lazy(SPIKES)
    eager = make_eager(SPIKES)
    assert lazy == eager
    assert eager == lazy
    assert lazy != eager[[2, 4]]


def test_get_closest(make_lazy):
    """``get(t)`` reads the spikes around ``t`` in each unit. The data have no
    empty unit, because ``Ts.get(t)`` fails on an empty Ts. Unit 3 is also
    empty in the group: its only spike is outside the time support."""
    spikes = {k: t for k, t in SPIKES.items() if k not in (9, 3)}
    lazy, eager = make_lazy(spikes), make_eager(spikes)
    for t in [0.0, 2.0, 7.0, 7.05, 19.0, 60.0]:
        assert_same(lazy.get(t), eager.get(t))


#################################
# Read only what the operation needs
#################################


def n_spikes_in(ep):
    """The number of spikes in the epochs ``ep``."""
    return sum(
        np.sum((t >= s) & (t <= e)) for t in SPIKES.values() for s, e in ep.values
    )


# Operations with epochs, and the time windows that they read.
OPERATIONS_WITH_EPOCHS = [
    (lambda g: g.restrict(EP), EP),
    (lambda g: g.count(0.5, EP), EP),
    (lambda g: g.get(2, 8), nap.IntervalSet(2, 8)),
    (lambda g: nap.compute_tuning_curves(g, TSD, bins=5, epochs=EP), EP),
]


@pytest.fixture
def counted(make_lazy):
    """A lazy group whose stored arrays count their reads.

    Returns
    -------
    lazy : LazyTsGroup
    construction : dict
        The number of values that the construction read from each array.
    n_read : function
        Gives the number of values read from each array since the last call.
    """
    counters = {}

    def wrap(name, array):
        counters[name] = CountingArray(array)
        return counters[name]

    lazy = make_lazy(SPIKES, wrap=wrap)
    construction = {name: c.n_read for name, c in counters.items()}
    last = dict(construction)

    def n_read():
        new = {name: c.n_read - last[name] for name, c in counters.items()}
        last.update({name: c.n_read for name, c in counters.items()})
        return new

    return lazy, construction, n_read


def test_ragged_reads(counted, request):
    if request.node.callspec.params["make_lazy"] != "ragged":
        pytest.skip("ragged source only")
    lazy, construction, n_read_arrays = counted
    eager = make_eager(SPIKES)

    def n_read():
        return n_read_arrays()["ragged_array"]

    # construction: the first and the last spike of each unit that is not empty
    assert construction == {
        "ragged_array": 2 * sum(len(t) > 0 for t in SPIKES.values())
    }

    # metadata: no spike read
    lazy.rates, lazy.index, lazy.time_support, repr(lazy), repr(lazy.data)
    lazy.getby_threshold("rate", 1e9)
    assert n_read() == 0

    # units: only the selected units
    lazy[[2, 4]]
    assert n_read() == len(SPIKES[2]) + len(SPIKES[4])
    lazy[7]
    assert n_read() == len(SPIKES[7])

    # time: the spikes in each epoch, plus a binary search in each unit that
    # is only partly in the epoch
    search = 2 * 3 * int(np.ceil(np.log2(N_TOTAL) + 1))
    for operation, ep in OPERATIONS_WITH_EPOCHS:
        assert_same(operation(lazy), operation(eager))
        in_ep = n_spikes_in(ep)
        assert in_ep <= n_read() <= in_ep + len(ep) * search
    lazy.restrict(nap.IntervalSet(100, 200))
    assert n_read() == 0

    # closest spike: 2 spikes in each unit, plus one binary search in each unit
    run(lambda g: g.get(7.0), lazy)
    n_units = len(SPIKES)
    assert n_read() <= 2 * n_units + n_units * int(np.ceil(np.log2(N_TOTAL) + 1))

    # all the spikes: the group keeps nothing, so each call reads them again
    lazy.count(1.0)
    lazy.count(1.0)
    assert n_read() == 2 * N_TOTAL
    assert not any(
        isinstance(v, np.ndarray) and len(v) == N_TOTAL for v in lazy.__dict__.values()
    )


def test_sorted_reads(counted, request):
    if request.node.callspec.params["make_lazy"] != "sorted":
        pytest.skip("sorted source only")
    lazy, construction, n_read = counted
    eager = make_eager(SPIKES)
    n_units = len(SPIKES)
    search = 2 * int(np.ceil(np.log2(N_TOTAL) + 1))

    # construction: all of clusters, and the times of the first and last
    # spike of each unit
    assert construction["clusters"] == N_TOTAL
    assert construction["times"] <= 2 * n_units

    # metadata: no spike read
    lazy.rates, lazy.index, lazy.time_support, repr(lazy), repr(lazy.data)
    lazy.getby_threshold("rate", 1e9)
    assert n_read() == {"times": 0, "clusters": 0}

    # units: all of clusters, and times only in the chunks with selected spikes
    lazy[[2, 4]]
    reads = n_read()
    assert reads["clusters"] == N_TOTAL and reads["times"] <= N_TOTAL
    lazy[3]  # one spike, in the last chunk
    reads = n_read()
    assert reads["clusters"] == N_TOTAL and reads["times"] <= 64

    # time: one slice of the spikes in each epoch, plus two binary searches
    for operation, ep in OPERATIONS_WITH_EPOCHS:
        assert_same(operation(lazy), operation(eager))
        in_ep = n_spikes_in(ep)
        reads = n_read()
        assert reads["clusters"] == in_ep
        assert in_ep <= reads["times"] <= in_ep + len(ep) * search
    lazy.restrict(nap.IntervalSet(100, 200))
    reads = n_read()
    assert reads["clusters"] == 0 and reads["times"] <= search

    # closest spike: the times of 2 spikes in each unit, plus one binary search
    run(lambda g: g.get(7.0), lazy)
    assert n_read()["times"] <= 2 * n_units + search

    # all the spikes: the group keeps nothing, so each call reads them again
    lazy.count(1.0)
    lazy.count(1.0)
    assert n_read() == {"times": 2 * N_TOTAL, "clusters": 2 * N_TOTAL}
    assert not any(
        isinstance(v, np.ndarray) and len(v) == N_TOTAL for v in lazy.__dict__.values()
    )



@pytest.mark.parametrize(
    "starts, ends, max_windows, expected",
    [
        # few intervals: one window for each interval
        ([0, 5, 9], [1, 6, 10], 16, ([0, 5, 9], [1, 6, 10])),
        # too many intervals: keep the largest gap
        ([0, 2, 9], [1, 3, 10], 2, ([0, 9], [3, 10])),
        # one window: the span
        ([0, 2, 9], [1, 3, 10], 1, ([0], [10])),
        # touching intervals: always merged
        ([0, 1, 5], [1, 2, 6], 16, ([0, 5], [2, 6])),
        ([0], [1], 16, ([0], [1])),
    ],
)
def test_windows(starts, ends, max_windows, expected):
    s, e = _windows(np.array(starts, float), np.array(ends, float), max_windows)
    np.testing.assert_array_equal(s, expected[0])
    np.testing.assert_array_equal(e, expected[1])


def test_fragmented_epochs(counted, request):
    """With many short epochs over a long time, the read skips the gaps,
    up to ``max_windows`` windows."""
    lazy, _, n_read = counted
    eager = make_eager(SPIKES)
    starts = np.arange(0, 20, 1.0)
    ep = nap.IntervalSet(starts, starts + 0.1)
    # binary search reads for each window: in each unit for a ragged source,
    # in `times` for a sorted source
    n_searches = 2 * (len(SPIKES) if "ragged" in request.node.name else 1)
    search = n_searches * int(np.ceil(np.log2(N_TOTAL) + 1))

    def n_read_spikes():
        reads = n_read()
        return reads.get("ragged_array", reads.get("times"))

    for max_windows in [1, 4, 256]:
        lazy._source.max_windows = max_windows
        windows = nap.IntervalSet(*_windows(ep.start, ep.end, max_windows))
        assert len(windows) == min(max_windows, len(ep))
        assert_same(lazy.restrict(ep), eager.restrict(ep))
        in_windows = n_spikes_in(windows)
        assert in_windows <= n_read_spikes() <= in_windows + len(windows) * search

    # the windows skip the largest gaps
    assert n_spikes_in(ep) < n_spikes_in(nap.IntervalSet(*_windows(ep.start, ep.end, 4)))
    assert n_spikes_in(nap.IntervalSet(*_windows(ep.start, ep.end, 4))) < n_spikes_in(
        nap.IntervalSet(0, 19.1)
    )


@pytest.mark.parametrize("chunk_size", [1, 7, 1_000_000])
def test_sorted_chunk_sizes(store, chunk_size):
    """The size of the chunks does not change the results."""
    source = sorted_source(SPIKES, store, chunk_size=chunk_size)
    lazy = LazyTsGroup(source, metadata={"quality": [QUALITY[k] for k in source.keys]})
    eager = make_eager(SPIKES)
    np.testing.assert_array_equal(lazy.rates, eager.rates)
    for operation in [
        lambda g: g[[2, 3]],
        lambda g: g.restrict(EP),
        lambda g: g.count(0.5),
    ]:
        assert_same(run(operation, lazy), run(operation, eager))


def test_sorted_keys_from_clusters(store):
    """Without ``keys``, the keys are the keys in ``clusters``: a unit with no
    spike is not in the group."""
    spikes = {k: t for k, t in SPIKES.items() if len(t)}
    source = sorted_source(spikes, store)
    times, clusters = source._times, source._clusters
    lazy = LazyTsGroup(_SortedArraySource(times, clusters))
    np.testing.assert_array_equal(lazy.index, sorted(spikes))
    assert_same(lazy.count(0.5), make_eager(spikes).count(0.5))


def test_sorted_equal_times_in_key_order(store):
    """Equal times come back in the order of the keys, as in a TsGroup."""
    times = store("times", np.array([1.0, 2.0, 2.0, 2.0, 3.0]))
    clusters = store("clusters", np.array([5, 9, 2, 5, 2]))
    lazy = LazyTsGroup(_SortedArraySource(times, clusters))
    with warnings.catch_warnings():
        # the Ts of unit 9 has one spike: its time support has no duration
        warnings.simplefilter("ignore")
        eager = nap.TsGroup(
            {2: nap.Ts([2.0, 3.0]), 5: nap.Ts([1.0, 2.0]), 9: nap.Ts([2.0])}
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert lazy == eager
        np.testing.assert_array_equal(lazy._clusters, eager._clusters)


def test_sorted_unsorted_warns(store):
    """A read of unsorted times gives a warning."""
    times = store("times", np.array([1.0, 3.0, 2.0, 4.0]))
    clusters = store("clusters", np.array([0, 1, 0, 1]))
    lazy = LazyTsGroup(_SortedArraySource(times, clusters))
    with pytest.warns(UserWarning, match="Spike times are not sorted"):
        lazy.to_tsd()


def test_sorted_errors():
    with pytest.raises(ValueError, match="must have the same length"):
        _SortedArraySource(np.array([1.0, 2.0]), np.array([0]))
    with pytest.raises(ValueError, match=r"not in keys: \[3\]"):
        _SortedArraySource(np.array([1.0, 2.0]), np.array([0, 3]), keys=[0, 1])


#################################
# Closed source and unsorted spikes
#################################


def test_after_close(make_lazy):
    """After ``source.close()``, the metadata work and a spike read raises a
    RuntimeError. A selection made before still works."""
    lazy = make_lazy(SPIKES)
    selection = lazy[[7]]
    lazy._source.close()

    np.testing.assert_array_equal(lazy.rates, make_eager(SPIKES).rates)
    repr(lazy)
    for operation in [
        lambda g: g.count(1.0),
        lambda g: g[7],
        lambda g: g.restrict(EP),
        lambda g: g.to_tsd(),
    ]:
        with pytest.raises(RuntimeError, match="The file is closed"):
            operation(lazy)
    np.testing.assert_array_equal(selection[7].t, SPIKES[7])


def test_ragged_unsorted_warns(store):
    """A read of a unit with unsorted spikes gives a warning."""
    spikes = {
        3: np.array([5.0, 1.0, 9.0, 6.0]),
        0: np.array([]),
        1: np.array([2.0, 3.0, 4.0]),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lazy = LazyTsGroup(ragged_source(spikes, store))

    with pytest.warns(UserWarning, match=r"units \[3\] are not sorted"):
        lazy[3]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_array_equal(lazy[1].t, [2.0, 3.0, 4.0])


#################################
# Public API
#################################


def test_exported():
    assert nap.LazyTsGroup is LazyTsGroup
    assert issubclass(nap.LazyTsGroup, nap.TsGroup)


@pytest.mark.parametrize("lazy", [True, False])
def test_from_ragged_arrays(store, lazy):
    keys = list(SPIKES)
    flat = np.concatenate([SPIKES[k] for k in keys])
    index = np.cumsum([len(SPIKES[k]) for k in keys])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        units = nap.TsGroup.from_ragged_arrays(
            store("ragged_array", flat),
            index,
            keys=keys,
            metadata={"quality": [QUALITY[k] for k in keys]},
            lazy=lazy,
        )
    assert type(units) is (nap.LazyTsGroup if lazy else nap.TsGroup)
    assert_same(run(lambda g: g.restrict(EP), units), make_eager(SPIKES).restrict(EP))
    assert units == make_eager(SPIKES)


@pytest.mark.parametrize("lazy", [True, False])
def test_from_sorted_arrays(store, lazy):
    """The round trip through ``to_tsd`` gives the same group."""
    eager = make_eager(SPIKES)
    tsd = eager.to_tsd()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        units = nap.TsGroup.from_sorted_arrays(
            store("times", tsd.t),
            store("clusters", tsd.values.astype(np.int64)),
            keys=eager.index,
            metadata={"quality": eager.quality.values},
            lazy=lazy,
        )
    assert type(units) is (nap.LazyTsGroup if lazy else nap.TsGroup)
    assert units == eager


def test_from_ragged_arrays_default_keys():
    units = nap.TsGroup.from_ragged_arrays(
        np.array([0.5, 1.5, 2.0, 1.0, 3.0]), np.array([3, 5])
    )
    np.testing.assert_array_equal(units.index, [0, 1])
    np.testing.assert_array_equal(units[1].t, [1.0, 3.0])


@pytest.mark.parametrize("as_dataframe", [False, True])
def test_metadata_order(as_dataframe):
    """The metadata follow the order of the units in the arrays (ragged) or
    the order of ``keys`` (sorted). The group sorts them with the keys."""
    metadata = {"area": ["a7", "a2"], "depth": [7.0, 2.0]}
    if as_dataframe:
        metadata = pd.DataFrame(metadata)

    ragged = nap.TsGroup.from_ragged_arrays(
        np.array([0.5, 1.5, 1.0, 3.0]),
        np.array([2, 4]),
        keys=[7, 2],
        metadata=metadata,
    )
    sorted_ = nap.TsGroup.from_sorted_arrays(
        np.array([0.5, 1.0, 1.5, 3.0]),
        np.array([7, 2, 7, 2]),
        keys=[7, 2],
        metadata=metadata,
    )
    for units in [ragged, sorted_]:
        np.testing.assert_array_equal(units.index, [2, 7])
        np.testing.assert_array_equal(units.area.values, ["a2", "a7"])
        np.testing.assert_array_equal(units.depth.values, [2.0, 7.0])

    # without keys, the metadata follow the sorted keys in clusters
    units = nap.TsGroup.from_sorted_arrays(
        np.array([0.5, 1.0, 1.5, 3.0]),
        np.array([7, 2, 7, 2]),
        metadata={"area": ["a2", "a7"]},
    )
    np.testing.assert_array_equal(units.area.values, ["a2", "a7"])


@pytest.mark.parametrize(
    "make, message",
    [
        (
            lambda: nap.TsGroup.from_ragged_arrays(
                np.arange(4.0), np.array([2, 4]), keys=[0]
            ),
            "one key for each unit",
        ),
        (
            lambda: nap.TsGroup.from_ragged_arrays(
                np.arange(4.0), np.array([2, 4]), keys=[1, 1]
            ),
            "Two keys have the same value",
        ),
        (
            lambda: nap.TsGroup.from_ragged_arrays(np.arange(4.0), np.array([3, 2])),
            "must not decrease",
        ),
        (
            lambda: nap.TsGroup.from_ragged_arrays(np.arange(4.0), np.array([2, 5])),
            "goes past the end",
        ),
        (
            lambda: nap.TsGroup.from_ragged_arrays(
                np.arange(4.0), np.array([2, 4]), metadata={"area": ["a"]}
            ),
            "Metadata 'area' must have 2 values",
        ),
        (
            lambda: nap.TsGroup.from_sorted_arrays(np.arange(4.0), np.array([0, 1])),
            "must have the same length",
        ),
        (
            lambda: nap.TsGroup.from_sorted_arrays(
                np.arange(4.0), np.array([0, 1, 0, 1]), keys=[0, 1, 1]
            ),
            "Two keys have the same value",
        ),
        (
            lambda: nap.TsGroup.from_sorted_arrays(
                np.arange(4.0), np.array([0, 1, 0, 1]), keys=[0]
            ),
            r"not in keys: \[1\]",
        ),
        (
            lambda: nap.TsGroup.from_sorted_arrays(
                np.arange(4.0), np.array([0, 1, 0, 1]), metadata={"area": "a"}
            ),
            "Metadata 'area' must have 2 values",
        ),
    ],
)
def test_from_arrays_errors(make, message):
    with pytest.raises(ValueError, match=message):
        make()
