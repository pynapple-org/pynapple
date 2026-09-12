"""Tests that ``restrict`` dispatches to the intended implementation.

``_Base.restrict`` chooses between:
  - ``_restrict_ranges`` (searchsorted boundaries + contiguous copy) for few
    intervals over numpy data, and
  - ``_restrict`` (the numba merge scan) for many intervals or non-numpy data.

These tests spy on the two functions to assert the routing happens, check the
threshold helper, and verify both paths produce identical output.
"""

from unittest.mock import patch

import numpy as np
import pytest

import pynapple as nap
import pynapple.core.base_class as base_class
from pynapple.core._core_functions import _use_searchsorted_restrict

from .helper_tests import MockArray


def _tsd(n=1000):
    t = np.arange(float(n))
    return nap.Tsd(t=t, d=t.copy(), time_support=nap.IntervalSet(0, n - 1))


def _tiled_intervals(n, m):
    """m disjoint intervals tiled over [0, n-1]."""
    edges = np.linspace(0, n - 1, 2 * m + 1)
    return nap.IntervalSet(start=edges[0:-1:2].copy(), end=edges[1::2].copy())


def _spy():
    """Patch both restrict implementations in the base_class namespace, keeping
    their real behavior (wraps=...) so the operation still runs correctly."""
    return (
        patch.object(base_class, "_restrict_ranges", wraps=base_class._restrict_ranges),
        patch.object(base_class, "_restrict", wraps=base_class._restrict),
    )


# --------------------------------------------------------------------------
# threshold helper
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_intervals, n_samples, expected",
    [
        (1, 100_000, True),  # 1024 < 100000
        (10, 100_000, True),  # 10240 < 100000
        (100, 100_000, False),  # 102400 !< 100000
        (100, 10_000_000, True),  # 102400 < 10M
        (10_000, 10_000_000, False),  # 10.24M !< 10M
    ],
)
def test_use_searchsorted_threshold(n_intervals, n_samples, expected):
    assert _use_searchsorted_restrict(n_intervals, n_samples) == expected


# --------------------------------------------------------------------------
# routing (spy)
# --------------------------------------------------------------------------
def test_few_intervals_route_to_searchsorted():
    tsd = _tsd(100_000)
    ep = nap.IntervalSet(100, 200)  # 1 interval -> searchsorted path
    ranges_patch, scan_patch = _spy()
    with ranges_patch as ranges, scan_patch as scan:
        out = tsd.restrict(ep)
    assert ranges.call_count == 1
    assert scan.call_count == 0
    # sanity on the result
    keep = (tsd.t >= 100) & (tsd.t <= 200)
    np.testing.assert_array_equal(out.t, tsd.t[keep])


def test_many_intervals_route_to_scan():
    n = 100_000
    tsd = _tsd(n)
    ep = _tiled_intervals(n, 200)  # 200 * 1024 !< 100000 -> merge scan
    ranges_patch, scan_patch = _spy()
    with ranges_patch as ranges, scan_patch as scan:
        tsd.restrict(ep)
    assert scan.call_count == 1
    assert ranges.call_count == 0


def test_ts_without_values_routes_to_searchsorted():
    t = np.arange(100_000.0)
    ts = nap.Ts(t=t, time_support=nap.IntervalSet(0, t[-1]))
    ep = nap.IntervalSet(100, 200)
    ranges_patch, scan_patch = _spy()
    with ranges_patch as ranges, scan_patch as scan:
        out = ts.restrict(ep)
    assert ranges.call_count == 1
    assert scan.call_count == 0
    assert len(out) == int(np.sum((t >= 100) & (t <= 200)))


@pytest.mark.parametrize(
    "ep",
    [
        nap.IntervalSet(100, 200),  # few intervals
        # many intervals: numpy data would route to the scan here, lazy must not
        _tiled_intervals(100_000, 200),
    ],
)
def test_non_numpy_data_routes_to_searchsorted(ep):
    """Array-like (non-ndarray) values must use the searchsorted copy path.

    The interval-count heuristic is tuned for numpy; gathering from an h5py/zarr
    dataset is a point selection, which loses to the slice reads at every interval
    count, so lazy data skips the heuristic. `_concat_ranges` only needs `.shape`,
    `.dtype` and slice indexing, all guaranteed by the array-like contract.
    """
    t = np.arange(100_000.0)
    tsd = nap.Tsd(
        t=t,
        d=MockArray(np.arange(100_000.0)),
        time_support=nap.IntervalSet(0, t[-1]),
        load_array=False,
    )
    ranges_patch, scan_patch = _spy()
    with ranges_patch as ranges, scan_patch as scan:
        out = tsd.restrict(ep)
    assert ranges.call_count == 1
    assert scan.call_count == 0

    expected = nap.Tsd(
        t=t, d=np.arange(100_000.0), time_support=nap.IntervalSet(0, t[-1])
    ).restrict(ep)
    np.testing.assert_array_equal(out.t, expected.t)
    np.testing.assert_array_equal(out.values, expected.values)


# --------------------------------------------------------------------------
# both paths must produce identical output (incl. N-D data)
# --------------------------------------------------------------------------
@pytest.fixture(
    params=["ts", "tsd", "tsdframe", "tsdtensor"],
)
def obj(request):
    n = 1000
    t = np.arange(float(n))
    ep = nap.IntervalSet(0, n - 1)
    rng = np.random.default_rng(0)
    if request.param == "ts":
        return nap.Ts(t=t, time_support=ep)
    if request.param == "tsd":
        return nap.Tsd(t=t, d=rng.random(n), time_support=ep)
    if request.param == "tsdframe":
        return nap.TsdFrame(t=t, d=rng.random((n, 4)), time_support=ep)
    return nap.TsdTensor(t=t, d=rng.random((n, 3, 2)), time_support=ep)


@pytest.mark.parametrize(
    "ep",
    [
        nap.IntervalSet(100, 200),  # single window
        nap.IntervalSet(start=[10, 300, 700], end=[50, 400, 900]),  # few windows
    ],
)
def test_searchsorted_and_scan_paths_are_equivalent(obj, ep):
    with patch.object(base_class, "_use_searchsorted_restrict", return_value=True):
        via_searchsorted = obj.restrict(ep)
    with patch.object(base_class, "_use_searchsorted_restrict", return_value=False):
        via_scan = obj.restrict(ep)

    np.testing.assert_array_equal(via_searchsorted.t, via_scan.t)
    if hasattr(obj, "values"):
        np.testing.assert_array_equal(via_searchsorted.d, via_scan.d)
