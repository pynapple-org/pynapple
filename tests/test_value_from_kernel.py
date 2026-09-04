"""Tests for the ``value_from`` matching kernel.

:func:`jitvaluefrom_ranges` locates, for every input timestamp, the target that
``before`` / ``closest`` / ``after`` should pick, working from per-epoch slice
boundaries. It picks between a binary search and a merge scan per epoch, so both
branches are exercised here, and its tie-breaking is checked against an
independent brute-force oracle rather than against a copy of itself.

The tie-breaking rules are not arbitrary: they are the behaviour pynapple has
always had, and they are reachable from user code because duplicate timestamps
are accepted (never rejected nor deduplicated, and a unit conversion can create
them). They are pinned explicitly in ``test_tie_breaking_is_pinned``.
"""

import numpy as np
import pytest
from numba import jit

import pynapple as nap
from pynapple.core._core_functions import _concat_ranges
from pynapple.core._jitted_functions import (
    VALUE_FROM_BSEARCH_RATIO,
    jitrestrict_with_count,
    jitvaluefrom_ranges,
    use_bsearch_match,
)

MODES = {"before": 0, "closest": 1, "after": 2}


# --------------------------------------------------------------------------
# frozen pre-rewrite kernel, kept as a regression reference
# --------------------------------------------------------------------------


@jit(nopython=True, cache=True)
def legacy_jitvaluefrom(
    time_array, time_target_array, count, count_target, starts, mode
):
    """The ``jitvaluefrom`` that pynapple shipped before ``jitvaluefrom_ranges``.

    Copied here verbatim when it was removed from the library, so the rewrite can
    keep being checked against the behaviour that actually shipped rather than only
    against a reimplementation of what that behaviour was believed to be. Do not
    "fix" or tidy this: its value is that it is unchanged. It takes both arrays
    already restricted, plus per-epoch counts.
    """
    m = starts.shape[0]
    n = time_array.shape[0]
    d = time_target_array.shape[0]

    idx = np.full(n, np.nan)

    if n > 0 and d > 0:
        for k in range(m):
            if count[k] > 0 and count_target[k] > 0:
                t = np.sum(count[0:k])
                i = np.sum(count_target[0:k])
                maxt = t + count[k]
                maxi = i + count_target[k]
                while t < maxt:
                    if mode != 1:
                        interval = time_target_array[i] - time_array[t]
                    else:
                        interval = abs(time_target_array[i] - time_array[t])

                    idx[t] = float(i)

                    i += 1
                    while i < maxi:
                        if mode != 1:
                            new_interval = time_target_array[i] - time_array[t]
                            break_cond = (
                                ((new_interval > 0) and (interval <= 0))
                                or (interval >= 0)
                                if mode == 0
                                else ((new_interval < 0) and (interval >= 0))
                                or (interval >= 0)
                            )
                            nan_cond = interval > 0 if mode == 0 else new_interval < 0
                        else:
                            new_interval = abs(time_target_array[i] - time_array[t])
                            break_cond = new_interval > interval
                            nan_cond = False

                        if break_cond:
                            if nan_cond:
                                idx[t] = np.nan
                            break
                        else:
                            idx[t] = float(i)
                            interval = new_interval
                            i += 1

                    if i == maxi:
                        if mode == 2:
                            new_interval = time_target_array[i - 1] - time_array[t]
                            nan_cond = new_interval < 0
                        elif mode == 0:
                            nan_cond = interval > 0
                        else:
                            nan_cond = False
                        if nan_cond:
                            idx[t] = np.nan
                    i -= 1
                    t += 1

    return idx


def run_legacy(time_array, time_target_array, starts, ends, mode):
    """Drive the frozen kernel and translate its output to the new convention.

    It returns float indices into the *restricted* target with NaN for unmatched;
    the new kernel returns int64 indices into the *full* target with -1.
    """
    keep_in, count = jitrestrict_with_count(time_array, starts, ends)
    keep_tg, count_target = jitrestrict_with_count(time_target_array, starts, ends)
    idx = legacy_jitvaluefrom(
        time_array[keep_in],
        time_target_array[keep_tg],
        count,
        count_target,
        starts,
        MODES[mode],
    )
    out = np.full(len(idx), -1, dtype=np.int64)
    matched = ~np.isnan(idx)
    out[matched] = keep_tg[idx[matched].astype(np.int64)]
    return out


# --------------------------------------------------------------------------
# brute-force oracle
# --------------------------------------------------------------------------


def oracle(time_array, time_target_array, starts, ends, mode):
    """O(n*d) reference for the kernel, written independently of it.

    Returns, per kept timestamp, the index into the full target array, or -1 when
    no target qualifies. Epochs are concatenated in order, matching the kernel.
    """
    out = []
    for start, end in zip(starts, ends):
        in_epoch = time_array[(time_array >= start) & (time_array <= end)]
        targets = np.flatnonzero(
            (time_target_array >= start) & (time_target_array <= end)
        )
        for timestamp in in_epoch:
            if len(targets) == 0:
                out.append(-1)
                continue
            candidates = time_target_array[targets]
            if mode == "before":
                ok = np.flatnonzero(candidates <= timestamp)
                if not len(ok):
                    out.append(-1)
                elif candidates[ok[-1]] == timestamp:
                    # an exact hit resolves to the first of the equal run
                    out.append(targets[np.flatnonzero(candidates == timestamp)[0]])
                else:
                    out.append(targets[ok[-1]])
            elif mode == "after":
                ok = np.flatnonzero(candidates >= timestamp)
                out.append(targets[ok[0]] if len(ok) else -1)
            else:
                # closest: the LAST index achieving the minimum distance. That one
                # rule covers both a run of identical targets and two distinct
                # targets that happen to be equidistant.
                dist = np.abs(candidates - timestamp)
                out.append(targets[np.flatnonzero(dist == dist.min())[-1]])
    return np.array(out, dtype=np.int64)


def run_kernel(time_array, time_target_array, starts, ends, mode):
    """Drive the kernel the way ``_value_from`` does."""
    return jitvaluefrom_ranges(
        time_array,
        time_target_array,
        np.searchsorted(time_array, starts, side="left"),
        np.searchsorted(time_array, ends, side="right"),
        np.searchsorted(time_target_array, starts, side="left"),
        np.searchsorted(time_target_array, ends, side="right"),
        MODES[mode],
    )


# --------------------------------------------------------------------------
# tie-breaking, pinned explicitly
# --------------------------------------------------------------------------


def test_tie_breaking_is_pinned():
    """Duplicate targets and equidistant neighbours resolve as they always have.

    ``before`` and ``after`` land on the *first* member of a run of identical
    target timestamps, while ``closest`` lands on the *last*. That is inconsistent
    but long-standing and user-visible, so it is behaviour, not an implementation
    detail.
    """
    time_array = np.array([1.0, 2.0, 3.0])
    time_target_array = np.array([1.0, 2.0, 2.0, 4.0])
    starts, ends = np.array([0.0]), np.array([5.0])

    expected = {
        # t=2.0 hits the duplicate run at indices 1,2 -> first of the run
        "before": [0, 1, 2],
        "after": [0, 1, 3],
        # t=2.0 -> last of the run; t=3.0 is equidistant from 2.0 and 4.0 -> later
        "closest": [0, 2, 3],
    }
    for mode, want in expected.items():
        got = run_kernel(time_array, time_target_array, starts, ends, mode)
        np.testing.assert_array_equal(got, want, err_msg=f"mode={mode}")


def test_equidistant_tie_resolves_to_the_later_target():
    """A strict `<` in the distance comparison would silently flip these."""
    time_array = np.array([4.0])
    time_target_array = np.array([0.0, 3.0, 5.0])  # |4-3| == |5-4|
    got = run_kernel(
        time_array, time_target_array, np.array([0.0]), np.array([9.0]), "closest"
    )
    np.testing.assert_array_equal(got, [2])


def test_tie_breaking_is_visible_through_the_public_api():
    """The kernel's tie rules are reachable without touching internals."""
    target = nap.Tsd(
        t=np.array([1.0, 2.0, 2.0, 4.0]), d=np.array([10.0, 20.0, 21.0, 40.0])
    )
    ts = nap.Ts(t=np.array([2.0]))
    assert ts.value_from(target, mode="before").values[0] == 20.0
    assert ts.value_from(target, mode="after").values[0] == 20.0
    assert ts.value_from(target, mode="closest").values[0] == 21.0


# --------------------------------------------------------------------------
# differential against the oracle
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize(
    "kind", ["uniform", "exact_hits", "duplicate_runs", "integer_grid"]
)
def test_matches_oracle_and_legacy_kernel(mode, kind):
    """Randomised differential test across the input shapes that matter.

    Checked against both references: the brute-force oracle (independent of how the
    kernel works) and the frozen pre-rewrite kernel (independent of what anyone
    believes the rules are). Agreeing with only one of them is not enough.

    ``integer_grid`` is not redundant with ``uniform``: exact distance ties between
    two distinct targets essentially never occur with random floats, so a bug in
    the equidistant rule is invisible without it.
    """
    rng = np.random.default_rng(abs(hash((mode, kind))) % (2**32))
    horizon = 10.0
    for _ in range(400):
        n = int(rng.integers(0, 25))
        d = int(rng.integers(0, 25))
        time_array = np.sort(rng.uniform(0, horizon, n))
        time_target_array = np.sort(rng.uniform(0, horizon, d))
        if kind == "exact_hits" and n and d:
            time_target_array = np.sort(
                np.concatenate([time_target_array, rng.choice(time_array, min(n, 4))])
            )
        elif kind == "duplicate_runs" and d:
            time_target_array = np.sort(
                np.repeat(time_target_array[: max(1, d // 3)], 4)
            )
        elif kind == "integer_grid":
            time_array = np.sort(rng.integers(0, 8, n).astype(float))
            time_target_array = np.sort(rng.integers(0, 8, max(d, 1)).astype(float))

        n_epochs = int(rng.integers(1, 4))
        edges = np.sort(rng.uniform(-1, horizon + 1, 2 * n_epochs))
        starts, ends = edges[0::2].copy(), edges[1::2].copy()
        keep = starts < ends
        if not keep.any():
            continue
        starts, ends = starts[keep], ends[keep]

        got = run_kernel(time_array, time_target_array, starts, ends, mode)
        context = (
            f"t={time_array!r} tt={time_target_array!r} "
            f"starts={starts!r} ends={ends!r} mode={mode}"
        )
        np.testing.assert_array_equal(
            got,
            oracle(time_array, time_target_array, starts, ends, mode),
            err_msg=f"disagrees with the oracle: {context}",
        )
        np.testing.assert_array_equal(
            got,
            run_legacy(time_array, time_target_array, starts, ends, mode),
            err_msg=f"disagrees with the pre-rewrite kernel: {context}",
        )


# --------------------------------------------------------------------------
# both dispatch branches
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_in, n_tg, expected_branch",
    [
        (1, VALUE_FROM_BSEARCH_RATIO, "merge"),  # exactly at the threshold
        (1, VALUE_FROM_BSEARCH_RATIO + 1, "bsearch"),  # one past it
        (1, VALUE_FROM_BSEARCH_RATIO - 1, "merge"),
        (0, 1000, "bsearch"),  # degenerate, must not divide by anything
        (1000, 0, "merge"),
        (1000, 1000, "merge"),
    ],
)
def test_bsearch_threshold(n_in, n_tg, expected_branch):
    """Pin the dispatch predicate itself.

    Nothing else can: both branches return identical results by construction, so a
    flipped predicate is invisible to every correctness test in this file.
    """
    assert use_bsearch_match(n_in, n_tg) == (expected_branch == "bsearch")


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize(
    "n_in, n_tg, expected_branch",
    [
        (1, 4 * VALUE_FROM_BSEARCH_RATIO, "bsearch"),
        (4, 8 * VALUE_FROM_BSEARCH_RATIO, "bsearch"),
        (200, 200, "merge"),
        (VALUE_FROM_BSEARCH_RATIO, VALUE_FROM_BSEARCH_RATIO, "merge"),
    ],
)
def test_both_dispatch_branches_agree_with_oracle(mode, n_in, n_tg, expected_branch):
    """The per-epoch switch must not change results, only speed.

    The branch taken inside the compiled kernel cannot be observed, so the sizes are
    chosen to sit unambiguously on each side of the threshold, confirmed here
    against the same predicate the kernel calls.
    """
    assert use_bsearch_match(n_in, n_tg) == (expected_branch == "bsearch")

    rng = np.random.default_rng(0)
    horizon = 100.0
    time_array = np.sort(rng.uniform(0, horizon, n_in))
    time_target_array = np.sort(rng.uniform(0, horizon, n_tg))
    starts, ends = np.array([0.0]), np.array([horizon])

    np.testing.assert_array_equal(
        run_kernel(time_array, time_target_array, starts, ends, mode),
        oracle(time_array, time_target_array, starts, ends, mode),
    )


@pytest.mark.parametrize("mode", list(MODES))
def test_mixed_density_epochs(mode):
    """One IntervalSet where the switch resolves differently per epoch."""
    rng = np.random.default_rng(3)
    # epoch 0: sparse input over a dense target -> binary search
    # epoch 1: comparable sizes -> merge scan
    sparse_in = np.sort(rng.uniform(0, 10, 2))
    dense_in = np.sort(rng.uniform(20, 30, 300))
    time_array = np.concatenate([sparse_in, dense_in])
    time_target_array = np.concatenate(
        [
            np.sort(rng.uniform(0, 10, 4 * VALUE_FROM_BSEARCH_RATIO)),
            np.sort(rng.uniform(20, 30, 300)),
        ]
    )
    starts, ends = np.array([0.0, 20.0]), np.array([10.0, 30.0])

    np.testing.assert_array_equal(
        run_kernel(time_array, time_target_array, starts, ends, mode),
        oracle(time_array, time_target_array, starts, ends, mode),
    )


# --------------------------------------------------------------------------
# degenerate inputs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize(
    "time_array, time_target_array",
    [
        (np.array([]), np.array([1.0, 2.0])),  # no input
        (np.array([1.0, 2.0]), np.array([])),  # no target
        (np.array([]), np.array([])),  # neither
        (np.array([1.0]), np.array([1.0])),  # exactly one of each, coincident
    ],
)
def test_degenerate_inputs(mode, time_array, time_target_array):
    starts, ends = np.array([0.0]), np.array([5.0])
    got = run_kernel(time_array, time_target_array, starts, ends, mode)
    assert len(got) == len(time_array)
    np.testing.assert_array_equal(
        got, oracle(time_array, time_target_array, starts, ends, mode)
    )


@pytest.mark.parametrize("mode", list(MODES))
def test_epoch_with_input_but_no_target(mode):
    """Every timestamp in such an epoch is unmatched, whatever the mode."""
    time_array = np.array([1.0, 2.0, 11.0, 12.0])
    # brackets every timestamp of the first epoch on both sides, so a match exists
    # there for all three modes; nothing at all lands in [10, 13]
    time_target_array = np.array([0.5, 2.5])
    got = run_kernel(
        time_array,
        time_target_array,
        np.array([0.0, 10.0]),
        np.array([3.0, 13.0]),
        mode,
    )
    assert (got[2:] == -1).all(), "epoch without any target must be all unmatched"
    assert (got[:2] >= 0).all(), "epoch with bracketing targets must all match"


# --------------------------------------------------------------------------
# _concat_ranges
# --------------------------------------------------------------------------


def test_concat_ranges_returns_a_view_only_when_contiguous_and_allowed():
    array = np.arange(20.0)
    contiguous = (np.array([2, 7]), np.array([7, 12]))  # ranges tile [2, 12)
    gapped = (np.array([2, 8]), np.array([5, 12]))  # a gap at [5, 8)

    view = _concat_ranges(array, *contiguous, copy=False)
    assert np.shares_memory(view, array)
    np.testing.assert_array_equal(view, array[2:12])

    assert not np.shares_memory(_concat_ranges(array, *contiguous, copy=True), array)
    assert not np.shares_memory(_concat_ranges(array, *gapped, copy=False), array)
    np.testing.assert_array_equal(
        _concat_ranges(array, *gapped, copy=False),
        np.concatenate([array[2:5], array[8:12]]),
    )


def test_concat_ranges_keeps_trailing_dimensions():
    array = np.arange(30.0).reshape(10, 3)
    out = _concat_ranges(array, np.array([1, 6]), np.array([3, 8]), copy=True)
    np.testing.assert_array_equal(out, np.concatenate([array[1:3], array[6:8]]))


def test_concat_ranges_empty_selection():
    array = np.arange(10.0)
    out = _concat_ranges(array, np.array([4]), np.array([4]), copy=False)
    assert out.shape == (0,)
