# TsGroup: sorted-array internal representation, with lazy NWB materialization

## Context

Issue #662 proposes replacing `TsGroup`'s current dict-of-per-unit-`Ts` storage with a
single global sorted timestamp array + a per-spike unit index. This session benchmarked
that alternative (`benchmarks/benchmark_tsgroup_count.py`) against the current
implementation for every major `TsGroup` operation, using real numba/numpy kernels
verified for correctness against the current implementation:

- `count`, `time_diff`, `restrict` (incl. fragmented time supports), and `value_from` are
  all faster on the sorted array — up to 5x, 12x, 20x, and 42x respectively — because the
  current per-unit loop repeats shared bookkeeping (epoch boundaries, target-array
  searches) once per unit even when it's identical across units.
- Slicing a subset of units (`TsGroup[unit_ids]`) is the one case that regresses, up to
  30x slower, because a purely time-sorted array has no per-unit contiguity: selecting a
  few units out of many still requires scanning everything.

Decision made in this conversation: accept that slicing tradeoff. `TsGroup`'s internal
representation becomes the sorted array **unconditionally** — one layout, no dual-view
complexity — except for one case: data loaded from NWB, which is ragged on disk
(`spike_times`/`spike_times_index`, h5py-backed) and should not be eagerly read,
concatenated, and sorted just because a `TsGroup` object was constructed. For that one
source, a thin lazy wrapper defers building the real sorted-array `TsGroup` until the
first actual operation is called on it.

This plan lists the changes in dependency order: new kernels first (additive, testable in
isolation), then the internal array attributes, then rewiring `TsGroup` itself
method-by-method, then the NWB lazy wrapper, then back-compat/tests/docs.

## Open design point to confirm before starting

`TsGroup` members can be `Tsd` (spike times + a value) as well as bare `Ts`, and this is
allowed to be mixed within one group today. A single shared `d` array alongside the
sorted `t`/`unit_index` needs a dtype rule: simplest is "all values must share one numpy
dtype, or none of them have values" (reject mixed `Ts`+`Tsd` groups going forward, or
upcast/NaN-pad). Recommend going with "require uniform dtype, NaN-pad only floats" and
flagging it in the PR description — this is a behavior decision, not just an
implementation detail.

## Ordered changes

### 1. New kernels (additive, zero risk — add + unit-test before touching `TsGroup`)

Promote the prototypes already built and correctness-checked in
`benchmarks/benchmark_tsgroup_count.py` into production kernels:

- `pynapple/core/_jitted_functions.py`:
  - `jitcount_grouped` (from `_jit_count_sorted_array`): single sweep over sorted
    `(t, unit_index)` → `(n_bins, n_units)` matrix.
  - `jitrestrict_grouped` (from `_jit_restrict_sorted_array`): merge-scan over
    `(t, unit_index[, d])` + epoch boundaries → filtered arrays.
  - `jitvaluefrom_grouped` (from `_jit_value_from_sorted_array`): merge-scan match
    against a shared target.
  - `jittimediff_grouped` (from `_jit_time_diff_sorted_array`): single sweep, last-seen-
    per-unit table.
  - `take` no longer needs a numba kernel or remap table now that `_unit_index` holds
    real external keys directly (see step 2): it's just `np.isin(self._unit_index,
    unit_ids)` + boolean indexing, a plain vectorized numpy operation. Drop
    `_jit_take_sorted_array`/`jittake_grouped` from the plan; confirm with a quick
    benchmark that `np.isin` + boolean mask isn't slower than the old remap-based kernel
    before finalizing (not yet re-measured under this simplification).
  - `jitsubsample_grouped` (**new, not yet prototyped**): needs a per-unit running count
    (e.g. reuse the `count`-style sweep to get per-unit totals, then a second pass picking
    each unit's random subset via a per-unit cumulative position counter). Build and
    correctness-check this the same way the others were checked, against
    `Ts.time_diff`-style per-unit looping, before relying on it.
- `pynapple/core/_core_functions.py`: thin wrappers around the above, mirroring the
  existing `_count`/`_restrict`/`_value_from` wrapper pattern. Reuse
  `_restrict_ranges`/`_use_searchsorted_restrict` directly where the benchmark already
  showed the existing vectorized path is competitive (low fragmentation).
- Add unit tests for every new kernel in `tests/` mirroring the `check_*_correctness`
  functions already written in the benchmark script (same method: build a small
  `TsGroup`, compare kernel output to the current per-unit implementation, exactly).

### 2. Internal sorted-array attributes directly on `TsGroup`

No new class. `TsGroup` gains plain attributes, set at construction time, next to the
existing `self.index`/`self.time_support`/`self._metadata`:

- `self._times`: sorted float64, every unit's spikes merged into one global array.
- `self._unit_index`: int64, same length as `self._times`, holding the **real external
  key** for each spike directly — no separate encoding. By construction,
  `np.unique(self._unit_index) == self.index` always holds. This is simpler than a dense
  `0..n_units-1` encoding and removes any risk of a private indexing array drifting out of
  sync with the public `.index`/`.keys()`.
- `self._data`: optional, same length as `self._times`, holds values for `Tsd`-valued
  members (uniform dtype per the open design point above); `None` when the group holds
  only bare `Ts`.
- A `_build_sorted_arrays(data, keys)` helper (module-level or private method) builds
  `_times`/`_unit_index`/`_data` (and `self.index`, unchanged from today) from the current
  constructor's input shape (dict of `Ts`/`Tsd`) — this is the eager path used for every
  non-NWB source.
- `take(unit_ids)` is the single primitive behind all key-based access, not just
  multi-key slicing, and is now trivial: `mask = np.isin(self._unit_index, unit_ids)`,
  then slice `self._times[mask]`/`self._unit_index[mask]`/`self._data[mask]` — no
  relabeling, since `_unit_index` already holds real keys throughout. The new group's
  `.index` is just `unit_ids` (sorted, matching the existing key-sorting convention).
  Single-key access (`tsgroup[0]`, `values()`, `items()`, iteration) is a thin wrapper —
  `take([key])` followed by unwrapping the lone result into a `Ts`/`Tsd` — not a second
  code path.
- **Dense positions are computed transiently, only where a kernel's output is itself a
  dense matrix.** `count`'s output is `(n_bins, n_units)`, and `counts[bin, pos] += 1`
  needs a small dense column position — computed once per `count()` call via
  `pos = np.searchsorted(self.index, self._unit_index)` (cheap: `self.index` is already
  sorted, one vectorized O(N log n_units) call, dwarfed by the kernel's own O(N) sweep),
  then handed to the kernel. This mapping is never stored; it's recomputed on demand and
  discarded. No other kernel built so far (`restrict`, `value_from`, `time_diff`, `take`)
  needs it at all, since none of them produce a dense per-unit-column output.

### 3. Rewire `TsGroup.__init__` (`pynapple/core/ts_group.py:193-322`)

- Call `_build_sorted_arrays(...)` and set `self._times`/`self._unit_index`/`self._data`
  (plus the existing `self.index`) instead of `UserDict.__init__(self, data)`.
- Keep subclassing `UserDict` for back-compat, but make `self.data` a property that
  lazily materializes a real `dict[key -> Ts/Tsd]` (via `take([key])` per key) the first
  time anything touches it directly (covers existing internal uses of `self.data[k]`
  throughout `ts_group.py` and any external code doing the same, e.g.
  `io/interface_nwb.py:377`, until those call sites are migrated in step 6).
- `rate` metadata (`ts_group.py:302`) computed from `self._unit_index` counts / time
  support duration, not from each object's `.rate` attribute.

### 4. Reimplement each public method against the new attributes (one at a time, independently testable)

In increasing order of coupling/risk:

1. `__len__`, `__contains__`, `keys()`, `.index` — unchanged, already backed by
   `self.index`.
2. `__getitem__` (single key) / `values()` / `items()` — via `take([key])`, unwrapped to a
   single `Ts`/`Tsd`.
3. `__getitem__` (list/boolean) / `_ts_group_from_keys` (`ts_group.py:362-413`) — via
   `take(unit_ids)` directly (returns the filtered `_times`/`_unit_index`[`/_data`] for a
   new `TsGroup` with `len(unit_ids)` units).
4. `count()` (`ts_group.py:702-839`) — via the `count` kernel on
   `self._times`/`self._unit_index`.
5. `to_tsd()` (`ts_group.py:841-968`) — becomes `self._times` + a value remap through
   `self.index`; the current manual concatenate+argsort loop goes away entirely.
6. `restrict()` (`ts_group.py:600-643`) and `get()` (`ts_group.py:1124-1147`) — via the
   `restrict` merge-scan kernel.
7. `value_from()` (`ts_group.py:645-700`) — via the `value_from` merge-scan kernel.
8. `time_diff()` (`ts_group.py:1075-1122`) — via the `time_diff` last-seen-per-unit kernel.
9. `subsample()` (`ts_group.py:2032-2133`) — via the `subsample` kernel.
10. `merge_group()`/`merge()` (`ts_group.py:1305-1507`) — concatenate two groups'
    `_times`/`_unit_index`/`_data` arrays directly (keys need no offsetting since
    `_unit_index` already holds real external keys, and the existing non-overlapping-key
    check already guarantees no collisions), instead of concatenating `.items()` lists.
11. `save()`/`_from_npz_reader()` (`ts_group.py:1509-1702`) — `save()` simplifies to
    dumping `self._times`/`self._unit_index` directly (it already nearly produces this
    shape today); `_from_npz_reader()` sets `self._times`/`self._unit_index`/`self._data`
    (and `self.index`) directly from the loaded flat arrays instead of reconstructing one
    `Ts`/`Tsd` per unit first.

### 5. NWB lazy wrapper

- New class in `pynapple/io/interface_nwb.py`, e.g. `_NWBLazyTsGroup(TsGroup)`:
  constructed directly from the NWB units table's `spike_times`/`spike_times_index`
  (h5py-backed, ragged) and metadata, **without** reading/concatenating/sorting anything.
- Design this with a single hook point to keep it cheap to maintain: `_times`,
  `_unit_index`, `_data` (and `.index`) become lazy properties on this subclass, guarded
  by a `self._materialized` flag. The first access to any of them calls
  `self._materialize()` once (reads each unit's ragged slice, concatenates, sorts, calls
  `_build_sorted_arrays`-equivalent logic, caches the arrays as real instance attributes,
  sets `self._materialized = True`); every subsequent access just returns the cached
  arrays like a normal `TsGroup`. No need to override every public method individually,
  since every method in step 4 already goes through these attributes.

### 6. Wire the loader

- `pynapple/io/interface_nwb.py` (currently ~line 220-377: eagerly builds one `Ts`/`Tsd`
  per unit, then `nap.TsGroup(tsgroup, metadata=metainfo)` at line 377) constructs the
  lazy wrapper instead of the eager dict.
- Other loaders (`io/phy.py`, `io/neurosuite.py`, `io/interface_neurosuite.py`,
  `io/interface_neo.py`) are unaffected — they call the public `TsGroup(dict_of_Ts, ...)`
  constructor, which still works (eager array construction via `_build_sorted_arrays`).
  No changes required there for correctness; revisit later only as a perf opportunity.

### 7. Back-compat pass

- `__getattr__`/pickling (`ts_group.py:328-360`): confirm old pickled objects still load;
  document any break.
- Audit `tests/test_ts_group.py` and any other code asserting `isinstance(tsgroup.data,
  dict)` or object identity (`tsgroup[k] is original_ts`) — identity is already not
  preserved today (verified in this session), so this is a low-risk check, not a new
  regression.

### 8. Tests

- Promote every `check_*_correctness` function from `benchmarks/benchmark_tsgroup_count.py`
  into `tests/test_ts_group.py` as permanent regression tests.
- Add an NWB-laziness test: constructing `_NWBLazyTsGroup` from a file must not read
  `spike_times` data; the first operation must trigger exactly one materialization, and a
  second operation must not re-materialize.
- Full existing `tests/test_ts_group.py` suite must keep passing unmodified (public
  contract preserved).

### 9. Docs/changelog

- Note the internal representation change and its performance characteristics (cite the
  benchmark numbers from this session: faster `count`/`time_diff`/`restrict`/`value_from`,
  slower slicing of a small subset from a large group).
- Document the new NWB behavior change explicitly: loading used to eagerly build every
  unit's `Ts`; now the first access after load pays that cost instead, and fast metadata-
  only access (e.g. inspecting `.rates`/`.index` without touching spike times) stays cheap.

## Verification

- Run `pytest tests/test_ts_group.py` (and the full suite) after each numbered step in
  section 4 — each method migration should be independently green before moving to the
  next.
- Run `benchmarks/benchmark_tsgroup_count.py` (promoting its grids to cover `restrict`'s
  fragmented case and `value_from`, which it already does) before/after to confirm the
  real `TsGroup` now matches the prototyped speedups.
- Add the NWB-laziness test described in section 8 and run it against a real or
  minimal mock NWB file to confirm no premature read.
