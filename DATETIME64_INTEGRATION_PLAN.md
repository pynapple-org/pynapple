# Plan: numpy datetime64 Integration in Pynapple

## Background

Pynapple internally uses `TsIndex` — a numpy float64 array of elapsed seconds. `numpy.datetime64`
arrays represent absolute wall-clock times (e.g. `np.datetime64('2023-11-01T10:30:00.5', 'us')`)
and have a distinct numpy dtype (`np.dtype('datetime64[...]')`). Unlike POSIX floats, they cannot
be passed directly to pynapple because the internal float64 conversion yields very large numbers
(seconds since 1970), causing the same precision issues.

The goal is to let users pass `datetime64` arrays transparently. Because `datetime64` has a
recognisable dtype, **auto-detection is possible**: no extra `time_units` flag is required from the
user. Pynapple can inspect the dtype at construction time and handle the conversion silently.

---

## Key Differences from POSIX Integration

| Aspect | POSIX (`PosixIndex`) | datetime64 (`Datetime64Index`) |
|---|---|---|
| Input dtype | `float64` — indistinguishable from plain seconds | `datetime64[*]` — detectable from dtype |
| Opt-in mechanism | `time_units="posix"` (explicit flag) | Auto-detected from dtype |
| `time_reference` type | `float` | `np.datetime64` scalar |
| Output property | `.posix` → float array | `.datetime` → datetime64 array |
| Precision concern | Large floats lose sub-ms precision | Depends on unit; safe path via microseconds |

---

## Core Design

Create a `Datetime64Index` subclass of `TsIndex` in `time_index.py`. It:

- Accepts a `numpy.datetime64` array at construction
- Subtracts a `_time_reference` (`np.datetime64` scalar, defaults to `t[0]`) to get relative seconds
- Converts via microseconds to preserve sub-millisecond precision without overflow
- Stores `_time_reference` as a `np.datetime64` instance attribute, propagated via `__array_finalize__`
- Exposes a `.datetime` property that reconstructs the original datetime64 array

Auto-detection happens in `_Base.__init__` and `IntervalSet.__init__` by checking
`np.issubdtype(t.dtype, np.datetime64)` before any other conversion.

---

## Precision Strategy

Converting datetime64 to relative float64 seconds via microseconds is safe for neuroscience:

```python
relative_us = (t - time_reference) / np.timedelta64(1, 'us')   # integer microseconds
relative_s  = relative_us.astype(np.float64) / 1e6             # float64 seconds
```

- For a 24-hour recording: `86400 * 1e6 = 8.64e10` microseconds → `86400.0` seconds.
  Float64 has ~15 significant digits, giving ~1 ns precision over a full day. Fine for neuroscience.
- Nanosecond-resolution datetime64 (`datetime64[ns]`) is cast to microseconds first, losing
  sub-microsecond precision. This matches pynapple's own `time_index_precision` default (9 decimal
  places ≈ 1 ns), so no practical loss.

---

## File-by-File Changes

### 1. `time_index.py` — New `Datetime64Index` class (~40 lines)

```python
class Datetime64Index(TsIndex):
    """
    TsIndex for numpy datetime64 timestamps. Stores a datetime64 time reference;
    internal values are relative float64 seconds.

    Parameters
    ----------
    t : numpy.ndarray of datetime64
        Absolute timestamps.
    time_reference : numpy.datetime64, optional
        Reference time. If None, the first timestamp is used as t=0.
    """

    def __new__(cls, t, time_reference=None):
        t = np.asarray(t)
        if not np.issubdtype(t.dtype, np.datetime64):
            raise TypeError("t must be a numpy datetime64 array.")
        if time_reference is None:
            time_reference = t[0]
        relative_us = (t - time_reference) / np.timedelta64(1, "us")
        relative_s  = relative_us.astype(np.float64) / 1e6
        obj = super().__new__(cls, relative_s, time_units="s")
        obj._time_reference = np.datetime64(time_reference)
        return obj

    def __array_finalize__(self, obj):
        self._time_reference = getattr(obj, "_time_reference", np.datetime64(0, "us"))

    @property
    def time_reference(self):
        """The reference datetime64 timestamp (t=0 in relative seconds)."""
        return self._time_reference

    @property
    def datetime(self):
        """Return timestamps as a numpy datetime64 array."""
        offset_us = (self.values * 1e6).astype(np.int64).astype("timedelta64[us]")
        return self._time_reference + offset_us
```

`TsIndex.format_timestamps` and `sort_timestamps` are unchanged — they operate on the relative
float64 seconds that `Datetime64Index` stores internally.

---

### 2. `base_class.py` — 3 targeted edits

#### a) `_Base.__init__`: auto-detect datetime64

```python
t_array = convert_to_numpy_array(t, "t")

if isinstance(t, TsIndex):               # already a TsIndex (or subclass) — keep as-is
    self.index = t
elif np.issubdtype(t_array.dtype, np.datetime64):   # auto-detected
    self.index = Datetime64Index(t_array)
else:
    self.index = TsIndex(t_array, time_units)
```

No new keyword argument is needed for the common case. A `time_reference` keyword can be threaded
through for users who need to share an explicit reference across objects (see Consistency section).

#### b) Time reference propagation in operations that slice the index

Methods that extract `self.index.values` (plain numpy float64) and pass the result to
`_define_instance` must re-wrap the output as a `Datetime64Index` when the input had one.

Affected methods: `restrict`, `count`, `value_from`, `time_diff`, `find_support`.

Pattern applied at each site (~2 lines per method):

```python
# after computing t_out (numpy float64 array, already in relative seconds)
if isinstance(self.index, Datetime64Index):
    t_out = Datetime64Index(
        self.index.time_reference + (t_out * 1e6).astype(np.int64).astype("timedelta64[us]"),
        time_reference=self.index.time_reference,
    )
self._define_instance(t_out, ...)
```

Total: ~10 lines across ~5 methods.

#### c) New convenience methods on `_Base`

```python
@property
def time_reference(self):
    """The datetime64 time reference, or None if the object is not datetime64-indexed."""
    if isinstance(self.index, Datetime64Index):
        return self.index.time_reference
    return None

def to_datetime(self):
    """
    Return timestamps as a numpy datetime64 array.

    Raises
    ------
    RuntimeError
        If the object was not created from a datetime64 array.
    """
    if not isinstance(self.index, Datetime64Index):
        raise RuntimeError(
            "This object does not have a datetime64 time reference. "
            "Create it from a numpy datetime64 array."
        )
    return self.index.datetime
```

---

### 3. `time_series.py` — 1 targeted edit in `_BaseTsd.__init__`

When a `time_support` `IntervalSet` is passed to the constructor, `_BaseTsd.__init__` restricts
the data and overwrites `self.index` with a plain `TsIndex`, losing the time reference. Fix:

```python
# existing (line ~199):
self.index = TsIndex(t)

# replace with:
if isinstance(self.index, Datetime64Index):
    dt = self.index.time_reference + (t * 1e6).astype(np.int64).astype("timedelta64[us]")
    self.index = Datetime64Index(dt, time_reference=self.index.time_reference)
else:
    self.index = TsIndex(t)
```

---

### 4. `interval_set.py` — auto-detect datetime64 in `__init__` (~20 lines)

`IntervalSet` start/end times must use the same relative reference as the time series they will be
used with. When start/end are `datetime64` arrays, detect and convert them automatically.

```python
# Near the top of IntervalSet.__init__, before format_timestamps calls:
if np.issubdtype(np.asarray(start).dtype, np.datetime64):
    start = np.asarray(start)
    end   = np.asarray(end)
    self._time_reference = start[0]
    start = ((start - self._time_reference) / np.timedelta64(1, "us")).astype(np.float64) / 1e6
    end   = ((end   - self._time_reference) / np.timedelta64(1, "us")).astype(np.float64) / 1e6
    time_units = "s"
else:
    self._time_reference = None
```

Add a property and a convenience method:

```python
@property
def time_reference(self):
    """The datetime64 time reference of the interval set, or None."""
    return self._time_reference

def to_datetime(self):
    """Return a DataFrame with start and end times as numpy datetime64."""
    if self._time_reference is None:
        raise RuntimeError("This IntervalSet does not have a datetime64 time reference.")
    import pandas as pd
    offset_start = (self.start * 1e6).astype(np.int64).astype("timedelta64[us]")
    offset_end   = (self.end   * 1e6).astype(np.int64).astype("timedelta64[us]")
    return pd.DataFrame({
        "start": self._time_reference + offset_start,
        "end":   self._time_reference + offset_end,
    })
```

---

## What Does NOT Change

| Component | Reason |
|---|---|
| `_jitted_functions.py` | All jitted ops work on relative float64 seconds |
| `_core_functions.py` | Same |
| `ts_group.py` | Holds `Ts`/`Tsd` objects; consistency ensured if members share the same `time_reference` |
| `config.py`, `metadata_class.py`, `utils.py` | Untouched |
| NWB / IO layer | Untouched; datetime64 support is transparent at construction |
| Existing `time_units` values (`"s"`, `"ms"`, `"us"`) | Behaviour unchanged |

---

## Time Reference Consistency

The same problem as in the POSIX plan applies: a `Tsd` and an `IntervalSet` built from different
datetime64 arrays will each pick their own `t[0]` / `start[0]` as `time_reference`. If those
differ, `restrict` will silently produce wrong results.

The safe strategy is **explicit time reference sharing**:

```python
time_reference = np.datetime64("2023-11-01T10:30:00", "us")  # agreed-upon reference

tsd = nap.Tsd(t=dt64_times, d=data)
# tsd.time_reference == dt64_times[0]  (auto-detected, may differ from above)

ep = nap.IntervalSet(
    start=dt64_starts,
    end=dt64_ends,
    time_reference=time_reference,   # pinned to same reference
)
```

This requires threading a `time_reference` keyword through `IntervalSet.__init__` and
`Datetime64Index.__new__` (already present in the class design above).

A future utility `nap.shared_time_reference(*objects)` could extract the earliest datetime64
timestamp across a collection of arrays and intervals, returning a single `np.datetime64` value to
pass to all constructors.

---

## Usage Example (Target API)

```python
import pynapple as nap
import numpy as np

# datetime64 timestamps from a recording system (microsecond resolution)
dt64_times = np.array([
    "2023-11-01T10:30:00.000000",
    "2023-11-01T10:30:01.000000",
    "2023-11-01T10:30:02.500000",
], dtype="datetime64[us]")
data = np.random.randn(3)

# No time_units flag needed — dtype is detected automatically
tsd = nap.Tsd(t=dt64_times, d=data)
# internally: time_reference = dt64_times[0], index stores [0.0, 1.0, 2.5]

tsd.t              # → array([0.0, 1.0, 2.5])   (relative seconds, as always)
tsd.to_datetime()  # → array(['2023-11-01T10:30:00', '...01', '...02.5'], dtype='datetime64[us]')
tsd.time_reference # → np.datetime64('2023-11-01T10:30:00', 'us')

# IntervalSet from datetime64 — also auto-detected
dt64_starts = np.array(["2023-11-01T10:30:00.500000"], dtype="datetime64[us]")
dt64_ends   = np.array(["2023-11-01T10:30:02.000000"], dtype="datetime64[us]")

ep = nap.IntervalSet(
    start=dt64_starts,
    end=dt64_ends,
    time_reference=tsd.time_reference,   # share the same reference for safety
)
ep.start         # → array([0.5])   (relative seconds)
ep.to_datetime() # → DataFrame with datetime64 start/end columns

tsd.restrict(ep)  # safe: both use the same time_reference, relative times align
```

---

## Summary of Changes

| File | Change | Approx. lines |
|---|---|---|
| `time_index.py` | New `Datetime64Index` class | ~40 |
| `base_class.py` | dtype auto-detection in `__init__`, propagation in 5 ops, 2 new methods | ~30 |
| `time_series.py` | Time reference preservation in `_BaseTsd.__init__` | ~5 |
| `interval_set.py` | datetime64 auto-detection in `__init__`, property, `to_datetime()` | ~20 |
| **Total** | | **~95 lines** |

The design is purely additive — no existing behaviour changes, transparent for datetime64 inputs,
and all heavy numerical operations remain unchanged.