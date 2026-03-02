# Plan: POSIX Time Integration in Pynapple

## Background

Pynapple internally uses `TsIndex` — a numpy float64 array of elapsed seconds. POSIX timestamps
(seconds since Unix time_reference, e.g. `1709234567.123`) are large floats that break the assumption of
"small" elapsed-time values and can introduce numerical precision issues. The goal is to let users
pass POSIX timestamps transparently, with pynapple silently working in relative time internally.

---

## Core Design

Create a `PosixIndex` subclass of `TsIndex` in `time_index.py`. It:

- Accepts raw POSIX timestamps at construction
- Subtracts a stored `_time_reference` (first timestamp by default, or user-supplied) to convert to relative seconds
- Stores the time_reference as an instance attribute, propagated through numpy operations via `__array_finalize__`
- Exposes a `.posix` property returning `self.values + self._time_reference`

Everything downstream works in relative seconds — no numerical instability, no changes to jitted
functions, no changes to the public API for existing users.

---

## File-by-File Changes

### 1. `time_index.py` — New `PosixIndex` class (~35 lines)

```python
class PosixIndex(TsIndex):
    """
    TsIndex for POSIX timestamps. Stores an time_reference offset; internal values are relative seconds.

    Parameters
    ----------
    t : numpy.ndarray
        POSIX timestamps (seconds since Unix time_reference).
    time_reference : float, optional
        Reference time (POSIX seconds). If None, the first timestamp is used as t=0.
    """

    def __new__(cls, t, time_reference=None):
        t = np.asarray(t, dtype=np.float64)
        if time_reference is None:
            time_reference = t[0]          # first timestamp becomes t=0
        relative_t = t - time_reference
        obj = super().__new__(cls, relative_t, time_units="s")
        obj._time_reference = time_reference
        return obj

    def __array_finalize__(self, obj):
        self._time_reference = getattr(obj, "_time_reference", 0.0)

    @property
    def time_reference(self):
        """The POSIX reference time (float, seconds since Unix time_reference)."""
        return self._time_reference

    @property
    def posix(self):
        """Return timestamps as POSIX floats (relative seconds + time_reference)."""
        return self.values + self._time_reference
```

`TsIndex.format_timestamps` and `sort_timestamps` are unchanged — they operate on the relative
seconds that `PosixIndex` stores internally.

---

### 2. `base_class.py` — 3 targeted edits

#### a) `_Base.__init__`: handle `time_units="posix"`

```python
# Add a branch alongside the existing "s" / "ms" / "us" handling:
if isinstance(t, TsIndex):
    self.index = t
elif time_units == "posix":
    self.index = PosixIndex(convert_to_numpy_array(t, "t"))
else:
    self.index = TsIndex(convert_to_numpy_array(t, "t"), time_units)
```

#### b) Epoch propagation in operations that slice the index

Methods that extract `self.index.values` (a plain numpy array) and pass the result to
`_define_instance` must re-wrap the output as a `PosixIndex` when the input had one.

Affected methods: `restrict`, `count`, `value_from`, `time_diff`, `find_support`.

Pattern applied at each site (approximately 2 lines per method):

```python
# after computing t_out (numpy array, already in relative seconds)
if isinstance(self.index, PosixIndex):
    t_out = PosixIndex(t_out + self.index.time_reference)  # re-attach time_reference
self._define_instance(t_out, ...)
```

Total: ~10 lines across ~5 methods.

#### c) New convenience methods on `_Base`

```python
@property
def posix_time_reference(self):
    """The POSIX time_reference of the time index, or None if not a POSIX object."""
    return self.index.time_reference if isinstance(self.index, PosixIndex) else None

def to_posix(self):
    """
    Return timestamps as POSIX floats (seconds since Unix time_reference).

    Raises
    ------
    RuntimeError
        If the object was not created with ``time_units="posix"``.
    """
    if not isinstance(self.index, PosixIndex):
        raise RuntimeError(
            "This object does not have a POSIX time_reference. "
            "Create it with time_units='posix'."
        )
    return self.index.posix
```

---

### 3. `time_series.py` — 1 targeted edit in `_BaseTsd.__init__`

When a `time_support` `IntervalSet` is passed to the constructor, `_BaseTsd.__init__` restricts
the data and overwrites `self.index` with a plain `TsIndex`, losing the time_reference. Fix:

```python
# existing (line ~199):
self.index = TsIndex(t)

# replace with:
if isinstance(self.index, PosixIndex):
    self.index = PosixIndex(t + self.index.time_reference)
else:
    self.index = TsIndex(t)
```

---

### 4. `interval_set.py` — support `time_units="posix"` (~15 lines)

`IntervalSet` start/end times must use the same relative reference as the time series they will be
used with. When created with `time_units="posix"`, store the time_reference and subtract it from start/end.

```python
# Near the top of IntervalSet.__init__, before format_timestamps calls:
if time_units == "posix":
    self._posix_time_reference = float(np.min([start[0], end[0]]))
    start = start - self._posix_time_reference
    end   = end   - self._posix_time_reference
    time_units = "s"
else:
    self._posix_time_reference = None
```

Add a property and a convenience method:

```python
@property
def posix_time_reference(self):
    """The POSIX time_reference of the interval set, or None."""
    return self._posix_time_reference

def to_posix(self):
    """Return a DataFrame with start and end times as POSIX floats."""
    if self._posix_time_reference is None:
        raise RuntimeError("This IntervalSet does not have a POSIX time_reference.")
    import pandas as pd
    return pd.DataFrame({
        "start": self.start + self._posix_time_reference,
        "end":   self.end   + self._posix_time_reference,
    })
```

---

## What Does NOT Change

| Component | Reason |
|---|---|
| `_jitted_functions.py` | All jitted ops work on relative float64 seconds |
| `_core_functions.py` | Same |
| `ts_group.py` | Holds `Ts`/`Tsd` objects; consistency is ensured if members share the same time_reference |
| `config.py`, `metadata_class.py`, `utils.py` | Untouched |
| NWB / IO layer | Untouched; POSIX support is opt-in at construction time |
| Existing `time_units` values (`"s"`, `"ms"`, `"us"`) | Behaviour unchanged |

---

## Epoch Consistency

When restricting a POSIX time series with a POSIX `IntervalSet`, both must subtract **the same
time_reference**, otherwise their relative times will not align and `restrict` will silently produce wrong
results.

The only safe strategy is **explicit time_reference sharing**: both objects receive the same `time_reference` value,
either directly or via a helper.

```python
time_reference = posix_times[0]   # or any agreed-upon reference timestamp

tsd = nap.Tsd(t=posix_times, d=data, time_units="posix")
# tsd.posix_time_reference == posix_times[0]

ep = nap.IntervalSet(start=..., end=..., time_units="posix", time_reference=time_reference)
# ep.posix_time_reference  == time_reference  (same value → relative times align)
```

This means both `PosixIndex.__new__` and `IntervalSet.__init__` must accept an explicit `time_reference`
keyword. When omitted, `PosixIndex` defaults to `t[0]` and `IntervalSet` defaults to `start[0]`,
but mixing objects with different defaults is the user's responsibility.

A future utility function `nap.posix_time_reference(*objects)` could compute the minimum timestamp across
all provided arrays or intervals and return a single shared time_reference value.

---

## Usage Example (Target API)

```python
import pynapple as nap
import numpy as np

# POSIX timestamps from a recording system
posix_times = np.array([1700000000.0, 1700000001.0, 1700000002.5])
data = np.random.randn(3)

time_reference = posix_times[0]  # shared reference: 1700000000.0

tsd = nap.Tsd(t=posix_times, d=data, time_units="posix", time_reference=time_reference)
# internally: index stores [0.0, 1.0, 2.5], time_reference = 1700000000.0

tsd.t           # → array([0.0, 1.0, 2.5])   (relative, as always)
tsd.to_posix()  # → array([1700000000.0, 1700000001.0, 1700000002.5])
tsd.posix_time_reference # → 1700000000.0

ep = nap.IntervalSet(start=1700000000.5, end=1700000002.0, time_units="posix", time_reference=time_reference)
# internally: start=[0.5], end=[2.0], time_reference=1700000000.0  (same reference as tsd)
ep.start        # → array([0.5])   (relative)
ep.to_posix()   # → DataFrame with POSIX start/end

tsd.restrict(ep)  # safe: both subtract the same time_reference, relative times align
```

---

## Summary of Changes

| File | Change | Approx. lines |
|---|---|---|
| `time_index.py` | New `PosixIndex` class | ~35 |
| `base_class.py` | `__init__` branch, time_reference propagation in 5 ops, 2 new methods | ~25 |
| `time_series.py` | Epoch preservation in `_BaseTsd.__init__` | ~4 |
| `interval_set.py` | POSIX time_reference in `__init__`, property, `to_posix()` | ~15 |
| **Total** | | **~80 lines** |

The design is purely additive — no existing behaviour changes, opt-in via `time_units="posix"`,
and all heavy numerical operations remain unchanged.