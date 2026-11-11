from __future__ import annotations

from collections.abc import Callable
from inspect import signature
from itertools import zip_longest
from textwrap import wrap
from typing import Any

import numpy as np
import pandas as pd
from numba import njit
from numpy.lib.mixins import NDArrayOperatorsMixin

from .interval_set import IntervalSet
from .time_index import TsIndex
from .time_series import Ts, Tsd, TsdFrame
from .ts_group import TsGroup
from .utils import _format_bytes

REPR_WIDTH = 100
REPR_COLUMN_GAP = 2


def _normalize_positions(
    key,
    size: int,
    axis_name: str,
) -> tuple[np.ndarray, bool]:
    """Normalize a positional index."""
    if isinstance(key, (int, np.integer)):
        position = int(key)
        if position < 0:
            position += size
        if not 0 <= position < size:
            raise IndexError(f"{axis_name} index {key} is out of bounds.")
        return np.array([position], dtype=int), True

    positions = np.asarray(np.arange(size)[key])

    if positions.ndim == 0:
        return np.array([int(positions)], dtype=int), True
    if positions.ndim != 1:
        raise IndexError(f"{axis_name} indices must be one-dimensional.")

    return positions.astype(int, copy=False), False


def _format_preview(
    preview: pd.DataFrame,
    *,
    show_index: bool = True,
) -> list[str]:
    """Format a bounded, left-aligned table."""

    def positions(size: int) -> list[int | None]:
        if size <= 6:
            return list(range(size))
        return [0, 1, None, size - 2, size - 1]

    def format_value(value: object) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        return str(value)

    row_positions = positions(len(preview))
    column_positions = positions(len(preview.columns))

    headers = [
        "…" if column is None else str(preview.columns[column])
        for column in column_positions
    ]
    if show_index:
        headers.insert(
            0,
            str(preview.columns.name or preview.index.name or ""),
        )

    rows = [headers]

    if show_index and preview.columns.name and preview.index.name:
        rows.append([str(preview.index.name)] + [""] * len(column_positions))

    for row in row_positions:
        if row is None:
            rows.append(["…"] * len(headers))
            continue

        values = [
            "…" if column is None else format_value(preview.iloc[row, column])
            for column in column_positions
        ]
        if show_index:
            values.insert(0, str(preview.index[row]))
        rows.append(values)

    widths = [max(len(row[column]) for row in rows) for column in range(len(headers))]

    return [
        "  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip()
        for row in rows
    ]


def _format_trial_stack(
    preview: pd.DataFrame,
    trial_ids: np.ndarray,
    *,
    show_index: bool = True,
) -> str:
    """Render a bounded preview of the first trial."""
    if not len(trial_ids):
        return "No trials."

    table = _format_preview(preview, show_index=show_index)

    labels = [str(trial_ids[0])]
    if len(trial_ids) <= 3:
        labels.extend(str(trial_id) for trial_id in trial_ids[1:])
    else:
        labels.extend(
            [
                str(trial_ids[1]),
                "…",
                str(trial_ids[-1]),
            ]
        )

    inner_width = max(
        max(map(len, table)) + 2,
        max(map(len, labels)) + 4,
    )
    height = len(table) + 2
    offset = 2 * (len(labels) - 1)
    canvas = [
        [" "] * (inner_width + 2 + offset) for _ in range(height + len(labels) - 1)
    ]

    for depth in reversed(range(len(labels))):
        x = 2 * depth
        y = len(labels) - depth - 1
        title = f"─ {labels[depth]} "
        top = f"┌{title.ljust(inner_width, '─')}┐"
        bottom = f"└{'─' * inner_width}┘"
        content = table if depth == 0 else [""] * len(table)
        lines = [
            top,
            *(f"│ {line:<{inner_width - 2}} │" for line in content),
            bottom,
        ]

        for row, line in enumerate(lines, start=y):
            canvas[row][x : x + len(line)] = line

    label = "trial ↗ "
    front_row = len(labels) - 1
    return "\n".join(
        (label if row == front_row else " " * len(label)) + "".join(line).rstrip()
        for row, line in enumerate(canvas)
    )


def _format_axes(
    left_title: str,
    left_count: int | str,
    left_columns,
    right_title: str,
    right_count: int | str,
    right_columns,
    *,
    width: int = REPR_WIDTH,
) -> str:
    """Format two axis summaries side by side."""
    metadata_label = "metadata: "
    label_width = len(metadata_label)
    gap = " " * REPR_COLUMN_GAP
    column_width = (width - label_width - REPR_COLUMN_GAP) // 2

    left_names = ", ".join(str(column) for column in left_columns) or "none"
    right_names = ", ".join(str(column) for column in right_columns) or "none"

    left_lines = wrap(
        left_names,
        width=column_width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    right_lines = wrap(
        right_names,
        width=column_width,
        break_long_words=False,
        break_on_hyphens=False,
    )

    indent = " " * label_width
    lines = [
        f"{indent}{left_title:<{column_width}}{gap}{right_title}",
        f"{indent}{f'count: {left_count}':<{column_width}}{gap}count: {right_count}",
    ]

    for index, (left, right) in enumerate(
        zip_longest(left_lines, right_lines, fillvalue="")
    ):
        label = metadata_label if index == 0 else indent
        lines.append(f"{label}{left:<{column_width}}{gap}{right}".rstrip())

    return "\n".join(lines)


def _copy_metadata(obj, *, drop: tuple[str, ...] = ()):
    """Copy an object's metadata."""
    metadata = obj._metadata.copy()

    for column in drop:
        if column in metadata.columns:
            metadata.drop(column)

    return metadata


def _expand_key(key: object, ndim: int) -> tuple[object, ...]:
    """Expand ellipsis and fill omitted axes."""
    keys = key if isinstance(key, tuple) else (key,)
    ellipses = sum(item is Ellipsis for item in keys)

    if ellipses > 1:
        raise IndexError("An index can only have one ellipsis.")

    missing = ndim - (len(keys) - ellipses)
    if missing < 0:
        raise IndexError(f"Too many indices for {ndim}-dimensional data.")

    if ellipses:
        position = next(i for i, item in enumerate(keys) if item is Ellipsis)
        return keys[:position] + (slice(None),) * missing + keys[position + 1 :]

    return keys + (slice(None),) * missing


class _TrialMetadataMixin:
    trials: IntervalSet
    durations: np.ndarray
    alignment: str
    window_starts: np.ndarray
    window_ends: np.ndarray
    _windowed: bool = False

    @property
    def trial_ids(self) -> np.ndarray:
        """Trial identifiers from the interval index."""
        return self.trials.index

    def groupby(self, by: str):
        """
        Group trials by a trial metadata column.

        Trials with the same metadata value are collected into a trial object.

        Parameters
        ----------
        by : str
            Name of a metadata column in ``trials``.

        Returns
        -------
        dict
            Mapping from metadata values to trial subsets. Each subset
            has the same type as the original object: :class:`TsTrials`
            or :class:`TsdTrials`.

        Raises
        ------
        KeyError
            If the trial metadata column does not exist.

        Notes
        -----
        Grouping selects trials, not units or features. Each subset retains
        the original alignment and the selected trials' window boundaries
        and metadata. Unit or feature metadata is retained.

        A group containing one trial still has a trial axis.
        For :class:`TsdTrials`, the shared time axis is unchanged, including
        time points that may be invalid for every trial in a group.

        Use :meth:`groupby_apply` to apply a function to each subset.

        Examples
        --------
        Create three trials with two experimental conditions:

        >>> import numpy as np
        >>> import pynapple as nap
        >>> time = np.arange(13) / 2
        >>> source = nap.Tsd(t=time, d=time)
        >>> intervals = nap.IntervalSet(
        ...     start=[0.0, 2.0, 4.0],
        ...     end=[1.0, 3.0, 5.0],
        ...     metadata={"condition": ["a", "b", "a"]},
        ... )
        >>> data = nap.TsdTrials(source, intervals)
        >>> groups = data.groupby("condition")

        Condition ``"a"`` contains two trials; condition ``"b"`` contains
        one. Both groups retain the three relative time points:

        >>> groups["a"].shape
        (2, 3)
        >>> groups["b"].shape
        (1, 3)
        >>> groups["a"].t
        array([0. , 0.5, 1. ])

        The values remain associated with their original trials:

        >>> groups["a"].d
        array([[0. , 0.5, 1. ],
               [4. , 4.5, 5. ]])
        """

        if by not in self.trials.metadata_columns:
            raise KeyError(f"Unknown trial metadata column: {by!r}")

        return {
            key: self._select_trial_positions(np.asarray(positions, dtype=np.int64))
            for key, positions in self.trials.groupby(by).items()
        }

    def groupby_apply(
        self,
        by: str,
        func: Callable[..., Any],
        **kwargs: Any,
    ) -> dict[Any, Any]:
        """
        Apply a function to each group of trials.

        Trials are grouped by a trial metadata column,
        then each subset is passed to ``func``.

        Parameters
        ----------
        by : str
            Name of a metadata column in ``trials``.
        func : callable
            Function called as ``func(group, **kwargs)``. Its first
            argument is a :class:`TsTrials` or :class:`TsdTrials` subset,
            not a raw array.
        **kwargs
            Additional keyword arguments passed to ``func``.

        Returns
        -------
        dict
            Mapping from metadata values to the corresponding function
            results. Results are returned without further conversion
            or combination.

        Raises
        ------
        KeyError
            If the trial metadata column does not exist.

        Notes
        -----
        This method is equivalent to applying ``func`` to each value
        returned by :meth:`groupby`.

        For :class:`TsdTrials`, use ``axis=0`` to reduce across trials.
        For example, ``np.nanmean`` ignores NaN padding when averaging
        trials with different durations. Reducing only the trial axis
        returns a :class:`~pynapple.Tsd` or :class:`~pynapple.TsdFrame`,
        depending on whether a feature axis is present.

        Examples
        --------
        Compute a mean response for each experimental condition:

        >>> import numpy as np
        >>> import pynapple as nap
        >>> time = np.arange(13) / 2
        >>> source = nap.Tsd(t=time, d=time)
        >>> intervals = nap.IntervalSet(
        ...     start=[0.0, 2.0, 4.0],
        ...     end=[1.0, 3.0, 5.0],
        ...     metadata={"condition": ["a", "b", "a"]},
        ... )
        >>> data = nap.TsdTrials(source, intervals)
        >>> means = data.groupby_apply("condition", np.nanmean, axis=0)

        The result for each condition retains the relative time axis:

        >>> means["a"].t
        array([0. , 0.5, 1. ])
        >>> means["a"].d
        array([2. , 2.5, 3. ])

        Functions can also return ordinary Python values. Count the
        number of trials in each condition:

        >>> sizes = data.groupby_apply("condition", len)
        >>> sizes["a"], sizes["b"]
        (2, 1)
        """

        return {key: func(group, **kwargs) for key, group in self.groupby(by).items()}

    def _select_trial_positions(self, positions: np.ndarray):
        """Select trials by position."""
        raise NotImplementedError

    def _window_summary(self) -> str:
        """Return the trial duration or selected window summary."""
        if not len(self.durations):
            return "empty"
        if not self._windowed:
            if np.allclose(self.durations, self.durations[0]):
                return f"{self.durations[0]:g} sec"
            minimum = float(np.min(self.durations))
            maximum = float(np.max(self.durations))
            return f"variable ({minimum:g}–{maximum:g} sec)"
        if np.allclose(self.window_starts, self.window_starts[0]) and np.allclose(
            self.window_ends, self.window_ends[0]
        ):
            return f"({self.window_starts[0]:g}, {self.window_ends[0]:g}) sec"
        start_min = float(np.min(self.window_starts))
        start_max = float(np.max(self.window_starts))
        end_min = float(np.min(self.window_ends))
        end_max = float(np.max(self.window_ends))
        return (
            f"variable (start: {start_min:g}–{start_max:g}, "
            f"end: {end_min:g}–{end_max:g}) sec"
        )

    def _window_label(self) -> str:
        """Return the duration or window label."""
        return "window" if self._windowed else "duration"


@njit(cache=True)
def _count_spikes(
    spike_times_ns: np.ndarray,
    trial_index: np.ndarray,
    neuron_index: np.ndarray,
    first_edge_ns: int,
    bin_size_ns: int,
    valid: np.ndarray,
    n_units: int,
) -> np.ndarray:
    """Count spikes using integer bin indices."""
    n_trials, n_bins = valid.shape
    counts = np.zeros(
        (n_trials, n_bins, n_units),
        dtype=np.int64,
    )

    for event in range(len(spike_times_ns)):
        trial = trial_index[event]
        neuron = neuron_index[event]
        bin_index = (spike_times_ns[event] - first_edge_ns) // bin_size_ns

        if 0 <= bin_index < n_bins and valid[trial, bin_index]:
            counts[trial, bin_index, neuron] += 1

    return counts


class TsTrials(_TrialMetadataMixin):
    """
    Trial-aligned timestamps from multiple units.

    Each interval defines a trial. Events within a trial are expressed
    relative to its start, so time zero marks the beginning of each trial.

    Parameters
    ----------
    source : TsGroup
        Timestamp data for each unit. All entries must be
        :class:`~pynapple.Ts` objects.
    trials : IntervalSet
        Trial intervals in the same time reference as the source.
        Interval metadata is retained as trial metadata.

    Attributes
    ----------
    trials : IntervalSet
        Original trial intervals and their metadata.
    trial_ids : numpy.ndarray
        Trial identifiers from the interval index.
    unit_keys : numpy.ndarray
        Unit identifiers from the source :class:`~pynapple.TsGroup`.
    unit_metadata : metadata table
        Metadata associated with the units.
    spikes : numpy.ndarray
        Aligned event times in seconds.
    trial_index : numpy.ndarray
        Positional trial index for each event.
    neuron_index : numpy.ndarray
        Positional unit index for each event.
    shape : tuple of int
        Number of trials and units, ordered as ``(trial, unit)``.
    alignment : str
        Alignment reference: ``"start"`` initially, or the metadata column
        passed to :meth:`realign_to`.
    window_starts, window_ends : numpy.ndarray
        Current window boundaries for each trial, in aligned seconds.
    durations : numpy.ndarray
        Current window duration for each trial, in seconds.

    Notes
    -----
    Indexing follows the axis order ``(trial, unit)`` and uses positions,
    not trial identifiers or unit keys.

    Selecting one trial returns a :class:`~pynapple.TsGroup`. Selecting
    one trial and one unit returns a :class:`~pynapple.Ts`. Keeping the
    trial axis with a slice or list returns a :class:`TsTrials` object.

    The event arrays are parallel: ``spikes[i]``, ``trial_index[i]``, and
    ``neuron_index[i]`` describe the same event. Events are not guaranteed
    to be globally sorted by aligned time.

    Use :meth:`realign_to` to express times relative to an event stored
    in trial metadata. Use :meth:`count` to create a :class:`TsdTrials`
    object with axes ``(trial, time, unit)``.

    Examples
    --------
    Create two units and two trials:

    >>> import pynapple as nap
    >>> support = nap.IntervalSet(start=0.0, end=4.0)
    >>> source = nap.TsGroup(
    ...     {
    ...         10: nap.Ts(t=[0.25, 0.75, 2.25], time_support=support),
    ...         20: nap.Ts(t=[0.5, 2.5, 2.75], time_support=support),
    ...     },
    ...     time_support=support,
    ...     metadata={"region": ["ca1", "ca3"]},
    ... )
    >>> intervals = nap.IntervalSet(
    ...     start=[0.0, 2.0],
    ...     end=[1.0, 3.0],
    ...     metadata={"condition": ["a", "b"], "cue": [0.5, 2.5]},
    ... )
    >>> trials = nap.TsTrials(source, intervals)
    >>> trials.shape
    (2, 2)

    Select the first unit by position. Its source key is ``10``:

    >>> trials[0, 0].t
    array([0.25, 0.75])
    >>> trials[1, 0].t
    array([0.25])

    Keep the trial axis when selecting a subset:

    >>> trials[:1].shape
    (1, 2)

    Align events to the cue timestamp in each trial with
    :meth:`realign_to`:

    >>> aligned = trials.realign_to("cue")
    >>> aligned[0, 0].t
    array([-0.25,  0.25])

    Count events in 0.5-second bins with :meth:`count`:

    >>> counts = trials.count(0.5)
    >>> counts.shape
    (2, 2, 2)

    Group trials by their condition metadata with :meth:`groupby`:

    >>> groups = trials.groupby("condition")
    >>> groups["a"].shape
    (1, 2)
    """

    def __init__(
        self,
        source: TsGroup,
        trials: IntervalSet,
    ) -> None:
        if not isinstance(source, TsGroup):
            raise TypeError("source must be a TsGroup.")
        if not isinstance(trials, IntervalSet):
            raise TypeError("trials must be an IntervalSet.")

        if any(not isinstance(source[key], Ts) for key in source):
            raise TypeError("TsTrials requires a TsGroup containing Ts entries.")

        self.trials = trials
        starts = np.asarray(trials.start, dtype=float)
        ends = np.asarray(trials.end, dtype=float)

        self.alignment = "start"
        self.window_starts = np.zeros(len(trials), dtype=float)
        self.window_ends = ends - starts
        self.durations = self.window_ends - self.window_starts

        self.unit_keys = np.asarray(source.index)
        self.unit_metadata = _copy_metadata(source, drop=("rate",))

        (
            self.spikes,
            self.trial_index,
            self.neuron_index,
        ) = self._pack(source, trials, self.unit_keys)

    @classmethod
    def _from_events(
        cls,
        *,
        spikes: np.ndarray,
        trial_index: np.ndarray,
        neuron_index: np.ndarray,
        durations: np.ndarray,
        window_starts: np.ndarray,
        window_ends: np.ndarray,
        trials: IntervalSet,
        unit_keys: np.ndarray,
        unit_metadata,
        alignment: str,
        windowed: bool = False,
    ) -> TsTrials:
        spikes = np.asarray(spikes)
        trial_index = np.asarray(trial_index, dtype=np.int64)
        neuron_index = np.asarray(neuron_index, dtype=np.int64)

        obj = cls.__new__(cls)
        obj.spikes = spikes
        obj.trial_index = trial_index
        obj.neuron_index = neuron_index
        obj.durations = np.asarray(durations, dtype=float)
        obj.window_starts = np.asarray(window_starts, dtype=float)
        obj.window_ends = np.asarray(window_ends, dtype=float)
        obj._windowed = windowed
        obj.trials = trials
        obj.unit_keys = np.asarray(unit_keys)
        obj.unit_metadata = unit_metadata
        obj.alignment = alignment
        return obj

    @staticmethod
    def _pack(
        source: TsGroup,
        trials: IntervalSet,
        unit_keys: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create aligned event arrays."""
        time_chunks = []
        neuron_chunks = []

        for neuron, key in enumerate(unit_keys):
            times = np.asarray(source[key].t, dtype=float)
            if not len(times):
                continue

            time_chunks.append(times)
            neuron_chunks.append(np.full(len(times), neuron, dtype=np.int64))

        if not time_chunks:
            return (
                np.array([], dtype=float),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            )

        absolute_times = np.concatenate(time_chunks)
        neuron_index = np.concatenate(neuron_chunks)

        trial_starts = np.asarray(trials.start, dtype=float)
        trial_ends = np.asarray(trials.end, dtype=float)

        if not len(trial_starts):
            return (
                np.array([], dtype=float),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            )

        trial_index = np.searchsorted(trial_starts, absolute_times, side="right") - 1
        candidates = trial_index >= 0
        inside = np.zeros(len(absolute_times), dtype=bool)
        inside[candidates] = (
            absolute_times[candidates] <= trial_ends[trial_index[candidates]]
        )

        absolute_times = absolute_times[inside]
        neuron_index = neuron_index[inside]
        trial_index = trial_index[inside]

        spikes = absolute_times - trial_starts[trial_index]

        return (spikes, trial_index.astype(np.int64, copy=False), neuron_index)

    @property
    def shape(self) -> tuple[int, int]:
        """Number of trials and units, ordered as ``(trial, unit)``."""
        return len(self.trial_ids), len(self.unit_keys)

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        """Return per-unit spike previews and a trial summary."""
        selected = self.trial_index == 0
        times = self.spikes[selected]
        neurons = self.neuron_index[selected]
        rows = []

        for unit in range(len(self.unit_keys)):
            spikes = np.sort(times[neurons == unit])
            if len(spikes) > 3:
                timestamps = f"{spikes[0]:.3f}  {spikes[1]:.3f}  …  {spikes[-1]:.3f}"
            else:
                timestamps = "  ".join(f"{time:.3f}" for time in spikes)

            rows.append((len(spikes), timestamps))

        preview = pd.DataFrame(
            rows,
            index=pd.Index(self.unit_keys, name="unit ↓"),
            columns=["n", "time →"],
        )
        stack = _format_trial_stack(preview, self.trial_ids)

        size = self.spikes.nbytes + self.trial_index.nbytes + self.neuron_index.nbytes

        return (
            f"{stack}\n\n"
            f"align: {self.alignment!r}, "
            f"{self._window_label()}: {self._window_summary()}\n"
            f"unbinned, shape: (trial: {self.shape[0]}, unit: {self.shape[1]}), "
            f"{len(self.spikes)} spikes, size: {_format_bytes(size)}"
        )

    def __getitem__(self, key):
        """Select trials and units by position."""
        trial_key, unit_key = self._split_key(key)

        trial_positions, trial_scalar = _normalize_positions(
            trial_key,
            self.shape[0],
            "trial",
        )
        unit_positions, unit_scalar = _normalize_positions(
            unit_key,
            self.shape[1],
            "unit",
        )

        if trial_scalar:
            trial = trial_positions[0]

            if unit_scalar:
                return self._get_spike_train(
                    trial,
                    unit_positions[0],
                )

            return self._get_trial_group(
                trial,
                unit_positions,
            )

        return self._subset(
            trial_positions,
            unit_positions,
        )

    @staticmethod
    def _split_key(key: object) -> tuple[object, object]:
        trial_key, unit_key = _expand_key(key, 2)
        return trial_key, unit_key

    def _get_spike_train(self, trial: int, unit: int) -> Ts:
        selected = (self.trial_index == trial) & (self.neuron_index == unit)

        return Ts(
            t=self.spikes[selected],
            time_support=IntervalSet(
                start=self.window_starts[trial],
                end=self.window_ends[trial],
            ),
        )

    def _get_trial_group(
        self,
        trial: int,
        unit_positions: np.ndarray,
    ) -> TsGroup:
        support = IntervalSet(
            start=self.window_starts[trial],
            end=self.window_ends[trial],
        )
        selected_keys = self.unit_keys[unit_positions]

        data = {
            key: self._get_spike_train(trial, unit)
            for unit, key in zip(unit_positions, selected_keys)
        }

        metadata = self.unit_metadata.loc[pd.Index(selected_keys)]
        if metadata.shape[1] == 0:
            metadata = None

        return TsGroup(data, time_support=support, metadata=metadata)

    def _subset(
        self,
        trial_positions: np.ndarray,
        unit_positions: np.ndarray,
    ) -> TsTrials:
        """Create a subset with remapped event indices."""
        trial_map = np.full(self.shape[0], -1, dtype=np.int64)
        unit_map = np.full(self.shape[1], -1, dtype=np.int64)

        trial_map[trial_positions] = np.arange(
            len(trial_positions),
            dtype=np.int64,
        )
        unit_map[unit_positions] = np.arange(
            len(unit_positions),
            dtype=np.int64,
        )

        trial_index = trial_map[self.trial_index]
        neuron_index = unit_map[self.neuron_index]
        selected = (trial_index >= 0) & (neuron_index >= 0)

        trials = self.trials[trial_positions]
        unit_keys = self.unit_keys[unit_positions]
        unit_metadata = self.unit_metadata.loc[unit_keys].copy()

        return self._from_events(
            spikes=self.spikes[selected],
            trial_index=trial_index[selected],
            neuron_index=neuron_index[selected],
            durations=self.durations[trial_positions],
            window_starts=self.window_starts[trial_positions],
            window_ends=self.window_ends[trial_positions],
            windowed=self._windowed,
            trials=trials,
            unit_keys=unit_keys,
            unit_metadata=unit_metadata,
            alignment=self.alignment,
        )

    def _select_trial_positions(
        self,
        positions: np.ndarray,
    ) -> TsTrials:
        return self._subset(
            positions,
            np.arange(self.shape[1]),
        )

    def realign_to(self, column: str) -> TsTrials:
        """
        Align events to timestamps stored in trial metadata.

        The timestamp in the selected column becomes time zero for each trial.
        Events before that timestamp have negative times,
        and events after it have positive times.

        Parameters
        ----------
        column : str
            Trial metadata column containing absolute timestamps in seconds.
            Each timestamp must be finite and lie within its original trial.

        Returns
        -------
        TsTrials
            Events and window boundaries expressed relative to the selected
            timestamps. Trial membership, durations, and metadata are retained.
            The original object is unchanged.

        Raises
        ------
        KeyError
            If the metadata column does not exist.
        TypeError
            If the column does not contain numeric timestamps.
        ValueError
            If a timestamp is missing, non-finite, or outside its trial.

        Notes
        -----
        Metadata timestamps use the original recording time reference, not
        the current aligned time reference.

        Calling this method again aligns to the newly selected column;
        it does not accumulate alignment offsets.

        Alignment does not restore events removed by :meth:`get`.

        Examples
        --------
        Align two trials to a cue occurring 0.5 seconds after each start:

        >>> import pynapple as nap
        >>> support = nap.IntervalSet(start=0.0, end=4.0)
        >>> source = nap.TsGroup(
        ...     {10: nap.Ts(t=[0.25, 0.75, 2.25], time_support=support)},
        ...     time_support=support,
        ... )
        >>> intervals = nap.IntervalSet(
        ...     start=[0.0, 2.0],
        ...     end=[1.0, 3.0],
        ...     metadata={"cue": [0.5, 2.5]},
        ... )
        >>> trials = nap.TsTrials(source, intervals)
        >>> aligned = trials.realign_to("cue")
        >>> aligned[0, 0].t
        array([-0.25,  0.25])

        The trial boundaries move into the same reference frame:

        >>> aligned.window_starts
        array([-0.5, -0.5])
        >>> aligned.window_ends
        array([0.5, 0.5])

        The original alignment is unchanged:

        >>> trials[0, 0].t
        array([0.25, 0.75])
        """

        if column not in self.trials.metadata_columns:
            raise KeyError(f"Unknown trial metadata column: {column!r}")

        alignment = np.asarray(self.trials[column])

        if not np.issubdtype(alignment.dtype, np.number) or np.issubdtype(
            alignment.dtype, np.bool_
        ):
            raise TypeError(f"Trial metadata column {column!r} must be numeric.")

        alignment = alignment.astype(float, copy=False)

        if not np.all(np.isfinite(alignment)):
            raise ValueError(
                f"Trial metadata column {column!r} contains missing or "
                "non-finite timestamps."
            )

        trial_starts = np.asarray(self.trials.start, dtype=float)
        trial_ends = np.asarray(self.trials.end, dtype=float)
        outside = (alignment < trial_starts) | (alignment > trial_ends)

        if np.any(outside):
            trial_ids = self.trial_ids[outside].tolist()
            raise ValueError(
                f"Alignment timestamps must lie within their trials: {trial_ids}"
            )

        target_offset = alignment - trial_starts
        current_offset = (
            np.zeros_like(trial_starts)
            if self.alignment == "start"
            else np.asarray(self.trials[self.alignment], dtype=float) - trial_starts
        )
        shifts = target_offset - current_offset

        return self._from_events(
            spikes=self.spikes - shifts[self.trial_index],
            trial_index=self.trial_index.copy(),
            neuron_index=self.neuron_index.copy(),
            durations=self.durations.copy(),
            window_starts=self.window_starts - shifts,
            window_ends=self.window_ends - shifts,
            windowed=self._windowed,
            trials=self.trials,
            unit_keys=self.unit_keys.copy(),
            unit_metadata=self.unit_metadata.copy(),
            alignment=column,
        )

    def to_tsdframe(self) -> TsdFrame:
        """
        Convert aligned events to a two-column time series.

        Each row represents one event. Its timestamp is the aligned event
        time, and its values identify the trial and unit.

        Returns
        -------
        TsdFrame
            Events sorted by aligned time, with columns ``"trial"`` and
            ``"unit"`` containing trial identifiers and unit keys.

        Notes
        -----
        The identifiers are labels from ``trial_ids`` and ``unit_keys``,
        not positional indices from ``trial_index`` and ``neuron_index``.

        Events from different trials share the same aligned time reference,
        so the returned object may contain repeated timestamps. Trials or
        units without events have no rows.

        The time support spans the earliest current window start through
        the latest current window end. Trial and unit metadata tables are
        not attached to the returned object.

        Examples
        --------
        Combine events from two trials into one table:

        >>> import pynapple as nap
        >>> support = nap.IntervalSet(start=0.0, end=4.0)
        >>> source = nap.TsGroup(
        ...     {10: nap.Ts(t=[0.25, 0.75, 2.5], time_support=support)},
        ...     time_support=support,
        ... )
        >>> intervals = nap.IntervalSet(
        ...     start=[0.0, 2.0], end=[1.0, 3.0]
        ... )
        >>> trials = nap.TsTrials(source, intervals)
        >>> events = trials.to_tsdframe()
        >>> events.t
        array([0.25, 0.5 , 0.75])
        >>> list(events.columns)
        ['trial', 'unit']

        The event at aligned time 0.5 belongs to the second trial:

        >>> events.d.tolist()
        [[0, 10], [1, 10], [0, 10]]
        """

        order = np.argsort(self.spikes, kind="stable")
        values = np.column_stack(
            [
                self.trial_ids[self.trial_index[order]],
                self.unit_keys[self.neuron_index[order]],
            ]
        )

        if len(self.durations):
            support = IntervalSet(
                start=float(self.window_starts.min()),
                end=float(self.window_ends.max()),
            )
        else:
            support = IntervalSet([], [])

        return TsdFrame(
            t=self.spikes[order],
            d=values,
            columns=["trial", "unit"],
            time_support=support,
        )

    def get(self, start: float, end: float) -> TsTrials:
        """
        Select events within an aligned time window across all trials.

        Events at both ``start`` and ``end`` are included.
        The window is intersected with each trial's current window.

        Parameters
        ----------
        start : float
            Window start in seconds, relative to the current alignment.
        end : float
            Window end in seconds, relative to the current alignment.
            Must be greater than ``start``.

        Returns
        -------
        TsTrials
            Selected events with updated window boundaries and durations.

        Raises
        ------
        ValueError
            If either boundary is non-finite or ``start >= end``.

        Notes
        -----
        Selection uses the inclusive interval ``[start, end]``. Events must
        lie within both the requested window and the trial's current window.
        Trials without overlap contain no events.

        Event times retain their current reference: selecting a window does
        not move its start to zero. Use :meth:`realign_to` to change the
        alignment reference.

        The original intervals and metadata in ``trials`` are unchanged.
        The selected boundaries are stored in ``window_starts`` and
        ``window_ends``. Trials without overlap have zero window duration.

        Examples
        --------
        Select events between 0.5 and 1.0 seconds after trial onset:

        >>> import pynapple as nap
        >>> support = nap.IntervalSet(start=0.0, end=2.0)
        >>> source = nap.TsGroup(
        ...     {10: nap.Ts(t=[0.25, 0.5, 0.75, 1.0], time_support=support)},
        ...     time_support=support,
        ... )
        >>> trials = nap.TsTrials(
        ...     source, nap.IntervalSet(start=0.0, end=1.5)
        ... )
        >>> selected = trials.get(0.5, 1.0)
        >>> selected[0, 0].t
        array([0.5 , 0.75, 1.  ])

        Both boundary events are included, and time zero remains the original trial start:

        >>> selected.window_starts
        array([0.5])
        >>> selected.window_ends
        array([1.])
        >>> selected.durations
        array([0.5])
        """

        start = float(start)
        end = float(end)

        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError("start and end must be finite.")
        if start >= end:
            raise ValueError("start must be less than end.")

        window_starts = np.clip(
            start,
            self.window_starts,
            self.window_ends,
        )
        window_ends = np.clip(
            end,
            self.window_starts,
            self.window_ends,
        )

        event_starts = window_starts[self.trial_index]
        event_ends = window_ends[self.trial_index]
        selected = (
            (self.spikes >= start)
            & (self.spikes <= end)
            & (self.spikes >= event_starts)
            & (self.spikes <= event_ends)
        )

        return self._from_events(
            spikes=self.spikes[selected],
            trial_index=self.trial_index[selected],
            neuron_index=self.neuron_index[selected],
            durations=window_ends - window_starts,
            window_starts=window_starts,
            window_ends=window_ends,
            windowed=True,
            trials=self.trials,
            unit_keys=self.unit_keys.copy(),
            unit_metadata=self.unit_metadata.copy(),
            alignment=self.alignment,
        )

    def count(
        self,
        bin_size: float,
        *,
        time_units: str = "s",
        dtype: np.dtype | type = np.float64,
    ) -> TsdTrials:
        """
        Count events in fixed-width bins for each trial and unit.

        All trials share a bin grid in aligned time. Only bins fully contained
        in a trial's current window are valid. Valid bins without events
        contain zero; invalid bins contain NaN.

        Parameters
        ----------
        bin_size : float
            Positive bin width, expressed in ``time_units``.
        time_units : {"s", "ms", "us"}, optional
            Units of ``bin_size``. Default is ``"s"``.
        dtype : numpy.dtype or type, optional
            Floating-point output dtype. Default is ``numpy.float64``.
            Integer dtypes are not supported because invalid bins contain NaN.

        Returns
        -------
        TsdTrials
            Event counts with axes ``(trial, time, unit)``. The time axis
            contains bin centers in seconds. Trial metadata, unit metadata,
            alignment, and window boundaries are retained.

        Raises
        ------
        TypeError
            If ``bin_size`` is not an int or float, or ``dtype`` is not
            floating-point.
        ValueError
            If ``bin_size`` is nonpositive or rounds to less than one
            nanosecond.

        Notes
        -----
        Bin edges are integer multiples of the bin width relative to aligned
        time zero. Bins use the half-open convention ``[left, right)``:
        an event on a bin edge belongs to the bin on its right.

        Partial bins at trial boundaries are invalid rather than counted
        over a shorter duration. An event at the final right edge is therefore
        not included in the preceding bin.

        Times and bin widths are rounded to integer nanoseconds for counting.
        The returned values are counts, not firing rates. Divide by the
        returned ``bin_size`` in seconds to obtain rates in events per second.

        Examples
        --------
        Count events in two trials with different durations:

        >>> import pynapple as nap
        >>> support = nap.IntervalSet(start=0.0, end=4.0)
        >>> source = nap.TsGroup(
        ...     {10: nap.Ts(t=[0.25, 0.75, 2.25, 2.5], time_support=support)},
        ...     time_support=support,
        ... )
        >>> intervals = nap.IntervalSet(
        ...     start=[0.0, 2.0], end=[1.0, 2.75]
        ... )
        >>> trials = nap.TsTrials(source, intervals)
        >>> counts = trials.count(0.5)
        >>> counts.t
        array([0.25, 0.75])
        >>> counts.d[:, :, 0]
        array([[ 1.,  1.],
               [ 1., nan]])

        The second bin is not fully contained in the shorter trial:

        >>> counts.valid
        array([[ True,  True],
               [ True, False]])

        Express the same bin width in milliseconds:

        >>> counts = trials.count(500, time_units="ms")

        Convert counts to rates, or average across trials while ignoring
        invalid bins:

        >>> import numpy as np
        >>> rates = counts / counts.bin_size
        >>> mean_rate = np.nanmean(rates, axis=0)
        """

        if not isinstance(bin_size, (int, float)):
            raise TypeError("bin_size must be an int or float.")
        if bin_size <= 0:
            raise ValueError("bin_size must be positive.")

        bin_size = float(
            TsIndex.format_timestamps(
                np.array([bin_size]),
                time_units,
            )[0]
        )

        dtype = np.dtype(dtype)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError(
                "dtype must be floating-point because invalid bins contain NaN."
            )

        n_trials, n_units = self.shape

        if n_trials == 0:
            return TsdTrials._from_arrays(
                values=np.empty((0, 0, n_units), dtype=dtype),
                time=np.array([], dtype=float),
                valid=np.empty((0, 0), dtype=bool),
                durations=np.array([], dtype=float),
                window_starts=np.array([], dtype=float),
                window_ends=np.array([], dtype=float),
                windowed=self._windowed,
                trials=self.trials,
                feature_names=pd.Index(self.unit_keys),
                feature_metadata=self.unit_metadata,
                sample_period=bin_size,
                bin_size=bin_size,
                data_kind="count",
                alignment=self.alignment,
            )

        nanoseconds = 1_000_000_000
        bin_size_ns = int(np.rint(bin_size * nanoseconds))

        if bin_size_ns <= 0:
            raise ValueError("bin_size must be at least one nanosecond.")

        bin_size = bin_size_ns / nanoseconds

        window_starts_ns = np.rint(self.window_starts * nanoseconds).astype(np.int64)
        window_ends_ns = np.rint(self.window_ends * nanoseconds).astype(np.int64)
        spike_times_ns = np.rint(self.spikes * nanoseconds).astype(np.int64)

        first_bin = int(
            np.floor_divide(
                window_starts_ns.min(),
                bin_size_ns,
            )
        )
        last_bin = int(
            -np.floor_divide(
                -window_ends_ns.max(),
                bin_size_ns,
            )
        )

        edge_numbers = np.arange(
            first_bin,
            last_bin + 1,
            dtype=np.int64,
        )
        edges_ns = edge_numbers * bin_size_ns
        first_edge_ns = int(edges_ns[0])

        time = (edges_ns[:-1].astype(float) + bin_size_ns / 2) / nanoseconds

        valid = (edges_ns[:-1][None, :] >= window_starts_ns[:, None]) & (
            edges_ns[1:][None, :] <= window_ends_ns[:, None]
        )

        counts = _count_spikes(
            spike_times_ns,
            self.trial_index,
            self.neuron_index,
            first_edge_ns,
            bin_size_ns,
            valid,
            n_units,
        )

        values = np.full(counts.shape, np.nan, dtype=dtype)
        values[valid] = counts[valid]

        return TsdTrials._from_arrays(
            values=values,
            time=time,
            valid=valid,
            durations=self.durations.copy(),
            window_starts=self.window_starts.copy(),
            window_ends=self.window_ends.copy(),
            windowed=self._windowed,
            trials=self.trials,
            feature_names=pd.Index(self.unit_keys),
            feature_metadata=self.unit_metadata.copy(),
            sample_period=bin_size,
            bin_size=bin_size,
            data_kind="count",
            alignment=self.alignment,
        )


class TsdTrials(_TrialMetadataMixin, NDArrayOperatorsMixin):
    """
    Trial-aligned sampled data with a shared relative time axis.

    Each interval defines a trial, with time zero at its start.
    Data are stored with axes ``(trial, time)`` for a scalar signal or
    ``(trial, time, feature)`` for multiple signals.
    Shorter trials are padded with NaN.

    Parameters
    ----------
    source : Tsd or TsdFrame
        Uniformly sampled source data containing at least two samples.
        Columns and column metadata are retained for a
        :class:`~pynapple.TsdFrame` source.
    trials : IntervalSet
        Trial intervals in the same time reference as the source.
        Trial starts must coincide with source samples within numerical
        tolerance, and the source must contain the required samples.
        Interval metadata is retained as trial metadata.

    Attributes
    ----------
    trials : IntervalSet
        Original trial intervals and their metadata.
    trial_ids : numpy.ndarray
        Trial identifiers from the interval index.
    values : numpy.ndarray
        Dense data with axes ``(trial, time)`` or
        ``(trial, time, feature)``.
    d : numpy.ndarray
        Alias for ``values``.
    t : numpy.ndarray
        Shared aligned timestamps in seconds.
    valid : numpy.ndarray
        Boolean mask with shape ``(trial, time)`` indicating which
        samples belong to each trial's current window.
    shape : tuple of int
        Dimensions of the data array.
    ndim : int
        Number of data dimensions: two or three.
    columns : pandas.Index
        Feature names for three-dimensional data. Not available for
        scalar trial data.
    feature_names : pandas.Index or None
        Feature names, or ``None`` for scalar trial data.
    feature_metadata : metadata table or None
        Metadata associated with features, or ``None`` for scalar data.
    sample_period : float
        Sampling interval in seconds.
    bin_size : float or None
        Bin width in seconds for data created by
        :meth:`TsTrials.count`; otherwise ``None``.
    data_kind : str
        Data interpretation: ``"continuous"`` for sampled source data
        or ``"count"`` for data created by :meth:`TsTrials.count`.
        Can be changed with :meth:`with_data`.
    alignment : str
        Time reference. Construction uses ``"start"``; data created by
        :meth:`TsTrials.count` inherit the timestamp object's alignment.
    window_starts, window_ends : numpy.ndarray
        Current window boundaries for each trial, in aligned seconds.
    durations : numpy.ndarray
        Current window duration for each trial, in seconds.

    Notes
    -----
    Construction copies source samples without interpolation. Samples
    at both trial boundaries are included within numerical tolerance.
    The shared relative time grid is reconstructed from the estimated
    sampling interval.

    The ``valid`` mask describes sample availability, not whether values
    are finite. A valid sample can contain NaN from the source data.
    Use ``np.nanmean`` rather than ``np.mean`` when NaN padding should
    be excluded from an average.

    Indexing uses positions, not trial identifiers or feature names.
    Selecting one trial returns a :class:`~pynapple.Tsd` or
    :class:`~pynapple.TsdFrame`, with invalid samples removed.
    Keeping both trial and time axes returns a :class:`TsdTrials`.
    Selecting a single time position returns a NumPy array or scalar.
    Lists on different axes select independently rather than pairwise.

    Arithmetic and shape-preserving NumPy ufuncs retain the axes and
    metadata. Supported NumPy reductions return different objects
    depending on the reduced axes:

    - Reducing only the trial axis returns a :class:`~pynapple.Tsd`
      or :class:`~pynapple.TsdFrame`. Times invalid in every trial
      are omitted.
    - Reducing only the feature axis returns a scalar
      :class:`TsdTrials`.
    - Reducing the time axis or all axes returns a NumPy result.
      Reductions over nonempty axes with ``keepdims=True`` also
      return NumPy results.

    Use ``np.asarray(data)`` to access the dense array for operations
    that are not supported directly. Use :meth:`with_data` to attach
    transformed values to the existing trial and time axes.

    Examples
    --------
    Create two trials with different durations from a scalar signal:

    >>> import numpy as np
    >>> import pynapple as nap
    >>> time = np.arange(9) / 2
    >>> source = nap.Tsd(t=time, d=time)
    >>> intervals = nap.IntervalSet(
    ...     start=[0.0, 2.0],
    ...     end=[1.0, 2.5],
    ...     metadata={"condition": ["a", "b"]},
    ... )
    >>> data = nap.TsdTrials(source, intervals)
    >>> data.shape
    (2, 3)
    >>> data.t
    array([0. , 0.5, 1. ])

    The shorter trial is padded rather than filled with zeros:

    >>> data.d
    array([[0. , 0.5, 1. ],
           [2. , 2.5, nan]])
    >>> data.valid
    array([[ True,  True,  True],
           [ True,  True, False]])

    Selecting one trial removes its padding and returns a time series:

    >>> data[1].t
    array([0. , 0.5])
    >>> data[1].d
    array([2. , 2.5])

    Use a slice to retain the trial axis:

    >>> data[1:].shape
    (1, 3)

    Average across trials while ignoring NaN padding:

    >>> mean = np.nanmean(data, axis=0)
    >>> mean.d
    array([1. , 1.5, 1. ])

    A frame source adds a feature axis and retains its column names:

    >>> frame = nap.TsdFrame(
    ...     t=time,
    ...     d=np.column_stack([time, -time]),
    ...     columns=["x", "y"],
    ... )
    >>> population = nap.TsdTrials(frame, intervals)
    >>> population.shape
    (2, 3, 2)
    >>> list(population.columns)
    ['x', 'y']

    Select the first feature across all trials and times:

    >>> population[..., 0].shape
    (2, 3)

    Select an inclusive time window with :meth:`get`:

    >>> window = data.get(0.0, 0.5)
    >>> window.t
    array([0. , 0.5])
    """

    __array_priority__ = 1000

    def __init__(
        self,
        source: Tsd | TsdFrame,
        trials: IntervalSet,
    ) -> None:
        if not isinstance(source, (Tsd, TsdFrame)):
            raise TypeError("source must be a Tsd or TsdFrame.")
        if not isinstance(trials, IntervalSet):
            raise TypeError("trials must be an IntervalSet.")

        timestamps = np.asarray(source.t, dtype=float)
        if len(timestamps) < 2:
            raise ValueError("source must contain at least two samples.")

        differences = np.diff(timestamps)
        sample_period = float(np.median(differences))
        tolerance = max(
            abs(sample_period) * 1e-6,
            np.finfo(float).eps * 32,
        )

        if sample_period <= 0 or not np.allclose(
            differences,
            sample_period,
            rtol=1e-6,
            atol=tolerance,
        ):
            raise ValueError("source must be uniformly sampled.")

        self.sample_period = sample_period
        self.bin_size = None
        self.data_kind = "continuous"

        self.alignment = "start"
        self.trials = trials
        starts = np.asarray(trials.start, dtype=float)
        ends = np.asarray(trials.end, dtype=float)

        self.window_starts = np.zeros(len(trials), dtype=float)
        self.window_ends = ends - starts
        self.durations = self.window_ends - self.window_starts

        source_values = np.asarray(source.d)

        if source_values.ndim == 1:
            self.feature_names = None
            self.feature_metadata = None
        else:
            self.feature_names = pd.Index(source.columns)
            self.feature_metadata = _copy_metadata(source)

        (
            self.values,
            self.time,
            self.valid,
        ) = self._align_source(
            timestamps=timestamps,
            source_values=source_values,
            starts=np.asarray(trials.start, dtype=float),
            durations=self.durations,
            sample_period=sample_period,
            tolerance=tolerance,
        )

    @classmethod
    def _from_arrays(
        cls,
        *,
        values: np.ndarray,
        time: np.ndarray,
        valid: np.ndarray,
        durations: np.ndarray,
        window_starts: np.ndarray,
        window_ends: np.ndarray,
        trials: IntervalSet,
        feature_names: pd.Index | None,
        feature_metadata,
        sample_period: float | None,
        bin_size: float | None,
        data_kind: str,
        alignment: str = "start",
        windowed: bool = False,
    ) -> TsdTrials:
        obj = cls.__new__(cls)
        obj.values = np.asarray(values)
        obj.time = TsIndex(np.asarray(time, dtype=float))
        obj.valid = np.asarray(valid, dtype=bool)
        obj.durations = np.asarray(durations, dtype=float)
        obj.window_starts = np.asarray(window_starts, dtype=float)
        obj.window_ends = np.asarray(window_ends, dtype=float)
        obj._windowed = windowed
        obj.trials = trials
        obj.feature_names = feature_names
        obj.feature_metadata = feature_metadata
        obj.sample_period = sample_period
        obj.bin_size = bin_size
        obj.data_kind = data_kind
        obj.alignment = alignment
        return obj

    def __array_function__(
        self,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        reductions = (
            np.mean,
            np.nanmean,
            np.sum,
            np.nansum,
            np.std,
            np.nanstd,
            np.var,
            np.nanvar,
            np.min,
            np.nanmin,
            np.max,
            np.nanmax,
            np.median,
            np.nanmedian,
            np.prod,
            np.nanprod,
            np.all,
            np.any,
        )

        if func not in reductions:
            return NotImplemented
        if not all(issubclass(cls, (TsdTrials, np.ndarray)) for cls in types):
            return NotImplemented

        arguments = signature(func).bind(*args, **kwargs)
        arguments.apply_defaults()
        parameters = arguments.arguments
        parameters["a"] = np.asarray(parameters["a"])

        result = func(**parameters)

        if parameters.get("out") is not None:
            return result

        axis = parameters["axis"]
        if axis is None:
            return result

        axes = axis if isinstance(axis, tuple) else (axis,)
        axes = tuple(int(value) % self.ndim for value in axes)

        keepdims = parameters.get("keepdims", False)
        if keepdims is np._NoValue:
            keepdims = False

        if 1 in axes or (axes and keepdims):
            return result

        return self._wrap_reduction(result, axes)

    def _wrap_reduction(
        self,
        values: np.ndarray,
        axes: tuple[int, ...],
    ) -> TsdTrials | Tsd | TsdFrame:
        """Wrap a reduction that preserves time."""
        if 0 not in axes:
            result = self._new(values=values)
            if 2 in axes:
                result.feature_names = None
                result.feature_metadata = None
            return result

        valid = self.valid.any(axis=0)
        time = self.t[valid]
        values = values[valid]

        support = (
            IntervalSet(
                start=float(self.window_starts.min()),
                end=float(self.window_ends.max()),
            )
            if len(self)
            else IntervalSet([], [])
        )

        if values.ndim == 1:
            return Tsd(t=time, d=values, time_support=support)

        return TsdFrame(
            t=time,
            d=values,
            time_support=support,
            columns=self.feature_names,
            metadata=self.feature_metadata,
        )

    @staticmethod
    def _align_source(
        *,
        timestamps: np.ndarray,
        source_values: np.ndarray,
        starts: np.ndarray,
        durations: np.ndarray,
        sample_period: float,
        tolerance: float,
    ) -> tuple[np.ndarray, TsIndex, np.ndarray]:
        """Align uniformly sampled data to trial starts."""
        n_trials = len(starts)
        sample_counts = (
            np.floor((durations + tolerance) / sample_period).astype(int) + 1
        )
        max_samples = int(sample_counts.max()) if len(sample_counts) else 0

        time = np.arange(max_samples) * sample_period
        valid = time[None, :] <= durations[:, None] + tolerance

        trailing_shape = source_values.shape[1:]
        values = np.full(
            (n_trials, max_samples, *trailing_shape),
            np.nan,
            dtype=np.result_type(source_values.dtype, float),
        )

        for trial, start in enumerate(starts):
            insertion = np.searchsorted(timestamps, start)
            candidates = np.clip(
                [insertion - 1, insertion],
                0,
                len(timestamps) - 1,
            )
            start_index = int(
                candidates[np.argmin(np.abs(timestamps[candidates] - start))]
            )

            if abs(timestamps[start_index] - start) > tolerance:
                raise ValueError(
                    "Trial starts must align with the source sampling grid."
                )

            count = int(valid[trial].sum())
            stop_index = start_index + count

            if stop_index > len(timestamps):
                raise ValueError(
                    "Trial intervals must be contained within the source time range."
                )

            expected = start + np.arange(count) * sample_period
            actual = timestamps[start_index:stop_index]

            if not np.allclose(
                actual,
                expected,
                rtol=1e-6,
                atol=tolerance,
            ):
                raise ValueError(
                    "Source samples do not align consistently across trials."
                )

            values[trial, :count] = source_values[start_index:stop_index]

        return values, TsIndex(time), valid

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the dense trial tensor."""
        return self.values.shape

    @property
    def ndim(self) -> int:
        """Number of tensor dimensions."""
        return self.values.ndim

    @property
    def t(self) -> np.ndarray:
        """Shared relative time axis."""
        return self.time.values

    @property
    def d(self) -> np.ndarray:
        """Dense trial values."""
        return self.values

    @property
    def columns(self) -> pd.Index:
        """Feature names for frame-like trial data."""
        if self.feature_names is None:
            raise AttributeError("Scalar TsdTrials has no columns.")
        return self.feature_names

    def __len__(self) -> int:
        return self.shape[0]

    def _shape_axes(self) -> str:
        """Return labeled tensor dimensions."""
        names = ["trial", "time"]
        if self.ndim == 3:
            names.append("unit" if self.data_kind == "count" else "feature")
        return (
            "("
            + ", ".join(f"{name}: {size}" for name, size in zip(names, self.shape))
            + ")"
        )

    def __repr__(self) -> str:
        """Return a stacked data preview and tensor summary."""
        if len(self):
            values = self.values[0]
            if values.ndim == 1:
                values = values[:, None]
                columns = ["value"]
            else:
                columns = self.feature_names

            preview = pd.DataFrame(
                values,
                index=pd.Index(
                    [f"{time:.6g}" for time in self.t],
                    name="time ↓",
                ),
                columns=columns,
            )
            preview.columns.name = (
                "unit →" if self.data_kind == "count" else "feature →"
            )
        else:
            preview = pd.DataFrame()

        stack = _format_trial_stack(preview, self.trial_ids)
        sampling_label = (
            "bin_size" if self.data_kind == "count" else "sampling interval"
        )

        return (
            f"{stack}\n\n"
            f"align: {self.alignment!r}, "
            f"{self._window_label()}: {self._window_summary()}, "
            f"{sampling_label}: {self._sample_size_summary()}\n"
            f"dtype: {self.values.dtype}, "
            f"shape: {self._shape_axes()}, "
            f"size: {_format_bytes(self.values.nbytes)}"
        )

    def _sample_size_summary(self) -> str:
        """Return the sampling interval or bin size."""
        size = self.bin_size if self.bin_size is not None else self.sample_period
        return f"{size:g} sec"

    def __array__(
        self,
        dtype=None,
        copy=None,
    ) -> np.ndarray:
        values = np.asarray(self.values, dtype=dtype)
        if copy:
            values = values.copy()
        return values

    def __array_ufunc__(
        self,
        ufunc,
        method,
        *inputs,
        **kwargs,
    ):
        if method != "__call__" or "out" in kwargs:
            return NotImplemented

        arrays = []
        for value in inputs:
            if isinstance(value, TsdTrials):
                if (
                    value.shape != self.shape
                    or not np.array_equal(value.t, self.t)
                    or not np.array_equal(
                        value.valid,
                        self.valid,
                    )
                ):
                    return NotImplemented

                arrays.append(value.values)
            else:
                arrays.append(value)

        result = ufunc(*arrays, **kwargs)

        if not isinstance(result, np.ndarray) or result.shape != self.shape:
            return result

        return self._new(values=result)

    def __getitem__(self, key):
        ## TODO: decide what to do with nonequal sampling
        """Select trials, times and features by position."""
        trial_key, time_key, feature_key = self._split_key(key)

        trial_positions, trial_scalar = _normalize_positions(
            trial_key,
            self.shape[0],
            "trial",
        )
        time_positions, time_scalar = _normalize_positions(
            time_key,
            self.shape[1],
            "time",
        )

        if self.ndim == 3:
            (
                feature_positions,
                feature_scalar,
            ) = _normalize_positions(
                feature_key,
                self.shape[2],
                "feature",
            )
        else:
            if feature_key != slice(None):
                raise IndexError("Two-dimensional TsdTrials has no feature axis.")
            feature_positions = None
            feature_scalar = False

        values = np.take(
            self.values,
            trial_positions,
            axis=0,
        )
        values = np.take(
            values,
            time_positions,
            axis=1,
        )
        valid = np.take(
            self.valid,
            trial_positions,
            axis=0,
        )
        valid = np.take(
            valid,
            time_positions,
            axis=1,
        )

        if feature_positions is not None:
            values = np.take(
                values,
                feature_positions,
                axis=2,
            )

        if time_scalar:
            values = values[:, 0]

            if trial_scalar:
                values = values[0]
            if feature_scalar and isinstance(values, np.ndarray):
                values = np.squeeze(values, axis=-1)

            return values

        if trial_scalar:
            values = values[0]
            trial_valid = valid[0]

            if feature_scalar:
                values = values[:, 0]

            return self._single_trial(
                trial=trial_positions[0],
                time_positions=time_positions,
                values=values,
                valid=trial_valid,
                feature_positions=feature_positions,
            )

        if feature_scalar:
            values = values[:, :, 0]
            feature_names = None
            feature_metadata = None
        elif feature_positions is not None:
            feature_names = self.feature_names.take(feature_positions)
            feature_metadata = self.feature_metadata.loc[feature_names]
        else:
            feature_names = None
            feature_metadata = None

        return self._from_arrays(
            values=values,
            time=self.t[time_positions],
            valid=valid,
            durations=self.durations[trial_positions],
            window_starts=self.window_starts[trial_positions],
            window_ends=self.window_ends[trial_positions],
            windowed=self._windowed,
            trials=self.trials[trial_positions],
            feature_names=feature_names,
            feature_metadata=feature_metadata,
            sample_period=self.sample_period,
            bin_size=self.bin_size,
            data_kind=self.data_kind,
            alignment=self.alignment,
        )

    def _split_key(self, key: object) -> tuple[object, object, object]:
        keys = _expand_key(key, self.ndim)
        if self.ndim == 2:
            return keys[0], keys[1], slice(None)
        return keys[0], keys[1], keys[2]

    def _single_trial(
        self,
        *,
        trial: int,
        time_positions: np.ndarray,
        values: np.ndarray,
        valid: np.ndarray,
        feature_positions: np.ndarray | None,
    ) -> Tsd | TsdFrame:
        time = self.t[time_positions]
        time = time[valid]
        values = values[valid]

        support = IntervalSet(
            start=self.window_starts[trial],
            end=self.window_ends[trial],
        )

        if values.ndim == 1:
            return Tsd(
                t=time,
                d=values,
                time_support=support,
            )

        feature_names = self.feature_names.take(feature_positions)
        feature_metadata = self.feature_metadata.loc[feature_names]

        return TsdFrame(
            t=time,
            d=values,
            time_support=support,
            columns=feature_names,
            metadata=feature_metadata,
        )

    def _select_trial_positions(
        self,
        positions: np.ndarray,
    ) -> TsdTrials:
        return self[positions]

    def _new(self, *, values: np.ndarray) -> TsdTrials:
        return self._from_arrays(
            values=values,
            time=self.t,
            valid=self.valid,
            durations=self.durations,
            window_starts=self.window_starts,
            window_ends=self.window_ends,
            windowed=self._windowed,
            trials=self.trials,
            feature_names=self.feature_names,
            feature_metadata=self.feature_metadata,
            sample_period=self.sample_period,
            bin_size=self.bin_size,
            data_kind=self.data_kind,
            alignment=self.alignment,
        )

    def with_data(
        self,
        values: np.ndarray,
        *,
        columns: pd.Index | list[str] | None = None,
        data_kind: str | None = None,
    ) -> TsdTrials:
        """
        Return new data with the existing trial and time axes.

        Use this method to attach externally computed values, such as
        model predictions or dimensionality-reduction results, to the original trials.

        Parameters
        ----------
        values : array-like
            Replacement data with shape ``(trial, time)`` or
            ``(trial, time, feature)``. The first two dimensions must
            match the original data. The feature dimension may change.
        columns : pandas.Index or list of str, optional
            Names for the replacement features. Must match the feature
            dimension and cannot be supplied for two-dimensional data.
            Explicit names discard existing feature metadata.
        data_kind : str, optional
            Replacement data interpretation. If omitted, the current
            value is retained. Use ``"continuous"`` for transformed values
            that should be selected by timestamp rather than bin edges.

        Returns
        -------
        TsdTrials
            Replacement values with the original time axis, validity mask,
            trial metadata, alignment, and window boundaries.
            The original object is unchanged.

        Raises
        ------
        ValueError
            If the data dimensions do not match, or the column names
            are incompatible with the feature dimension.

        Notes
        -----
        Invalid samples are set to NaN regardless of the supplied values.
        NaNs at valid samples are preserved.

        Without explicit ``columns``, feature names and metadata are
        retained when the feature count is unchanged. Otherwise, new
        features receive integer names and empty metadata.
        Two-dimensional values have no feature names or metadata.

        Matching dimensions do not verify sample identity: replacement
        values must remain in the original trial and time order.

        Changing ``data_kind`` does not transform the values or change
        ``sample_period`` or ``bin_size``. It changes how :meth:`get`
        interprets the data: ``"count"`` requires complete bins, while
        other values select by timestamp.

        Examples
        --------
        Attach a transformed signal to two trials:

        >>> import numpy as np
        >>> import pynapple as nap
        >>> time = np.arange(9) / 2
        >>> source = nap.Tsd(t=time, d=time)
        >>> intervals = nap.IntervalSet(
        ...     start=[0.0, 2.0], end=[1.0, 3.0]
        ... )
        >>> data = nap.TsdTrials(source, intervals)
        >>> squared = data.with_data(data.d ** 2)
        >>> squared.d
        array([[0.  , 0.25, 1.  ],
               [4.  , 6.25, 9.  ]])

        Replace the scalar signal with two named features:

        >>> features = np.stack([data.d, data.d ** 2], axis=-1)
        >>> transformed = data.with_data(
        ...     features, columns=["value", "squared"]
        ... )
        >>> transformed.shape
        (2, 3, 2)
        >>> list(transformed.columns)
        ['value', 'squared']
        """

        values = np.asarray(values)
        if values.ndim not in (2, 3) or values.shape[:2] != self.shape[:2]:
            raise ValueError(
                "values must have matching trial and time dimensions "
                "and an optional feature dimension."
            )
        if values.ndim == 2:
            if columns is not None:
                raise ValueError("Two-dimensional values have no feature axis.")
            names = None
            metadata = None
        elif columns is not None:
            names = pd.Index(columns)
            if len(names) != values.shape[2]:
                raise ValueError("columns must match the feature dimension.")
            metadata = pd.DataFrame(index=names)
        elif self.feature_names is not None and (
            len(self.feature_names) == values.shape[2]
        ):
            names = self.feature_names.copy()
            metadata = self.feature_metadata.copy()
        else:
            names = pd.RangeIndex(values.shape[2])
            metadata = pd.DataFrame(index=names)
        valid = self.valid[..., None] if values.ndim == 3 else self.valid
        result = self._new(values=np.where(valid, values, np.nan))
        result.feature_names = names
        result.feature_metadata = metadata
        if data_kind is not None:
            result.data_kind = data_kind
        return result

    def get(self, start: float, end: float) -> TsdTrials:
        """
        Select an aligned time window across all trials.

        For continuous data, samples at both boundaries are included.
        For count data, only bins fully contained in the window are selected.

        Parameters
        ----------
        start : float
            Window start in seconds, relative to the current alignment.
        end : float
            Window end in seconds, relative to the current alignment.
            Must be greater than ``start``.

        Returns
        -------
        TsdTrials
            Selected time points and values with updated validity,
            window boundaries, and durations. All trials and features
            are retained. The original object is unchanged.

        Raises
        ------
        ValueError
            If either boundary is non-finite or ``start >= end``.

        Notes
        -----
        The requested window is intersected with each trial's current
        window. Invalid samples remain NaN, and previously excluded
        samples are not restored.

        Continuous data use the inclusive interval ``[start, end]``.
        Boundary comparisons allow a small numerical tolerance based
        on the sampling interval.

        For ``data_kind="count"``, both bin edges must lie within the
        window. A bin is not included merely because its center lies
        within the window. Counts are not recomputed or split.

        Selected timestamps keep their existing reference; the window
        start does not become time zero. Original intervals and metadata
        in ``trials`` are unchanged.

        Examples
        --------
        Select a window from two sampled trials:

        >>> import numpy as np
        >>> import pynapple as nap
        >>> time = np.arange(9) / 2
        >>> source = nap.Tsd(t=time, d=time)
        >>> intervals = nap.IntervalSet(
        ...     start=[0.0, 2.0], end=[1.0, 3.0]
        ... )
        >>> data = nap.TsdTrials(source, intervals)
        >>> selected = data.get(0.5, 1.0)
        >>> selected.t
        array([0.5, 1. ])
        >>> selected.d
        array([[0.5, 1. ],
               [2.5, 3. ]])

        The window changes, but the alignment reference does not:

        >>> selected.window_starts
        array([0.5, 0.5])
        >>> selected.durations
        array([0.5, 0.5])
        """

        start = float(start)
        end = float(end)

        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError("start and end must be finite.")
        if start >= end:
            raise ValueError("start must be less than end.")

        tolerance = max(
            abs(self.sample_period) * 1e-6 if self.sample_period is not None else 0.0,
            np.finfo(float).eps * 32,
        )

        if self.data_kind == "count":
            half_bin = self.bin_size / 2
            selected = (self.t - half_bin >= start - tolerance) & (
                self.t + half_bin <= end + tolerance
            )
        else:
            selected = (self.t >= start - tolerance) & (self.t <= end + tolerance)

        time_positions = np.flatnonzero(selected)
        time = self.t[time_positions]
        values = np.take(self.values, time_positions, axis=1)
        valid = np.take(self.valid, time_positions, axis=1)

        window_starts = np.clip(
            start,
            self.window_starts,
            self.window_ends,
        )
        window_ends = np.clip(
            end,
            self.window_starts,
            self.window_ends,
        )

        if self.data_kind == "count":
            half_bin = self.bin_size / 2
            valid &= (
                time[None, :] - half_bin >= window_starts[:, None] - tolerance
            ) & (time[None, :] + half_bin <= window_ends[:, None] + tolerance)
        else:
            valid &= (time[None, :] >= window_starts[:, None] - tolerance) & (
                time[None, :] <= window_ends[:, None] + tolerance
            )

        if values.ndim == 3:
            values = np.where(valid[:, :, None], values, np.nan)
        else:
            values = np.where(valid, values, np.nan)

        return self._from_arrays(
            values=values,
            time=time,
            valid=valid,
            durations=window_ends - window_starts,
            window_starts=window_starts,
            window_ends=window_ends,
            windowed=True,
            trials=self.trials,
            feature_names=self.feature_names,
            feature_metadata=self.feature_metadata,
            sample_period=self.sample_period,
            bin_size=self.bin_size,
            data_kind=self.data_kind,
            alignment=self.alignment,
        )

    def normalize(
        self,
        axes: int | tuple[int, ...] | None = (0, 1),
    ) -> TsdTrials:
        """
        Standardize values to zero mean and unit variance.

        Parameters
        ----------
        axes : int, tuple of int, or None, optional
            Axes over which to compute the mean and standard deviation:
            ``0`` for trials, ``1`` for time, and ``2`` for units or
            features when present. The default, ``(0, 1)``, standardizes
            each unit or feature across trials and time. Use ``None``
            to standardize all values together.

        Returns
        -------
        TsdTrials
            Standardized data with the original axes and metadata.
            The original object is unchanged.

        Notes
        -----
        Invalid samples and NaNs are excluded from the statistics and
        remain NaN in the result. Constant groups become zero. The
        standard deviation is calculated with ``ddof=0``.

        Examples
        --------
        Create two trials from a sampled signal:

        >>> import numpy as np
        >>> import pynapple as nap
        >>> source = nap.Tsd(t=np.arange(20) / 10, d=np.arange(20))
        >>> trials = nap.IntervalSet(start=[0.0, 1.0], end=[0.9, 1.9])
        >>> data = nap.TsdTrials(source=source, trials=trials)

        Standardize across trials and time:

        >>> normalized = data.normalize()

        Standardize each trial independently across time:

        >>> normalized = data.normalize(axes=1)

        """
        valid = self.valid[..., None] if self.ndim == 3 else self.valid
        values = np.where(valid, self.values, np.nan)

        mean = np.nanmean(values, axis=axes, keepdims=True)
        std = np.nanstd(values, axis=axes, keepdims=True)
        std = np.where(std == 0, 1.0, std)

        return self._new(values=(values - mean) / std)

    def convolve(
        self,
        array: np.ndarray,
        trim: str = "both",
    ) -> TsdTrials:
        """
        Convolve each trial independently along the time axis.

        The same one-dimensional kernel is applied separately to every
        trial and feature. The full convolution is trimmed to preserve
        the original data shape and timestamps.

        Parameters
        ----------
        array : array-like
            Nonempty one-dimensional convolution kernel.
            Kernel positions correspond to successive time samples.
        trim : {"both", "left", "right"}, optional
            Side of the full convolution to trim. Default is ``"both"``,
            which takes the centered segment. ``"left"`` removes the
            first ``len(array) - 1`` samples; ``"right"`` removes the
            last ``len(array) - 1`` samples.

        Returns
        -------
        TsdTrials
            Convolved values with the original axes, validity mask,
            and metadata. The original object is unchanged.

        Raises
        ------
        ValueError
            If the kernel is empty or not one-dimensional, ``trim`` is
            unknown, or the time axis does not match a positive, uniform
            sampling interval.

        Notes
        -----
        Each trial is zero-padded outside its valid samples. Invalid
        samples are restored to NaN afterward. Trials and features never
        contribute to one another's convolution.

        The kernel is not normalized, and no edge correction is applied.
        For a moving average, supply a kernel that sums to one. Values
        near trial boundaries may be attenuated by the zero padding.

        NaNs within valid samples propagate locally through the kernel.
        For an even-length kernel, ``trim="both"`` removes
        ``(len(array) - 1) // 2`` samples from the left and the remainder
        from the right.

        Use :meth:`smooth` to construct and apply a Gaussian kernel.

        Examples
        --------
        Apply a three-sample moving average to a constant signal:

        >>> import numpy as np
        >>> import pynapple as nap
        >>> source = nap.Tsd(t=np.arange(5), d=np.ones(5))
        >>> data = nap.TsdTrials(source, nap.IntervalSet(0.0, 4.0))
        >>> averaged = data.convolve(np.ones(3) / 3)
        >>> np.round(averaged.d, 3)
        array([[0.667, 1.   , 1.   , 1.   , 0.667]])

        Boundary values are smaller because the signal is zero-padded.
        The time axis is unchanged:

        >>> averaged.t
        array([0., 1., 2., 3., 4.])
        """
        from scipy import signal

        kernel = np.asarray(array)
        if kernel.ndim != 1 or not kernel.size:
            raise ValueError("array must be a nonempty one-dimensional kernel.")
        if trim not in ("both", "left", "right"):
            raise ValueError("trim must be 'both', 'left', or 'right'.")

        period = self.sample_period
        if (
            period is None
            or period <= 0
            or not np.allclose(np.diff(self.t), period, rtol=1e-6, atol=1e-12)
        ):
            raise ValueError("Convolution requires a uniformly sampled time axis.")

        valid = self.valid[..., None] if self.ndim == 3 else self.valid
        values = np.where(valid, self.values, 0.0)

        if values.size:
            length = len(kernel)
            kernel_shape = (1, length) + (1,) * (self.ndim - 2)
            values = signal.convolve(
                values,
                kernel.reshape(kernel_shape),
                mode="full",
                method="direct",
            )

            if trim == "left":
                start = length - 1
            elif trim == "right":
                start = 0
            else:
                start = (length - 1) // 2

            values = values[:, start : start + self.shape[1]]

        return self._new(values=np.where(valid, values, np.nan))

    def smooth(
        self,
        std: float,
        windowsize: float | None = None,
        time_units: str = "s",
        size_factor: int = 100,
        norm: bool = True,
    ) -> TsdTrials:
        """
        Smooth each trial independently with a Gaussian kernel.

        Gaussian smoothing combines neighboring time samples to reduce rapid fluctuations. 
        Larger standard deviations produce stronger smoothing but reduce temporal detail.

        Parameters
        ----------
        std : float
            Positive, finite standard deviation of the Gaussian kernel,
            expressed in ``time_units``.
        windowsize : float or None, optional
            Positive, finite kernel duration in ``time_units``.
            If omitted, the kernel length is determined by ``std`` and
            ``size_factor``.
        time_units : {"s", "ms", "us"}, optional
            Units of ``std`` and ``windowsize``. Default is ``"s"``.
        size_factor : int, optional
            Positive multiplier used to determine kernel length from
            the standard deviation in samples. Default is ``100``.
            Ignored for length calculation when ``windowsize`` is given.
        norm : bool, optional
            Whether to normalize the kernel to sum to one.
            Default is ``True``.

        Returns
        -------
        TsdTrials
            Smoothed values with the original axes, validity mask,
            and metadata. The original object is unchanged.

        Raises
        ------
        ValueError
            If ``std`` or ``windowsize`` is nonpositive or non-finite,
            ``size_factor`` is not a positive integer, or the time axis
            does not match a positive, uniform sampling interval.

        Notes
        -----
        The standard deviation in samples is ``std_seconds / sample_period``.
        Kernel length is the integer part of this value multiplied by
        ``size_factor``, or the integer part of
        ``windowsize_seconds / sample_period`` when a duration is given.
        An even length is increased by one to obtain an odd-length kernel.

        Each trial and feature is smoothed separately. Values outside
        valid samples are treated as zero, and invalid samples are
        restored to NaN afterward. NaNs within valid samples propagate
        locally through the kernel.

        No edge correction is applied, so values near trial boundaries
        may be attenuated. A kernel much longer than a trial can affect
        every sample through zero padding.

        Smoothing does not convert counts to rates or change
        ``data_kind``. Use :meth:`convolve` to supply a custom kernel.

        Examples
        --------
        Create two trials from a signal sampled at 100 Hz:

        >>> import numpy as np
        >>> import pynapple as nap
        >>> time = np.arange(200) / 100
        >>> source = nap.Tsd(t=time, d=np.sin(2 * np.pi * time))
        >>> intervals = nap.IntervalSet(
        ...     start=[0.0, 1.0], end=[0.9, 1.9]
        ... )
        >>> data = nap.TsdTrials(source, intervals)

        Smooth with a Gaussian standard deviation of 50 ms:

        >>> smoothed = data.smooth(std=50, time_units="ms")

        Limit the kernel duration to 500 ms:

        >>> smoothed = data.smooth(
        ...     std=50, windowsize=500, time_units="ms"
        ... )
        >>> smoothed.shape == data.shape
        True
        """
        from scipy import signal

        if not np.isfinite(std) or std <= 0:
            raise ValueError("std must be finite and positive.")
        if windowsize is not None and (not np.isfinite(windowsize) or windowsize <= 0):
            raise ValueError("windowsize must be finite and positive.")
        if not isinstance(size_factor, int) or size_factor <= 0:
            raise ValueError("size_factor must be a positive integer.")

        period = self.sample_period
        if (
            period is None
            or period <= 0
            or not np.allclose(np.diff(self.t), period, rtol=1e-6, atol=1e-12)
        ):
            raise ValueError("Smoothing requires a uniformly sampled time axis.")

        std_seconds = TsIndex.format_timestamps(np.array([std]), time_units)[0]
        std_samples = std_seconds / period

        if windowsize is None:
            length = int(std_samples * size_factor)
        else:
            window_seconds = TsIndex.format_timestamps(
                np.array([windowsize]), time_units
            )[0]
            length = int(window_seconds / period)

        if length % 2 == 0:
            length += 1

        kernel = signal.windows.gaussian(length, std=std_samples)
        if norm:
            kernel /= kernel.sum()

        valid = self.valid[..., None] if self.ndim == 3 else self.valid
        values = np.where(valid, self.values, 0.0)

        if values.size:
            kernel_shape = (1, length) + (1,) * (self.ndim - 2)
            values = signal.convolve(
                values,
                kernel.reshape(kernel_shape),
                mode="same",
                method="direct",
            )

        return self._new(values=np.where(valid, values, np.nan))
