from __future__ import annotations

from itertools import zip_longest
from textwrap import wrap

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
    if key is Ellipsis:
        key = slice(None)

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
        "  ".join(
            value.ljust(width) for value, width in zip(row, widths, strict=True)
        ).rstrip()
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


class _TrialMetadataMixin:
    trials: IntervalSet
    durations: np.ndarray
    alignment: str
    window_starts: np.ndarray
    window_ends: np.ndarray

    @property
    def trial_ids(self) -> np.ndarray:
        return self.trials.index

    def groupby(self, by: str):
        """Group trials by a metadata column."""
        if by not in self.trials.metadata_columns:
            raise KeyError(f"Unknown trial metadata column: {by!r}")

        return {
            key: self._select_trial_positions(np.asarray(positions, dtype=np.int64))
            for key, positions in self.trials.groupby(by).items()
        }

    def _select_trial_positions(self, positions: np.ndarray):
        """Select trials by position."""
        raise NotImplementedError

    def _window_summary(self) -> str:
        """Return the trial duration or aligned window summary."""
        if not len(self.durations):
            return "empty"

        if self.alignment == "start":
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
        return "duration" if self.alignment == "start" else "window"


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
    Trial-aligned timestamps.

    Parameters
    ----------
    source : TsGroup
        Timestamp data for each unit.
    trials : IntervalSet
        Trial intervals, unit timestamps are aligned to trial starts.

    Notes
    -----
    Events are represented by three parallel arrays: sorted aligned spike
    times, positional trial indices, and positional neuron indices.
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
        """Create sorted aligned event arrays."""
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
            f"unbinned, shape: {self.shape} (trial, unit), "
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
    def _split_key(key) -> tuple[object, object]:
        if not isinstance(key, tuple):
            return key, slice(None)

        if len(key) > 2:
            raise IndexError("TsTrials supports trial and unit axes only.")
        if not key:
            return slice(None), slice(None)
        if len(key) == 1:
            return key[0], slice(None)

        return key

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
            for unit, key in zip(
                unit_positions,
                selected_keys,
                strict=True,
            )
        }

        metadata = self.unit_metadata.loc[pd.Index(selected_keys)]
        if metadata.shape[1] == 0:
            metadata = None

        return TsGroup(
            data,
            time_support=support,
            bypass_check=True,
            metadata=metadata,
        )

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
        """Return trials aligned to metadata timestamps."""
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

        shifts = alignment - trial_starts

        return self._from_events(
            spikes=self.spikes - shifts[self.trial_index],
            trial_index=self.trial_index.copy(),
            neuron_index=self.neuron_index.copy(),
            durations=self.durations.copy(),
            window_starts=self.window_starts - shifts,
            window_ends=self.window_ends - shifts,
            trials=self.trials,
            unit_keys=self.unit_keys.copy(),
            unit_metadata=self.unit_metadata.copy(),
            alignment=column,
        )

    def to_tsdframe(self) -> TsdFrame:
        """Return all aligned spikes as a two-column event table."""
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
        """Return data within an aligned time window."""
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
        selected = (self.spikes >= event_starts) & (self.spikes < event_ends)

        return self._from_events(
            spikes=self.spikes[selected],
            trial_index=self.trial_index[selected],
            neuron_index=self.neuron_index[selected],
            durations=window_ends - window_starts,
            window_starts=window_starts,
            window_ends=window_ends,
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
        """Count aligned spikes in fixed-width bins."""
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
    Trial-aligned timestamps with data.

    Parameters
    ----------
    source : Tsd or TsdFrame
        Uniformly sampled source data.
    trials : IntervalSet
        Trial intervals, data is aligned to trial starts.
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
    ) -> TsdTrials:
        obj = cls.__new__(cls)
        obj.values = np.asarray(values)
        obj.time = TsIndex(np.asarray(time, dtype=float))
        obj.valid = np.asarray(valid, dtype=bool)
        obj.durations = np.asarray(durations, dtype=float)
        obj.window_starts = np.asarray(window_starts, dtype=float)
        obj.window_ends = np.asarray(window_ends, dtype=float)
        obj.trials = trials
        obj.feature_names = feature_names
        obj.feature_metadata = feature_metadata
        obj.sample_period = sample_period
        obj.bin_size = bin_size
        obj.data_kind = data_kind
        obj.alignment = alignment
        return obj

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
        sample_counts = np.ceil(
            np.maximum(durations - tolerance, 0.0) / sample_period
        ).astype(int)
        max_samples = int(sample_counts.max()) if len(sample_counts) else 0

        time = np.arange(max_samples) * sample_period
        valid = time[None, :] < durations[:, None] - tolerance

        trailing_shape = source_values.shape[1:]
        values = np.full(
            (n_trials, max_samples, *trailing_shape),
            np.nan,
            dtype=np.result_type(source_values.dtype, float),
        )

        source_origin = timestamps[0]

        for trial, start in enumerate(starts):
            grid_position = (start - source_origin) / sample_period
            start_index = int(round(grid_position))

            if not np.isclose(
                grid_position,
                start_index,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    "Trial starts must align with the source sampling grid."
                )

            count = int(valid[trial].sum())
            stop_index = start_index + count

            if start_index < 0 or stop_index > len(timestamps):
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
        axes = (
            "(trial, time, unit)"
            if self.ndim == 3 and self.data_kind == "count"
            else self._shape_axes()
        )

        return (
            f"{stack}\n\n"
            f"align: {self.alignment!r}, "
            f"{self._window_label()}: {self._window_summary()}, "
            f"{sampling_label}: {self._sample_size_summary()}\n"
            f"dtype: {self.values.dtype}, "
            f"shape: {self.shape} {axes}, "
            f"size: {_format_bytes(self.values.nbytes)}"
        )

    def _sample_size_summary(self) -> str:
        """Return the sampling interval or bin size."""
        size = self.bin_size if self.bin_size is not None else self.sample_period
        return f"{size:g} sec"

    def _shape_axes(self) -> str:
        """Return tensor axis labels."""
        if self.ndim == 3:
            return "(trial, time, feature)"
        return "(trial, time)"

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
            trials=self.trials[trial_positions],
            feature_names=feature_names,
            feature_metadata=feature_metadata,
            sample_period=self.sample_period,
            bin_size=self.bin_size,
            data_kind=self.data_kind,
            alignment=self.alignment,
        )

    def _split_key(
        self,
        key,
    ) -> tuple[object, object, object]:
        if not isinstance(key, tuple):
            return key, slice(None), slice(None)

        if len(key) > self.ndim:
            raise IndexError(f"TsdTrials has {self.ndim} dimensions.")

        keys = key + (slice(None),) * (self.ndim - len(key))

        if self.ndim == 2:
            return keys[0], keys[1], slice(None)

        return keys

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
            trials=self.trials,
            feature_names=self.feature_names,
            feature_metadata=self.feature_metadata,
            sample_period=self.sample_period,
            bin_size=self.bin_size,
            data_kind=self.data_kind,
            alignment=self.alignment,
        )

    def get(self, start: float, end: float) -> TsdTrials:
        """Return data within an aligned time window."""
        start = float(start)
        end = float(end)

        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError("start and end must be finite.")
        if start >= end:
            raise ValueError("start must be less than end.")

        tolerance = np.finfo(float).eps * 16

        if self.data_kind == "count":
            half_bin = self.bin_size / 2
            selected = (self.t - half_bin >= start - tolerance) & (
                self.t + half_bin <= end + tolerance
            )
        else:
            selected = (self.t >= start - tolerance) & (self.t < end - tolerance)

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
                time[None, :] < window_ends[:, None] - tolerance
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
            0 for trials, 1 for time, and 2 for units or features when
            present. The default, (0, 1), standardises each unit or feature
            across trials and time. Use None to standardise all values
            together.

        Returns
        -------
        TsdTrials
            Standardized data with the original axes and metadata.
            The original object is unchanged.

        Notes
        -----
        Invalid bins and NaNs are excluded from the statistics and remain
        NaN in the result. Constant groups become zero. The standard
        deviation is calculated with ddof=0.

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

        Parameters
        ----------
        std : float
            Standard deviation of the Gaussian kernel in units of time.
        windowsize : float or None, optional
            Duration of the kernel. If None, its length is determined by
            std and size_factor. The kernel length is rounded up to an
            odd number of samples.
        time_units : str, optional
            Units of std and windowsize: 's', 'ms', or 'us'.
        size_factor : int, optional
            Kernel length as a multiple of the standard deviation in
            samples. Default is 100. Ignored when windowsize is provided.
        norm : bool, optional
            Whether to normalise the kernel to sum to one. Default is True.

        Returns
        -------
        TsdTrials
            Smoothed data with the original time axis, validity mask,
            and metadata. The original object is unchanged.

        Notes
        -----
        Smoothing requires a uniformly sampled time axis. Each unit or
        feature is smoothed separately within each trial.

        Invalid bins and samples outside each trial are treated as zero
        during convolution. Invalid bins remain NaN in the result.
        No edge correction is applied, so values near trial boundaries
        may be attenuated. NaNs within valid bins propagate locally.

        Examples
        --------
        Create two trials from a sampled signal:
        >>> import numpy as np
        >>> import pynapple as nap
        >>> source = nap.Tsd(t=np.arange(200) / 100, d=np.sin(np.arange(200)))
        >>> trials = nap.IntervalSet(start=[0.0, 1.0], end=[0.9, 1.9])
        >>> data = nap.TsdTrials(source=source, trials=trials)

        Smooth with a Gaussian standard deviation of 50 ms:
        >>> smoothed = data.smooth(std=50, time_units="ms")

        Specify a kernel duration of 500 ms:
        >>> smoothed = data.smooth(std=50, windowsize=500, time_units="ms")
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
