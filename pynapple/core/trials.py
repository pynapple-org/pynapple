from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit
from numpy.lib.mixins import NDArrayOperatorsMixin
from tabulate import tabulate

from .interval_set import IntervalSet
from .time_index import TsIndex
from .time_series import Ts, Tsd, TsdFrame
from .ts_group import TsGroup

MAX_REPR_TRIALS = 10
MAX_REPR_TRIAL_METADATA = 4
MAX_REPR_UNIT = 8
MAX_REPR_UNIT_METADATA = 4
MAX_REPR_FEATURE = 8
MAX_REPR_FEATURE_METADATA = 4


def _repr_positions(size: int, limit: int) -> list[int | None]:
    """Return head/tail positions separated by an ellipsis."""
    if size <= limit:
        return list(range(size))

    leading = (limit - 1) // 2
    trailing = limit - leading - 1

    return [
        *range(leading),
        None,
        *range(size - trailing, size),
    ]


def _format_bytes(size: int) -> str:
    """Return a compact decimal byte size."""
    units = ("B", "kB", "MB", "GB", "TB")
    value = float(size)

    for unit in units:
        if value < 1000 or unit == units[-1]:
            return f"{value:g}{unit}"
        value /= 1000

    return f"{value:g}TB"


def _get_metadata(obj, index: pd.Index) -> pd.DataFrame:
    """Return object metadata as a DataFrame."""
    metadata = getattr(obj, "_metadata", None)
    if metadata is None:
        return pd.DataFrame(index=index)

    columns = getattr(metadata, "columns", [])
    data = {column: np.asarray(metadata[column]) for column in columns}
    return pd.DataFrame(data, index=index)


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


def _metadata_summary(
    labels: pd.Index,
    metadata: pd.DataFrame | None,
    *,
    label_header: str,
    max_items: int,
    max_metadata: int,
) -> str:
    if metadata is None:
        metadata = pd.DataFrame(index=labels)

    item_positions = _repr_positions(len(labels), max_items)
    metadata_positions = _repr_positions(
        metadata.shape[1],
        max_metadata,
    )

    headers = [
        label_header,
        *[
            "..." if position is None else labels[position]
            for position in item_positions
        ],
    ]

    rows = [
        [
            "Metadata",
            *["..." if position is None else "" for position in item_positions],
        ]
    ]

    for metadata_position in metadata_positions:
        if metadata_position is None:
            rows.append(["..."] * len(headers))
            continue

        column = metadata.columns[metadata_position]
        rows.append(
            [
                column,
                *[
                    (
                        "..."
                        if item_position is None
                        else metadata.iloc[item_position, metadata_position]
                    )
                    for item_position in item_positions
                ],
            ]
        )

    return tabulate(
        rows,
        headers=headers,
        tablefmt="simple",
        floatfmt=".3f",
        numalign="left",
        stralign="left",
    )


class _TrialMetadataMixin:
    trial_ids: pd.Index
    trial_metadata: pd.DataFrame
    durations: np.ndarray
    alignment: str
    window_starts: np.ndarray
    window_ends: np.ndarray

    def query(self, expr: str, **kwargs):
        """Select trials using a pandas metadata query."""
        if not isinstance(expr, str):
            raise TypeError("expr must be a string.")
        if "inplace" in kwargs:
            raise TypeError("query does not support the inplace argument.")
        kwargs["level"] = kwargs.get("level", 0) + 1
        selected = self.trial_metadata.query(
            expr,
            inplace=False,
            **kwargs,
        )
        return self._select_trials(selected.index)

    def groupby(self, by: str):
        """Group trials by a metadata column."""
        if by not in self.trial_metadata.columns:
            raise KeyError(f"Unknown trial metadata column: {by!r}")

        groups = self.trial_metadata.groupby(
            by,
            sort=False,
        ).groups

        return {key: self._select_trials(index) for key, index in groups.items()}

    def _trial_metadata_positions(self) -> list[int | None]:
        """Return metadata columns shown in representations."""
        return _repr_positions(
            self.trial_metadata.shape[1],
            MAX_REPR_TRIAL_METADATA,
        )

    def _trial_positions(self, trial_ids) -> np.ndarray:
        """Convert trial labels to positional indices."""
        trial_ids = pd.Index(trial_ids)
        positions = self.trial_ids.get_indexer(trial_ids)
        if np.any(positions < 0):
            missing = trial_ids[positions < 0].tolist()
            raise KeyError(f"Trials not found: {missing}")
        return positions

    def _select_trials(self, trial_ids):
        """Select trials by label."""
        return self._select_trial_positions(self._trial_positions(trial_ids))

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
        self.trial_ids = pd.Index(trials.index)
        self.trial_metadata = _get_metadata(trials, self.trial_ids)

        starts = np.asarray(trials.start, dtype=float)
        ends = np.asarray(trials.end, dtype=float)

        self.alignment = "start"
        self.window_starts = np.zeros(len(trials), dtype=float)
        self.window_ends = ends - starts
        self.durations = self.window_ends - self.window_starts

        self.unit_keys = np.asarray(source.index)
        unit_index = pd.Index(self.unit_keys)
        self.unit_metadata = _get_metadata(
            source,
            unit_index,
        ).drop(columns="rate", errors="ignore")

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
        trial_ids: pd.Index,
        trial_metadata: pd.DataFrame,
        unit_keys: np.ndarray,
        unit_metadata: pd.DataFrame,
        alignment: str,
        sort: bool = False,
    ) -> TsTrials:
        spikes = np.asarray(spikes)
        trial_index = np.asarray(trial_index, dtype=np.int64)
        neuron_index = np.asarray(neuron_index, dtype=np.int64)

        if sort and len(spikes):
            order = np.argsort(spikes, kind="stable")
            spikes = spikes[order]
            trial_index = trial_index[order]
            neuron_index = neuron_index[order]

        obj = cls.__new__(cls)
        obj.spikes = spikes
        obj.trial_index = trial_index
        obj.neuron_index = neuron_index
        obj.durations = np.asarray(durations, dtype=float)
        obj.window_starts = np.asarray(window_starts, dtype=float)
        obj.window_ends = np.asarray(window_ends, dtype=float)
        obj.trials = trials
        obj.trial_ids = pd.Index(trial_ids)
        obj.trial_metadata = trial_metadata
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
        events = source.to_tsd()
        assignments = trials.in_interval(events)

        absolute_times = np.asarray(events.t, dtype=float)
        unit_labels = np.asarray(events.d)
        assigned_trials = np.asarray(assignments, dtype=float)

        inside = np.isfinite(assigned_trials)

        absolute_times = absolute_times[inside]
        unit_labels = unit_labels[inside]
        trial_index = assigned_trials[inside].astype(
            np.int64,
            copy=False,
        )

        neuron_index = pd.Index(unit_keys).get_indexer(unit_labels)
        if np.any(neuron_index < 0):
            raise RuntimeError("Could not map all spike unit labels.")

        trial_starts = np.asarray(trials.start, dtype=float)
        spikes = absolute_times - trial_starts[trial_index]

        order = np.argsort(spikes, kind="stable")
        return (
            spikes[order],
            trial_index[order],
            neuron_index[order].astype(np.int64, copy=False),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.trial_ids), len(self.unit_keys)

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        """Return compact trial and unit summaries."""
        return (
            f"{self._trial_summary()}\n\n"
            f"{self._unit_summary()}\n\n"
            f"align: {self.alignment!r}, "
            f"{self._window_label()}: {self._window_summary()}\n"
            f"unbinned, shape: {self.shape} (trial, unit)"
        )

    def _trial_summary(self) -> str:
        """Return a compact trial summary table."""
        spike_counts = np.bincount(
            self.trial_index,
            minlength=self.shape[0],
        )

        denominators = self.durations * self.shape[1]
        rates = np.divide(
            spike_counts,
            denominators,
            out=np.zeros(len(self), dtype=float),
            where=denominators > 0,
        )

        trial_positions = _repr_positions(
            len(self),
            MAX_REPR_TRIALS,
        )
        metadata_positions = self._trial_metadata_positions()

        metadata_headers = [
            "..." if position is None else str(self.trial_metadata.columns[position])
            for position in metadata_positions
        ]
        headers = [
            "Trial",
            "start",
            "end",
            *metadata_headers,
            "n spikes",
            "rate (Hz)",
        ]

        rows = []
        for position in trial_positions:
            if position is None:
                rows.append(["..."] * len(headers))
                continue

            metadata_values = [
                (
                    "..."
                    if metadata_position is None
                    else self.trial_metadata.iloc[
                        position,
                        metadata_position,
                    ]
                )
                for metadata_position in metadata_positions
            ]

            rows.append(
                [
                    self.trial_ids[position],
                    self.trials.start[position],
                    self.trials.end[position],
                    *metadata_values,
                    spike_counts[position],
                    rates[position],
                ]
            )

        return tabulate(
            rows,
            headers=headers,
            tablefmt="simple",
            floatfmt=".3f",
            numalign="left",
            stralign="left",
        )

    def _unit_summary(self) -> str:
        """Return a compact unit metadata table."""
        return _metadata_summary(
            pd.Index(self.unit_keys),
            self.unit_metadata,
            label_header="Unit",
            max_items=MAX_REPR_UNIT,
            max_metadata=MAX_REPR_UNIT_METADATA,
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
        spike_chunks = []
        trial_chunks = []
        neuron_chunks = []

        for new_trial, trial in enumerate(trial_positions):
            trial_selected = self.trial_index == trial

            for new_neuron, neuron in enumerate(unit_positions):
                selected = trial_selected & (self.neuron_index == neuron)
                spikes = self.spikes[selected]

                if not len(spikes):
                    continue

                spike_chunks.append(spikes)
                trial_chunks.append(
                    np.full(
                        len(spikes),
                        new_trial,
                        dtype=np.int64,
                    )
                )
                neuron_chunks.append(
                    np.full(
                        len(spikes),
                        new_neuron,
                        dtype=np.int64,
                    )
                )

        if spike_chunks:
            spikes = np.concatenate(spike_chunks)
            trial_index = np.concatenate(trial_chunks)
            neuron_index = np.concatenate(neuron_chunks)
        else:
            spikes = np.array([], dtype=self.spikes.dtype)
            trial_index = np.array([], dtype=np.int64)
            neuron_index = np.array([], dtype=np.int64)

        trial_ids = self.trial_ids.take(trial_positions)
        unit_keys = self.unit_keys[unit_positions]
        unit_index = pd.Index(unit_keys)

        return self._from_events(
            spikes=spikes,
            trial_index=trial_index,
            neuron_index=neuron_index,
            durations=self.durations[trial_positions],
            window_starts=self.window_starts[trial_positions],
            window_ends=self.window_ends[trial_positions],
            trials=self.trials[trial_positions],
            trial_ids=trial_ids,
            trial_metadata=self.trial_metadata.loc[trial_ids],
            unit_keys=unit_keys,
            unit_metadata=self.unit_metadata.loc[unit_index],
            alignment=self.alignment,
            sort=True,
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
        """Return trials aligned to absolute timestamps in a metadata column."""
        if column not in self.trial_metadata.columns:
            raise KeyError(f"Unknown trial metadata column: {column!r}")

        series = self.trial_metadata[column]
        if not pd.api.types.is_numeric_dtype(
            series.dtype
        ) or pd.api.types.is_bool_dtype(series.dtype):
            raise TypeError(f"Trial metadata column {column!r} must be numeric.")

        alignment = series.to_numpy(dtype=float)
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
            trial_ids=self.trial_ids.copy(),
            trial_metadata=self.trial_metadata.copy(),
            unit_keys=self.unit_keys.copy(),
            unit_metadata=self.unit_metadata.copy(),
            alignment=column,
            sort=True,
        )

    def to_tsdframe(self) -> TsdFrame:
        """Return all aligned spikes as a two-column event table."""
        trial_labels = self.trial_ids.to_numpy()[self.trial_index]
        unit_labels = self.unit_keys[self.neuron_index]
        values = np.column_stack([trial_labels, unit_labels])

        if len(self.durations):
            support = IntervalSet(
                start=float(self.window_starts.min()),
                end=float(self.window_ends.max()),
            )
        else:
            support = IntervalSet([], [])

        return TsdFrame(
            t=self.spikes.copy(),
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
            trial_ids=self.trial_ids.copy(),
            trial_metadata=self.trial_metadata.copy(),
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
                trial_ids=self.trial_ids,
                trial_metadata=self.trial_metadata,
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
            trial_ids=self.trial_ids.copy(),
            trial_metadata=self.trial_metadata.copy(),
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
        self.trial_ids = pd.Index(trials.index)
        self.trial_metadata = _get_metadata(
            trials,
            self.trial_ids,
        )
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
            self.feature_metadata = _get_metadata(
                source,
                self.feature_names,
            )

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
        trial_ids: pd.Index,
        trial_metadata: pd.DataFrame,
        feature_names: pd.Index | None,
        feature_metadata: pd.DataFrame | None,
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
        obj.trial_ids = pd.Index(trial_ids)
        obj.trial_metadata = trial_metadata
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
        """Return compact trial and feature summaries."""
        sampling_label = (
            "bin_size" if self.data_kind == "count" else "sampling interval"
        )
        return (
            f"{self._trial_summary()}\n\n"
            f"{self._feature_summary()}\n\n"
            f"align: {self.alignment!r}, "
            f"{self._window_label()}: {self._window_summary()}, "
            f"{sampling_label}: {self._sample_size_summary()}\n"
            f"dtype: {self.values.dtype}, "
            f"shape: {self.shape} {self._shape_axes()}, "
            f"size: {_format_bytes(self.values.nbytes)}"
        )

    def _trial_summary(self) -> str:
        """Return a compact per-trial summary."""
        trial_positions = _repr_positions(
            len(self),
            MAX_REPR_TRIALS,
        )
        metadata_positions = self._trial_metadata_positions()

        metadata_headers = [
            "..." if position is None else str(self.trial_metadata.columns[position])
            for position in metadata_positions
        ]

        value_header = "rate (Hz)" if self.data_kind == "count" else "mean"
        headers = [
            "Trial",
            "start",
            "end",
            *metadata_headers,
            value_header,
            "std",
        ]

        means, standard_deviations = self._trial_statistics()
        rows = []

        for position in trial_positions:
            if position is None:
                rows.append(["..."] * len(headers))
                continue

            metadata_values = [
                (
                    "..."
                    if metadata_position is None
                    else self.trial_metadata.iloc[
                        position,
                        metadata_position,
                    ]
                )
                for metadata_position in metadata_positions
            ]

            rows.append(
                [
                    self.trial_ids[position],
                    self.trials.start[position],
                    self.trials.end[position],
                    *metadata_values,
                    means[position],
                    standard_deviations[position],
                ]
            )

        return tabulate(
            rows,
            headers=headers,
            tablefmt="simple",
            floatfmt=".3f",
            numalign="left",
            stralign="left",
        )

    def _trial_statistics(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-trial mean and standard deviation."""
        values = self.values.astype(float, copy=False)

        valid = self.valid
        if values.ndim == 3:
            valid = np.broadcast_to(
                valid[:, :, None],
                values.shape,
            )

        valid = valid & np.isfinite(values)
        scaled = values / self.bin_size if self.data_kind == "count" else values

        axes = tuple(range(1, values.ndim))
        count = valid.sum(axis=axes)
        total = np.where(valid, scaled, 0.0).sum(axis=axes)

        mean = np.divide(
            total,
            count,
            out=np.full(len(self), np.nan),
            where=count > 0,
        )

        mean_shape = (len(self),) + (1,) * (values.ndim - 1)
        squared_error = np.where(
            valid,
            (scaled - mean.reshape(mean_shape)) ** 2,
            0.0,
        ).sum(axis=axes)

        variance = np.divide(
            squared_error,
            count,
            out=np.full(len(self), np.nan),
            where=count > 0,
        )

        return mean, np.sqrt(variance)

    def _feature_summary(self) -> str:
        """Return a compact feature metadata table."""
        if self.feature_names is None:
            return "Feature\n-------\nscalar"

        return _metadata_summary(
            self.feature_names,
            self.feature_metadata,
            label_header="Feature",
            max_items=MAX_REPR_FEATURE,
            max_metadata=MAX_REPR_FEATURE_METADATA,
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

        trial_ids = self.trial_ids.take(trial_positions)

        return self._from_arrays(
            values=values,
            time=self.t[time_positions],
            valid=valid,
            durations=self.durations[trial_positions],
            window_starts=self.window_starts[trial_positions],
            window_ends=self.window_ends[trial_positions],
            trials=self.trials[trial_positions],
            trial_ids=trial_ids,
            trial_metadata=self.trial_metadata.loc[trial_ids],
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
            trial_ids=self.trial_ids,
            trial_metadata=self.trial_metadata,
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
            trial_ids=self.trial_ids.copy(),
            trial_metadata=self.trial_metadata.copy(),
            feature_names=self.feature_names,
            feature_metadata=self.feature_metadata,
            sample_period=self.sample_period,
            bin_size=self.bin_size,
            data_kind=self.data_kind,
            alignment=self.alignment,
        )
