from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.mixins import NDArrayOperatorsMixin

from .interval_set import IntervalSet
from .time_index import TsIndex
from .time_series import Ts, Tsd, TsdFrame
from .ts_group import TsGroup


class _TrialMetadataMixin:
    trial_ids: pd.Index
    trial_metadata: pd.DataFrame

    def groupby(self, by: str, get_group=None):
        """
        Group trials by a metadata column.

        Parameters
        ----------
        by : str
            Name of the trial metadata column used for grouping.
        get_group : object, optional
            Return only this group. If omitted, return every group.

        Returns
        -------
        dict or TsTrials or TsdTrials
            A dictionary mapping metadata values to trial subsets, or one
            subset when ``get_group`` is specified.

        Raises
        ------
        KeyError
            If ``by`` is not a metadata column or ``get_group`` is unknown.

        Examples
        --------
        Group trials by outcome:

        >>> groups = trials.groupby("outcome")
        >>> hit_trials = groups["hit"]

        Return one group directly:

        >>> hit_trials = trials.groupby("outcome", get_group="hit")
        """
        if by not in self.trial_metadata.columns:
            raise KeyError(f"Unknown trial metadata column: {by!r}")

        groups = self.trial_metadata.groupby(by, sort=False).groups

        if get_group is not None:
            if get_group not in groups:
                raise KeyError(get_group)
            return self._select_trials(groups[get_group])

        return {key: self._select_trials(index) for key, index in groups.items()}


class TsTrials(_TrialMetadataMixin):
    """
    Collection of trial-aligned time series.

    Parameters
    ----------
    source : TsGroup
        Timestamp data for each unit.
    trials : IntervalSet
        Absolute start and end times of the trials.
    alignment_times : numpy.ndarray
        Absolute alignment time for each trial. Returned timestamps are
        expressed relative to these times.
    trial_ids : pandas.Index, optional
        Unique trial identifiers. Defaults to a consecutive integer index.
    trial_metadata : pandas.DataFrame, optional
        Metadata indexed by ``trial_ids``.
    unit_keys : numpy.ndarray, optional
        Keys from ``source`` to include. By default, all units are included.

    Attributes
    ----------
    source : TsGroup
        Original timestamp source.
    trials : IntervalSet
        Absolute trial intervals.
    alignment_times : numpy.ndarray
        Absolute alignment time for each trial.
    trial_ids : pandas.Index
        Trial identifiers.
    trial_metadata : pandas.DataFrame
        Metadata associated with the trials.
    unit_keys : numpy.ndarray
        Keys of the selected units.
    unit_metadata : pandas.DataFrame
        Metadata associated with the selected units.

    Notes
    -----
    The shape is ``(n_trials, n_units)``. Each cell conceptually contains one
    exact spike train, but cells are not materialized during construction.

    Selecting one trial returns a :class:`TsGroup` whose timestamps and time
    support are relative to that trial's alignment time. Selecting one trial
    and one unit returns a :class:`Ts`.

    Examples
    --------
    Create trial-aligned spike data:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import pynapple as nap
    >>> source = nap.TsGroup({
    ...     0: nap.Ts([0.1, 0.4, 1.2, 2.2]),
    ...     1: nap.Ts([0.2, 0.8, 1.5, 2.5]),
    ... })
    >>> epochs = nap.IntervalSet(start=[0.0, 1.0], end=[1.0, 3.0])
    >>> metadata = pd.DataFrame(
    ...     {"outcome": ["hit", "miss"]},
    ...     index=["trial_1", "trial_2"],
    ... )
    >>> trials = TsTrials(
    ...     source,
    ...     epochs,
    ...     alignment_times=np.array([0.0, 1.0]),
    ...     trial_metadata=metadata,
    ... )

    Select one trial:

    >>> trial = trials[0]
    >>> isinstance(trial, nap.TsGroup)
    True

    Select one exact spike train:

    >>> spikes = trials[0, 0]
    >>> isinstance(spikes, nap.Ts)
    True

    Trial-axis slicing stays lazy:

    >>> subset = trials[:1]
    >>> isinstance(subset, TsTrials)
    True

    Bin timestamps into a dense representation:

    >>> dense = trials.bin(0.1)
    >>> isinstance(dense, TsdTrials)
    True
    """

    def __init__(
        self,
        source: TsGroup,
        trials: IntervalSet,
        alignment_times: np.ndarray,
        *,
        trial_ids: pd.Index | None = None,
        trial_metadata: pd.DataFrame | None = None,
        unit_keys: np.ndarray | None = None,
    ):
        if not isinstance(source, TsGroup):
            raise TypeError("source must be a TsGroup.")
        if not isinstance(trials, IntervalSet):
            raise TypeError("trials must be an IntervalSet.")

        alignment_times = np.asarray(alignment_times, dtype=float)
        if alignment_times.ndim != 1 or len(alignment_times) != len(trials):
            raise ValueError("alignment_times must contain one value per trial.")

        if trial_ids is None:
            if trial_metadata is not None:
                trial_ids = pd.Index(trial_metadata.index)
            else:
                trial_ids = pd.RangeIndex(len(trials))
        else:
            trial_ids = pd.Index(trial_ids)

        if len(trial_ids) != len(trials):
            raise ValueError("trial_ids must contain one value per trial.")
        if not trial_ids.is_unique:
            raise ValueError("trial_ids must be unique.")

        if trial_metadata is None:
            trial_metadata = pd.DataFrame(index=trial_ids)
        else:
            trial_metadata = trial_metadata.copy()
            if len(trial_metadata) != len(trials):
                raise ValueError("trial_metadata must contain one row per trial.")
            if not trial_metadata.index.equals(trial_ids):
                raise ValueError("trial_metadata index must match trial_ids.")

        if unit_keys is None:
            unit_keys = np.asarray(source.index)
        else:
            unit_keys = np.asarray(unit_keys)
            missing = unit_keys[~np.isin(unit_keys, source.index)]
            if len(missing):
                raise KeyError(f"Units not in source: {missing.tolist()}")

        source_metadata = source._metadata.loc[unit_keys].copy()
        self.unit_metadata = source_metadata.drop(
            columns="rate",
            errors="ignore",
        )

        self.source = source
        self.trials = trials
        self.alignment_times = alignment_times
        self.trial_ids = trial_ids
        self.trial_metadata = trial_metadata
        self.unit_keys = unit_keys

    @property
    def shape(self) -> tuple[int, int]:
        """
        Number of trials and units.

        Returns
        -------
        tuple of int
            ``(n_trials, n_units)``.
        """
        return len(self.trials), len(self.unit_keys)

    def __len__(self) -> int:
        return len(self.trials)

    def __repr__(self) -> str:
        return (
            f"TsTrials(shape={self.shape}, "
            f"trials={self.shape[0]}, units={self.shape[1]})"
        )

    def __getitem__(self, key):
        """
        Select trials and units by position.

        Parameters
        ----------
        key : int, slice, array-like or tuple
            Positional index for the trial axis, or a ``(trial, unit)`` index.

        Returns
        -------
        TsTrials, TsGroup or Ts
            - Multiple trials return a lazy ``TsTrials``.
            - One trial returns a ``TsGroup``.
            - One trial and one unit return a ``Ts``.

        Notes
        -----
        Timestamps returned for one trial are relative to that trial's alignment time.

        Examples
        --------

        Select one trial:
        >>> trial = trials[0]

        Select one unit from one trial:
        >>> spikes = trials[0, 1]

        Select several trials and one unit:
        >>> subset = trials[:3, 0]
        """
        trial_key, unit_key = self._split_key(key)
        trial_positions, trial_scalar = self._normalize_positions(
            trial_key,
            len(self.trials),
            "trial",
        )
        unit_positions, unit_scalar = self._normalize_positions(
            unit_key,
            len(self.unit_keys),
            "unit",
        )

        if trial_scalar:
            group = self._get_trial(trial_positions[0])
            selected_keys = self.unit_keys[unit_positions]

            if unit_scalar:
                return group[selected_keys[0]]

            return group[selected_keys]

        return self._subset(trial_positions, unit_positions)

    def _split_key(self, key) -> tuple[object, object]:
        if not isinstance(key, tuple):
            return key, slice(None)

        if len(key) > 2:
            raise IndexError("TsTrials supports trial and unit axes only.")
        if not key:
            return slice(None), slice(None)
        if len(key) == 1:
            return key[0], slice(None)

        return key

    @staticmethod
    def _normalize_positions(
        key,
        size: int,
        axis_name: str,
    ) -> tuple[np.ndarray, bool]:
        if key is Ellipsis:
            key = slice(None)

        if isinstance(key, (int, np.integer)):
            position = int(key)
            if position < 0:
                position += size
            if not 0 <= position < size:
                raise IndexError(f"{axis_name} index {key} is out of bounds.")
            return np.array([position], dtype=int), True

        positions = np.arange(size)[key]
        positions = np.asarray(positions)

        if positions.ndim == 0:
            return np.array([int(positions)], dtype=int), True
        if positions.ndim != 1:
            raise IndexError(f"{axis_name} indices must be one-dimensional.")

        return positions.astype(int, copy=False), False

    def _get_trial(self, position: int) -> TsGroup:
        start = self.trials.start[position]
        end = self.trials.end[position]
        alignment = self.alignment_times[position]

        absolute_support = IntervalSet(start=start, end=end)
        relative_support = IntervalSet(
            start=start - alignment,
            end=end - alignment,
        )

        data = {}
        for unit_key in self.unit_keys:
            restricted = self.source[unit_key].restrict(absolute_support)
            data[unit_key] = Ts(
                t=restricted.t - alignment,
                time_support=relative_support,
            )

        metadata = self.unit_metadata.loc[self.unit_keys]
        if metadata.shape[1] == 0:
            metadata = None

        return TsGroup(
            data,
            time_support=relative_support,
            bypass_check=True,
            metadata=metadata,
        )

    def _subset(
        self,
        trial_positions: np.ndarray,
        unit_positions: np.ndarray,
    ) -> TsTrials:
        trial_ids = self.trial_ids.take(trial_positions)
        unit_keys = self.unit_keys[unit_positions]

        return TsTrials(
            source=self.source,
            trials=IntervalSet(
                start=self.trials.start[trial_positions],
                end=self.trials.end[trial_positions],
            ),
            alignment_times=self.alignment_times[trial_positions],
            trial_ids=trial_ids,
            trial_metadata=self.trial_metadata.loc[trial_ids],
            unit_keys=unit_keys,
        )

    def _select_trials(self, trial_ids) -> TsTrials:
        trial_ids = pd.Index(trial_ids)
        positions = self.trial_ids.get_indexer(trial_ids)

        if np.any(positions < 0):
            missing = trial_ids[positions < 0].tolist()
            raise KeyError(f"Trials not found: {missing}")

        return self._subset(
            positions,
            np.arange(len(self.unit_keys)),
        )

    def bin(
        self,
        bin_size: float,
        *,
        time_units: str = "s",
        dtype: np.dtype | type = np.float64,
    ) -> TsdTrials:
        """
        Count timestamps in bins aligned across trials.

        Bins are anchored to the common relative alignment time. Only bins
        fully contained within a trial are valid. Bins outside a trial are
        padded with NaN.

        Parameters
        ----------
        bin_size : float
            Width of each time bin.
        time_units : str, optional
            Units of ``bin_size``: ``"s"``, ``"ms"`` or ``"us"``.
            Defaults to seconds.
        dtype : numpy.dtype or type, optional
            Floating-point dtype of the dense output. A floating-point dtype is
            required because invalid bins are represented by NaN.

        Returns
        -------
        TsdTrials
            Dense spike counts with shape
            ``(n_trials, n_time_bins, n_units)``.

        Raises
        ------
        TypeError
            If ``bin_size`` is not numeric or ``dtype`` is not floating-point.
        ValueError
            If ``bin_size`` is not positive or ``time_units`` is invalid.

        Examples
        --------
        Count spikes in 10 ms bins:

        >>> dense = trials.bin(10, time_units="ms")
        >>> dense.shape
        (2, 200, 2)

        Select one binned trial:

        >>> trial = dense[0]
        >>> isinstance(trial, nap.TsdFrame)
        True

        Compute the population PSTH:

        >>> psth = dense.psth()
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
                "dtype must be floating-point because padded bins contain NaN."
            )

        n_trials, n_units = self.shape
        relative_starts = self.trials.start - self.alignment_times
        relative_ends = self.trials.end - self.alignment_times

        if n_trials == 0:
            return TsdTrials(
                values=np.empty((0, 0, n_units), dtype=dtype),
                time=np.array([]),
                trials=self.trials,
                valid=np.empty((0, 0), dtype=bool),
                trial_ids=self.trial_ids,
                trial_metadata=self.trial_metadata,
                feature_names=pd.Index(self.unit_keys),
                feature_metadata=self.unit_metadata,
                bin_size=bin_size,
                data_kind="count",
            )

        first_edge = np.floor(relative_starts.min() / bin_size) * bin_size
        last_edge = np.ceil(relative_ends.max() / bin_size) * bin_size
        n_bins = int(np.round((last_edge - first_edge) / bin_size))

        edges = first_edge + np.arange(n_bins + 1) * bin_size
        time = edges[:-1] + bin_size / 2

        valid = (edges[:-1][None, :] >= relative_starts[:, None]) & (
            edges[1:][None, :] <= relative_ends[:, None]
        )
        values = np.full(
            (n_trials, n_bins, n_units),
            np.nan,
            dtype=dtype,
        )

        for trial in range(n_trials):
            absolute_edges = edges + self.alignment_times[trial]

            for unit, unit_key in enumerate(self.unit_keys):
                timestamps = self.source[unit_key].t
                positions = np.searchsorted(
                    timestamps,
                    absolute_edges,
                    side="left",
                )
                counts = np.diff(positions)
                values[trial, valid[trial], unit] = counts[valid[trial]]

        return TsdTrials(
            values=values,
            time=time,
            trials=self.trials,
            valid=valid,
            trial_ids=self.trial_ids,
            trial_metadata=self.trial_metadata,
            feature_names=pd.Index(self.unit_keys),
            feature_metadata=self.unit_metadata,
            bin_size=bin_size,
            data_kind="count",
        )


class TsdTrials(_TrialMetadataMixin, NDArrayOperatorsMixin):
    """
    Collection of trial-aligned time series with data.

    ``TsdTrials`` stores data with trial and relative-time axes. Optional
    features occupy the trailing axis, following the convention used by
    :class:`TsdFrame`.

    Two-dimensional data have shape ``(trial, time)``. Three-dimensional data
    have shape ``(trial, time, feature)``.

    Parameters
    ----------
    values : numpy.ndarray
        Dense values with shape ``(trial, time)`` or
        ``(trial, time, feature)``.
    time : numpy.ndarray
        Shared relative time index.
    trials : IntervalSet
        Absolute start and end times of the trials.
    valid : numpy.ndarray
        Boolean array with shape ``(trial, time)``. True indicates that the
        corresponding sample or bin is valid for that trial.
    alignment_times : numpy.ndarray, optional
        Absolute alignment time for each trial. Defaults to the start of each
        trial.
    trial_ids : pandas.Index, optional
        Unique trial identifiers.
    trial_metadata : pandas.DataFrame, optional
        Metadata indexed by ``trial_ids``.
    feature_names : pandas.Index, optional
        Names of the trailing features for three-dimensional data.
    feature_metadata : pandas.DataFrame, optional
        Metadata indexed by ``feature_names``.
    bin_size : float, optional
        Bin width in seconds. Required when ``data_kind="count"``.
    data_kind : {"continuous", "count"}, optional
        Kind of values stored. Count data are divided by ``bin_size`` when
        :meth:`psth` is called with ``rate=True``.

    Attributes
    ----------
    values : numpy.ndarray
        Dense trial data.
    time : TsIndex
        Shared relative time index.
    trials : IntervalSet
        Absolute trial intervals.
    valid : numpy.ndarray
        Validity mask over trial and time.
    alignment_times : numpy.ndarray
        Absolute alignment times.
    trial_ids : pandas.Index
        Trial identifiers.
    trial_metadata : pandas.DataFrame
        Trial metadata.
    feature_names : pandas.Index or None
        Feature names for three-dimensional data.
    feature_metadata : pandas.DataFrame or None
        Feature metadata.
    bin_size : float or None
        Bin width in seconds.
    data_kind : str
        Either ``"continuous"`` or ``"count"``.

    Notes
    -----
    Padding is described by ``valid`` rather than inferred from NaN values.
    This allows genuine NaNs in the underlying signal to remain distinct from
    trial padding.

    Selecting one trial removes padded samples and returns an ordinary
    :class:`Tsd` or :class:`TsdFrame`. Selecting multiple trials returns another
    ``TsdTrials``.

    Examples
    --------
    Construct dense trial-aligned data:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import pynapple as nap
    >>> values = np.arange(24, dtype=float).reshape(2, 4, 3)
    >>> valid = np.array([
    ...     [True, True, True, False],
    ...     [True, True, True, True],
    ... ])
    >>> epochs = nap.IntervalSet(start=[0.0, 10.0], end=[3.0, 14.0])
    >>> dense = TsdTrials(
    ...     values=values,
    ...     time=np.array([0.5, 1.5, 2.5, 3.5]),
    ...     trials=epochs,
    ...     valid=valid,
    ...     alignment_times=np.array([0.0, 10.0]),
    ...     feature_names=pd.Index(["A", "B", "C"]),
    ... )

    Select one trial as a Pynapple time series:

    >>> trial = dense[0]
    >>> isinstance(trial, nap.TsdFrame)
    True
    >>> trial.shape
    (3, 3)

    Select one feature from one trial:

    >>> feature = dense[0, :, 0]
    >>> isinstance(feature, nap.Tsd)
    True

    Keep one feature across all trials:

    >>> subset = dense[:, :, 0]
    >>> isinstance(subset, TsdTrials)
    True
    >>> subset.shape
    (2, 4)

    Access the dense array:

    >>> np.asarray(dense).shape
    (2, 4, 3)
    """

    __array_priority__ = 1000

    def __init__(
        self,
        values: np.ndarray,
        time: np.ndarray,
        trials: IntervalSet,
        valid: np.ndarray,
        *,
        alignment_times: np.ndarray | None = None,
        trial_ids: pd.Index | None = None,
        trial_metadata: pd.DataFrame | None = None,
        feature_names: pd.Index | None = None,
        feature_metadata: pd.DataFrame | None = None,
        bin_size: float | None = None,
        data_kind: str = "continuous",
    ):
        values = np.asarray(values)
        valid = np.asarray(valid, dtype=bool)
        time = np.asarray(time, dtype=float)

        if values.ndim not in (2, 3):
            raise ValueError(
                "values must have shape (trial, time) or (trial, time, feature)."
            )
        if not isinstance(trials, IntervalSet):
            raise TypeError("trials must be an IntervalSet.")
        if values.shape[:2] != valid.shape:
            raise ValueError("valid must match the trial and time axes.")
        if values.shape[0] != len(trials):
            raise ValueError("Trial count does not match trials.")
        if values.shape[1] != len(time):
            raise ValueError("Time axis does not match values.")
        if data_kind not in {"continuous", "count"}:
            raise ValueError("data_kind must be 'continuous' or 'count'.")
        if data_kind == "count" and bin_size is None:
            raise ValueError("Count data requires bin_size.")

        if alignment_times is None:
            alignment_times = trials.start.copy()
        else:
            alignment_times = np.asarray(alignment_times, dtype=float)

        if alignment_times.shape != (len(trials),):
            raise ValueError("alignment_times must contain one value per trial.")

        if trial_ids is None:
            trial_ids = pd.RangeIndex(len(trials))
        else:
            trial_ids = pd.Index(trial_ids)

        if len(trial_ids) != len(trials):
            raise ValueError("trial_ids must contain one value per trial.")
        if not trial_ids.is_unique:
            raise ValueError("trial_ids must be unique.")

        if trial_metadata is None:
            trial_metadata = pd.DataFrame(index=trial_ids)
        else:
            trial_metadata = trial_metadata.copy()
            if len(trial_metadata) != len(trials):
                raise ValueError("trial_metadata must contain one row per trial.")
            if not trial_metadata.index.equals(trial_ids):
                raise ValueError("trial_metadata index must match trial_ids.")

        if values.ndim == 3:
            n_features = values.shape[2]

            if feature_names is None:
                feature_names = pd.RangeIndex(n_features)
            else:
                feature_names = pd.Index(feature_names)

            if len(feature_names) != n_features:
                raise ValueError("feature_names must contain one name per feature.")
            if not feature_names.is_unique:
                raise ValueError("feature_names must be unique.")

            if feature_metadata is None:
                feature_metadata = pd.DataFrame(index=feature_names)
            else:
                feature_metadata = feature_metadata.copy()
                if len(feature_metadata) != n_features:
                    raise ValueError(
                        "feature_metadata must contain one row per feature."
                    )
                if not feature_metadata.index.equals(feature_names):
                    raise ValueError("feature_metadata index must match feature_names.")
        elif feature_names is not None or feature_metadata is not None:
            raise ValueError(
                "Feature metadata is only valid for three-dimensional data."
            )

        self.values = values
        self.time = TsIndex(time)
        self.trials = trials
        self.valid = valid
        self.alignment_times = alignment_times
        self.trial_ids = trial_ids
        self.trial_metadata = trial_metadata
        self.feature_names = feature_names
        self.feature_metadata = feature_metadata
        self.bin_size = bin_size
        self.data_kind = data_kind

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Shape of the dense trial data.
        Returns
        -------
        tuple of int
            ``(trial, time)`` or ``(trial, time, feature)``.
        """
        return self.values.shape

    @property
    def ndim(self) -> int:
        """
        Number of dimensions in the dense trial data.

        Returns
        -------
        int
            Two for scalar data or three for data with features.
        """
        return self.values.ndim

    @property
    def t(self) -> np.ndarray:
        """
        Shared relative time index.

        Returns
        -------
        numpy.ndarray
            Relative sample or bin times.
        """
        return self.time.values

    @property
    def d(self) -> np.ndarray:
        """
        Dense trial data.

        Returns
        -------
        numpy.ndarray
            Underlying values.
        """
        return self.values

    @property
    def relative_support(self) -> IntervalSet:
        """
        Trial intervals relative to their alignment times.

        Returns
        -------
        IntervalSet
            One relative interval per trial.
        """
        return IntervalSet(
            start=self.trials.start - self.alignment_times,
            end=self.trials.end - self.alignment_times,
        )

    def __len__(self) -> int:
        return self.values.shape[0]

    def __repr__(self) -> str:
        padding = 100.0 * (1.0 - self.valid.mean()) if self.valid.size else 0.0
        return (
            f"TsdTrials(shape={self.shape}, "
            f"dtype={self.values.dtype}, padding={padding:.1f}%)"
        )

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        values = np.asarray(self.values, dtype=dtype)
        if copy:
            values = values.copy()
        return values

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or "out" in kwargs:
            return NotImplemented

        arrays = []
        for value in inputs:
            if isinstance(value, TsdTrials):
                if (
                    value.shape != self.shape
                    or not np.array_equal(value.t, self.t)
                    or not np.array_equal(value.valid, self.valid)
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
        """
        Select trials, times and features by position.

        Parameters
        ----------
        key : int, slice, array-like or tuple
            Positional index over ``(trial, time, feature)``. Two-dimensional
            data accept only trial and time indices.

        Returns
        -------
        TsdTrials, TsdFrame, Tsd, numpy.ndarray or scalar
            - Multiple trials with a time axis return ``TsdTrials``.
            - One trial with multiple features returns ``TsdFrame``.
            - One trial and one feature return ``Tsd``.
            - Selecting one time point returns a NumPy value or array.

        Notes
        -----
        Selecting one trial removes samples for which ``valid`` is false.

        Examples
        --------
        Select one trial:

        >>> trial = dense[0]

        Select one feature from one trial:

        >>> feature = dense[0, :, 0]

        Select one feature across all trials:

        >>> subset = dense[:, :, 0]

        Select one time point across trials:

        >>> values = dense[:, 2, :]
        """

        trial_key, time_key, feature_key = self._split_key(key)

        trial_positions, trial_scalar = TsTrials._normalize_positions(
            trial_key,
            len(self),
            "trial",
        )
        time_positions, time_scalar = TsTrials._normalize_positions(
            time_key,
            self.shape[1],
            "time",
        )

        if self.ndim == 3:
            feature_positions, feature_scalar = TsTrials._normalize_positions(
                feature_key,
                self.shape[2],
                "feature",
            )
        else:
            if feature_key != slice(None):
                raise IndexError("Two-dimensional TsdTrials has no feature axis.")
            feature_positions = None
            feature_scalar = False

        values = np.take(self.values, trial_positions, axis=0)
        values = np.take(values, time_positions, axis=1)
        valid = np.take(self.valid, trial_positions, axis=0)
        valid = np.take(valid, time_positions, axis=1)

        if feature_positions is not None:
            values = np.take(values, feature_positions, axis=2)

        if time_scalar:
            values = values[:, 0]
            if trial_scalar:
                values = values[0]
            if feature_scalar and isinstance(values, np.ndarray):
                values = np.squeeze(values, axis=-1)
            return values

        if trial_scalar:
            values = values[0]
            valid = valid[0]

            if feature_scalar:
                values = values[:, 0]

            return self._single_trial(
                trial_positions[0],
                time_positions,
                values,
                valid,
                feature_positions,
                feature_scalar,
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

        return TsdTrials(
            values=values,
            time=self.t[time_positions],
            trials=IntervalSet(
                start=self.trials.start[trial_positions],
                end=self.trials.end[trial_positions],
            ),
            valid=valid,
            alignment_times=self.alignment_times[trial_positions],
            trial_ids=trial_ids,
            trial_metadata=self.trial_metadata.loc[trial_ids],
            feature_names=feature_names,
            feature_metadata=feature_metadata,
            bin_size=self.bin_size,
            data_kind=self.data_kind,
        )

    def _split_key(self, key) -> tuple[object, object, object]:
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
        trial: int,
        time_positions: np.ndarray,
        values: np.ndarray,
        valid: np.ndarray,
        feature_positions: np.ndarray | None,
        feature_scalar: bool,
    ) -> Tsd | TsdFrame:
        time = self.t[time_positions]
        time = time[valid]
        values = values[valid]

        support = IntervalSet(
            start=self.trials.start[trial] - self.alignment_times[trial],
            end=self.trials.end[trial] - self.alignment_times[trial],
        )

        if values.ndim == 1:
            return Tsd(t=time, d=values, time_support=support)

        feature_names = self.feature_names.take(feature_positions)
        feature_metadata = self.feature_metadata.loc[feature_names]

        return TsdFrame(
            t=time,
            d=values,
            time_support=support,
            columns=feature_names,
            metadata=feature_metadata,
        )

    def _select_trials(self, trial_ids) -> TsdTrials:
        trial_ids = pd.Index(trial_ids)
        positions = self.trial_ids.get_indexer(trial_ids)

        if np.any(positions < 0):
            missing = trial_ids[positions < 0].tolist()
            raise KeyError(f"Trials not found: {missing}")

        return self[positions]

    def _new(self, *, values: np.ndarray) -> TsdTrials:
        return TsdTrials(
            values=values,
            time=self.t,
            trials=self.trials,
            valid=self.valid,
            alignment_times=self.alignment_times,
            trial_ids=self.trial_ids,
            trial_metadata=self.trial_metadata,
            feature_names=self.feature_names,
            feature_metadata=self.feature_metadata,
            bin_size=self.bin_size,
            data_kind=self.data_kind,
        )

    def psth(
        self,
        group_by: str | None = None,
        *,
        rate: bool = True,
    ) -> Tsd | TsdFrame | dict[object, Tsd | TsdFrame]:
        """
        Average values across trials.

        Invalid padded samples are excluded independently at each time point.
        For binned count data, values are converted to rates by default.

        Parameters
        ----------
        group_by : str, optional
            Trial metadata column used to compute one average per group.
            If omitted, all trials are averaged together.
        rate : bool, optional
            If True and the data contain counts, divide the average counts by
            the bin size to return rates in Hz. Defaults to True. This parameter
            has no effect on continuous data.

        Returns
        -------
        Tsd or TsdFrame or dict
            A ``Tsd`` for two-dimensional input or a ``TsdFrame`` for
            three-dimensional input. When ``group_by`` is specified, returns a
            dictionary mapping metadata values to these objects.

        Raises
        ------
        KeyError
            If ``group_by`` is not a trial metadata column.

        Examples
        --------
        Average all trials:

        >>> average = dense.psth()
        >>> isinstance(average, nap.TsdFrame)
        True

        Average trials by condition:

        >>> dense.trial_metadata["outcome"] = ["hit", "miss"]
        >>> averages = dense.psth(group_by="outcome")
        >>> set(averages)
        {'hit', 'miss'}

        Keep binned spike counts rather than converting to Hz:

        >>> average_counts = dense.psth(rate=False)
        """

        if group_by is None:
            return self._psth(np.arange(len(self)))

        if group_by not in self.trial_metadata.columns:
            raise KeyError(f"Unknown trial metadata column: {group_by!r}")

        groups = self.trial_metadata.groupby(group_by, sort=False).groups
        return {
            name: self._psth(
                self.trial_ids.get_indexer(pd.Index(trial_ids)),
                rate=rate,
            )
            for name, trial_ids in groups.items()
        }

    def _psth(
        self,
        trials: np.ndarray,
        *,
        rate: bool = True,
    ) -> Tsd | TsdFrame:
        values = self.values[trials]
        valid = self.valid[trials]

        expanded_valid = valid
        if values.ndim == 3:
            expanded_valid = valid[:, :, None]

        total = np.where(expanded_valid, values, 0).sum(axis=0)
        denominator = expanded_valid.sum(axis=0)

        mean = np.divide(
            total,
            denominator,
            out=np.full(total.shape, np.nan, dtype=float),
            where=denominator != 0,
        )

        if self.data_kind == "count" and rate:
            mean /= self.bin_size

        valid_time = valid.any(axis=0)
        time = self.t[valid_time]
        mean = mean[valid_time]

        if len(time):
            half_bin = self.bin_size / 2 if self.bin_size is not None else 0.0
            support = IntervalSet(
                start=time[0] - half_bin,
                end=time[-1] + half_bin,
            )
        else:
            support = IntervalSet([], [])

        if mean.ndim == 1:
            return Tsd(t=time, d=mean, time_support=support)

        return TsdFrame(
            t=time,
            d=mean,
            time_support=support,
            columns=self.feature_names,
            metadata=self.feature_metadata,
        )
