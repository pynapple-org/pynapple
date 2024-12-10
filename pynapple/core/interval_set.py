"""        
The class `IntervalSet` deals with non-overlaping epochs. `IntervalSet` objects can interact with each other or with the time series objects.
"""

import importlib
import warnings
from numbers import Number

import numpy as np
import pandas as pd
from numpy.lib.mixins import NDArrayOperatorsMixin
from tabulate import tabulate

from ._jitted_functions import (
    _jitfix_iset,
    jitdiff,
    jitin_interval,
    jitintersect,
    jitunion,
)
from .config import nap_config
from .metadata_class import _MetadataMixin, add_meta_docstring
from .time_index import TsIndex
from .utils import (
    _convert_iter_to_str,
    _get_terminal_size,
    _IntervalSetSliceHelper,
    check_filename,
    convert_to_numpy_array,
    is_array_like,
)

all_warnings = np.array(
    [
        "Some starts and ends are equal. Removing 1 microsecond!",
        "Some ends precede the relative start. Dropping them!",
        "Some starts precede the previous end. Joining them!",
        "Some epochs have no duration",
    ]
)


class IntervalSet(NDArrayOperatorsMixin, _MetadataMixin):
    """
    A class representing a (irregular) set of time intervals in elapsed time, with relative operations

    The `IntervalSet` object behaves like a numpy ndarray with the limitation that the object is not mutable.

    If start and end are not aligned, meaning that:

    1. len(start) != len(end)
    2. end[i] > start[i]
    3. start[i+1] < end[i]
    4. start and end are not sorted,

    IntervalSet will try to "fix" the data by eliminating some of the start and end data points.

    Parameters
    ----------
    start : numpy.ndarray or number or pandas.DataFrame or pandas.Series or iterable of (start, end) pairs
        Beginning of intervals.
        Alternatively, the `end` argument can be left out and `start` can be one of the following:

        - IntervalSet
        - pandas.DataFrame with columns ["start", "end"]
        - iterable of (start, end) pairs
        - a single (start, end) pair

    end : numpy.ndarray or number or pandas.Series, optional
        Ends of intervals.
    time_units : str, optional
        Time unit of the intervals ('us', 'ms', 's' [default])
    metadata: pandas.DataFrame or dict, optional
        Metadata associated with each interval. Metadata names are pulled from DataFrame columns or dictionary keys.
        The length of the metadata should match the length of the intervals.

    Raises
    ------
    RuntimeError
        If `start` and `end` arguments are of unknown type.

    Examples
    --------
    Initialize an IntervalSet with a list of start and end times:

    >>> import pynapple as nap
    >>> import numpy as np
    >>> start = [0, 10, 20]
    >>> end = [5, 12, 33]
    >>> ep = nap.IntervalSet(start=start, end=end)
    >>> ep
      index    start    end
          0        0      5
          1       10     12
          2       20     33
    shape: (3, 2), time unit: sec.

    Initialize an IntervalSet with an array of start and end pairs:

    >>> times = np.array([[0, 5], [10, 12], [20, 33]])
    >>> ep = nap.IntervalSet(times)
    >>> ep
      index    start    end
          0        0      5
          1       10     12
          2       20     33
    shape: (3, 2), time unit: sec.

    Initialize an IntervalSet with metadata:

    >>> start = [0, 10, 20]
    >>> end = [5, 12, 33]
    >>> metadata = {"label": ["a", "b", "c"]}
    >>> ep = nap.IntervalSet(start=start, end=end, metadata=metadata)
      index    start    end      label
          0        0      5  |   a
          1       10     12  |   b
          2       20     33  |   c
    shape: (3, 2), time unit: sec.

    Initialize an IntervalSet with a pandas DataFrame:

    >>> import pandas as pd
    >>> df = pd.DataFrame(data={"start": [0, 10, 20], "end": [5, 12, 33], "label": ["a", "b", "c"]})
    >>> ep = nap.IntervalSet(df)
    >>> ep
      index    start    end      label
          0        0      5  |   a
          1       10     12  |   b
          2       20     33  |   c
    shape: (3, 2), time unit: sec.

    Apply numpy functions to an IntervalSet:

    >>> ep = nap.IntervalSet(start=[0, 10], end=[5,20])
    >>> ep
      index    start    end
          0        0      5
          1       10     20
    shape: (2, 2), time unit: sec.

    >>> np.diff(ep, 1)
    UserWarning: Converting IntervalSet to numpy.array
    array([[ 5.],
            [10.]])

    Slicing an IntervalSet:

    >>> ep[:,0]
    array([ 0., 10.])

    >>> ep[0]
    start    end
    0        0      5
    shape: (1, 2)

    Modifying the `IntervalSet` will raise an error:

    >>> ep[0,0] = 1
    RuntimeError: IntervalSet is immutable. Starts and ends have been already sorted.
    """

    start: np.ndarray
    """The start times of each interval"""

    end: np.ndarray
    """The end times of each interval"""

    values: np.ndarray
    """Array of start and end times"""

    index: np.ndarray
    """Index of each interval, automatically set from 0 to n_intervals"""

    columns: np.ndarray
    """Column names of the IntervalSet, which are always ["start", "end"]"""

    nap_class: str
    """The pynapple class name"""

    def __init__(
        self,
        start,
        end=None,
        time_units="s",
        metadata=None,
    ):
        # set directly in __dict__ to avoid infinite recursion in __setattr__
        self.__dict__["_initialized"] = False
        if isinstance(start, IntervalSet):
            end = start.end.astype(np.float64)
            start = start.start.astype(np.float64)

        elif isinstance(start, pd.DataFrame):
            assert (
                "start" in start.columns and "end" in start.columns
            ), """
                DataFrame must contain columns name "start" and "end" for start and end times.                   
                """
            metadata = start.drop(columns=["start", "end"])
            end = start["end"].values.astype(np.float64)
            start = start["start"].values.astype(np.float64)

        else:
            if end is None:
                # Require iterable of (start, end) tuples
                try:
                    start_end_array = np.array(list(start)).reshape(-1, 2)
                    start, end = zip(*start_end_array)
                except (TypeError, ValueError):
                    raise ValueError(
                        "Unable to Interpret the input. Please provide a list of start-end pairs."
                    )

            args = {"start": start, "end": end}

            for arg, data in args.items():
                if isinstance(data, Number):
                    args[arg] = np.array([data])
                elif isinstance(data, (list, tuple)):
                    args[arg] = np.ravel(np.array(data))
                elif isinstance(data, pd.Series):
                    args[arg] = data.values
                elif isinstance(data, np.ndarray):
                    args[arg] = np.ravel(data)
                elif is_array_like(data):
                    args[arg] = convert_to_numpy_array(data, arg)
                else:
                    raise RuntimeError(
                        "Unknown format for {}. Accepted formats are numpy.ndarray, list, tuple or any array-like objects.".format(
                            arg
                        )
                    )

            start = args["start"]
            end = args["end"]

            assert len(start) == len(end), "Starts end ends are not of the same length"

        start = TsIndex.format_timestamps(start, time_units)
        end = TsIndex.format_timestamps(end, time_units)

        drop_meta = False
        if not (np.diff(start) > 0).all():
            if metadata is not None:
                msg1 = "Cannot add metadata to unsorted start times. "
                msg2 = " and dropping metadata"
            else:
                msg1 = ""
                msg2 = ""
            warnings.warn(
                "start is not sorted. " + msg1 + "Sorting it" + msg2 + ".", stacklevel=2
            )
            start = np.sort(start)
            drop_meta = True

        if not (np.diff(end) > 0).all():
            if metadata is not None:
                msg1 = "Cannot add metadata to unsorted end times. "
                msg2 = " and dropping metadata"
            else:
                msg1 = ""
                msg2 = ""
            warnings.warn(
                "end is not sorted. " + msg1 + "Sorting it" + msg2 + ".", stacklevel=2
            )
            end = np.sort(end)
            drop_meta = True

        data, to_warn = _jitfix_iset(start, end)

        if np.any(to_warn):
            msg = "\n".join(all_warnings[to_warn])
            warnings.warn(msg, stacklevel=2)
            if np.any(to_warn[1:]) and (metadata is not None):
                drop_meta = True
                warnings.warn("epochs have changed, dropping metadata.", stacklevel=2)

        self.values = data
        self.index = np.arange(data.shape[0], dtype="int")
        self.columns = np.array(["start", "end"])
        self.nap_class = self.__class__.__name__
        # initialize metadata to get all attributes before setting metadata
        _MetadataMixin.__init__(self)
        self._class_attributes = self.__dir__()  # get list of all attributes
        self._class_attributes.append("_class_attributes")  # add this property
        self._initialized = True
        if drop_meta is False:
            self.set_info(metadata)

    def __repr__(self):
        # Start by determining how many columns and rows.
        # This can be unique for each object
        cols, rows = _get_terminal_size()
        # max_cols = np.maximum(cols // 12, 5)
        max_rows = np.maximum(rows - 10, 2)
        # By default, the first three columns should always show.

        # Adding an extra column between actual values and metadata
        try:
            metadata = self._metadata
            col_names = metadata.columns
        except Exception:
            # Necessary for backward compatibility when saving IntervalSet as pickle
            metadata = pd.DataFrame(index=self.index)
            col_names = []

        headers = ["index", "start", "end"]
        if len(col_names):
            headers += [""] + [c for c in col_names]
        bottom = f"shape: {self.shape}, time unit: sec."

        # We rarely want to print everything as it can be very big.
        if len(self) > max_rows:
            n_rows = max_rows // 2
            if len(col_names):
                separator = np.array([["|"] * n_rows]).T
            else:
                separator = np.empty((n_rows, 0))
            data = np.vstack(
                (
                    np.hstack(
                        (
                            self.index[0:n_rows, None],
                            self.values[0:n_rows],
                            separator,
                            _convert_iter_to_str(metadata.values[0:n_rows]),
                        ),
                        dtype=object,
                    ),
                    np.array([["..." for _ in range(len(headers))]], dtype=object),
                    np.hstack(
                        (
                            self.index[-n_rows:, None],
                            self.values[0:n_rows],
                            separator,
                            _convert_iter_to_str(metadata.values[-n_rows:]),
                        ),
                        dtype=object,
                    ),
                )
            )
        else:
            if len(col_names):
                separator = np.array([["|"] * len(self)]).T
            else:
                separator = np.empty((len(self), 0))
            data = np.hstack(
                (
                    self.index[:, None],
                    self.values,
                    separator,
                    _convert_iter_to_str(metadata.values),
                ),
                dtype=object,
            )

        return tabulate(data, headers=headers, tablefmt="plain") + "\n" + bottom

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.values)

    def __setattr__(self, name, value):
        # necessary setter to allow metadata to be set as an attribute
        if self._initialized:
            if name in self._class_attributes:
                raise AttributeError(
                    f"Cannot set attribute '{name}'; IntervalSet is immutable. Use 'set_info()' to set '{name}' as metadata."
                )
            else:
                _MetadataMixin.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        # Necessary for backward compatibility with pickle

        # avoid infinite recursion when pickling due to
        # self._metadata.column having attributes '__reduce__', '__reduce_ex__'
        if name in ("__getstate__", "__setstate__", "__reduce__", "__reduce_ex__"):
            raise AttributeError(name)

        try:
            metadata = self._metadata
        except Exception:
            metadata = pd.DataFrame(index=self.index)

        if name == "_metadata":
            return metadata
        elif name in metadata.columns:
            return _MetadataMixin.__getattr__(self, name)
        else:
            return super().__getattr__(name)

    def __setitem__(self, key, value):
        if key in self.columns:
            raise RuntimeError(
                "IntervalSet is immutable. Starts and ends have been already sorted."
            )
        elif isinstance(key, str):
            _MetadataMixin.__setitem__(self, key, value)
        else:
            raise RuntimeError(
                "IntervalSet is immutable. Starts and ends have been already sorted."
            )

    def __getitem__(self, key):
        try:
            metadata = _MetadataMixin.__getitem__(self, key)
        except Exception:
            metadata = pd.DataFrame(index=self.index)

        if isinstance(key, str):
            # self[str]
            if key == "start":
                return self.values[:, 0]
            elif key == "end":
                return self.values[:, 1]
            elif key in self._metadata.columns:
                return _MetadataMixin.__getitem__(self, key)
            else:
                raise IndexError(
                    f"Unknown string argument. Should be in {['start', 'end'] + list(self._metadata.keys())}"
                )
        elif isinstance(key, list) and all(isinstance(x, str) for x in key):
            # self[[*str]]
            # easiest to convert to dataframe and then slice
            # in case of mixing ["start", "end"] with metadata columns
            df = self.as_dataframe()
            if all(x in key for x in ["start", "end"]):
                return IntervalSet(df[key])
            else:
                return df[key]
        elif isinstance(key, Number):
            # self[Number]
            output = self.values.__getitem__(key)
            return IntervalSet(start=output[0], end=output[1], metadata=metadata)
        elif isinstance(key, (slice, list, np.ndarray, pd.Series)):
            # self[array_like]
            output = self.values.__getitem__(key)
            metadata = _MetadataMixin.__getitem__(self, key).reset_index(drop=True)
            return IntervalSet(start=output[:, 0], end=output[:, 1], metadata=metadata)
        elif isinstance(key, tuple):
            if len(key) == 2:
                if isinstance(key[1], Number):
                    # self[Any, Number]
                    # allow number indexing for start and end times for backward compatibility
                    return self.values.__getitem__(key)

                elif isinstance(key[1], str):
                    # self[Any, str]
                    if key[1] == "start":
                        return self.values[key[0], 0]
                    elif key[1] == "end":
                        return self.values[key[0], 1]
                    elif key[1] in self._metadata.columns:
                        return _MetadataMixin.__getitem__(self, key)

                elif isinstance(key[1], (list, np.ndarray)):
                    if all(isinstance(x, str) for x in key[1]):
                        # self[Any, [*str]]
                        # easiest to convert to dataframe and then slice
                        # in case of mixing ["start", "end"] with metadata columns
                        df = self.as_dataframe()
                        if all(x in key[1] for x in ["start", "end"]):
                            return IntervalSet(df.loc[key])
                        else:
                            return df.loc[key]
                    elif all(isinstance(x, Number) for x in key[1]):
                        if all(x in [0, 1] for x in key[1]):
                            # self[Any, [0,1]]
                            # allow number indexing for start and end times for backward compatibility
                            output = self.values.__getitem__(key[0])
                            if isinstance(key[0], Number):
                                return IntervalSet(start=output[0], end=output[1])
                            else:
                                return IntervalSet(start=output[:, 0], end=output[:, 1])
                        else:
                            raise IndexError(
                                f"index {key[1]} out of bounds for IntervalSet axis 1 with size 2"
                            )
                    else:
                        raise IndexError(f"unknown index {key[1]} for index 2")

                elif isinstance(key[1], slice):
                    if key[1] == slice(None, None, None):
                        # self[Any, :]
                        output = self.values.__getitem__(key[0])
                        metadata = _MetadataMixin.__getitem__(self, key[0])

                        if isinstance(key[0], Number):
                            return IntervalSet(
                                start=output[0], end=output[1], metadata=metadata
                            )
                        else:
                            return IntervalSet(
                                start=output[:, 0],
                                end=output[:, 1],
                                metadata=metadata.reset_index(drop=True),
                            )

                    elif key[1] == slice(0, 2, None):
                        # self[Any, :2]
                        # allow number indexing for start and end times for backward compatibility
                        output = self.values.__getitem__(key[0])

                        if isinstance(key[0], Number):
                            return IntervalSet(start=output[0], end=output[1])
                        else:
                            return IntervalSet(start=output[:, 0], end=output[:, 1])

                    else:
                        raise IndexError(
                            f"index {key[1]} out of bounds for IntervalSet axis 1 with size 2"
                        )

                else:
                    raise IndexError(f"unknown type {type(key[1])} for index 2")

            else:
                raise IndexError(
                    "too many indices for IntervalSet: IntervalSet is 2-dimensional"
                )
        else:
            raise IndexError(f"unknown type {type(key)} for index")

    def __array__(self, dtype=None):
        return self.values.astype(dtype)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        new_args = []
        for a in args:
            if isinstance(a, self.__class__):
                new_args.append(a.values)
            else:
                new_args.append(a)

        out = ufunc(*new_args, **kwargs)

        if not nap_config.suppress_conversion_warnings:
            warnings.warn(
                "Converting IntervalSet to numpy.array",
                UserWarning,
            )
        return out

    def __array_function__(self, func, types, args, kwargs):
        new_args = []
        for a in args:
            if isinstance(a, self.__class__):
                new_args.append(a.values)
            else:
                new_args.append(a)

        out = func._implementation(*new_args, **kwargs)

        if not nap_config.suppress_conversion_warnings:
            warnings.warn(
                "Converting IntervalSet to numpy.array",
                UserWarning,
            )
        return out

    @property
    def start(self):
        return self.values[:, 0]

    @property
    def end(self):
        return self.values[:, 1]

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def size(self):
        return self.values.size

    @property
    def starts(self):
        """Return the starts of the IntervalSet as a Ts object

        Returns
        -------
        Ts
            The starts of the IntervalSet
        """
        warnings.warn(
            "starts is a deprecated function. It will be removed in future versions",
            category=DeprecationWarning,
            stacklevel=2,
        )
        time_series = importlib.import_module(".time_series", "pynapple.core")
        return time_series.Ts(t=self.values[:, 0])

    @property
    def ends(self):
        """Return the ends of the IntervalSet as a Ts object

        Returns
        -------
        Ts
            The ends of the IntervalSet
        """
        warnings.warn(
            "ends is a deprecated function. It will be removed in future versions",
            category=DeprecationWarning,
            stacklevel=2,
        )
        time_series = importlib.import_module(".time_series", "pynapple.core")
        return time_series.Ts(t=self.values[:, 1])

    @property
    def loc(self):
        """
        Slicing function to add compatibility with pandas DataFrame after removing it as a super class of IntervalSet
        """
        return _IntervalSetSliceHelper(self)

    @classmethod
    def _from_npz_reader(cls, file):
        """Load an IntervalSet object from a npz file.

        The file should contain the keys 'start', 'end' and 'type'.
        The 'type' key should be 'IntervalSet'.

        Parameters
        ----------
        file : NPZFile object
            opened npz file interface.

        Returns
        -------
        IntervalSet
            The IntervalSet object
        """
        ep = cls(start=file["start"], end=file["end"])
        if "_metadata" in file:  # load metadata if it exists
            if file["_metadata"]:  # check that metadata is not empty
                metadata = pd.DataFrame.from_dict(file["_metadata"].item())
                ep.set_info(metadata)
        return ep

    def time_span(self):
        """
        Time span of the interval set.

        Returns
        -------
        out: IntervalSet
            an IntervalSet with a single interval encompassing the whole IntervalSet
        """
        if len(self.metadata_columns):
            warnings.warn(
                "metadata incompatible with time_span method. dropping metadata from result",
                UserWarning,
            )
        s = self.values[0, 0]
        e = self.values[-1, 1]
        return IntervalSet(s, e)

    def tot_length(self, time_units="s"):
        """
        Total elapsed time in the set.

        Parameters
        ----------
        time_units : None, optional
            The time units to return the result in ('us', 'ms', 's' [default])

        Returns
        -------
        out: float
            _
        """
        tot_l = np.sum(self.values[:, 1] - self.values[:, 0])
        return TsIndex.return_timestamps(np.array([tot_l]), time_units)[0]

    def intersect(self, a):
        """
        Set intersection of IntervalSet

        Parameters
        ----------
        a : IntervalSet
            the IntervalSet to intersect self with

        Returns
        -------
        out: IntervalSet
            _
        """
        start1 = self.values[:, 0]
        end1 = self.values[:, 1]
        start2 = a.values[:, 0]
        end2 = a.values[:, 1]
        s, e, m = jitintersect(start1, end1, start2, end2)
        m1 = self._metadata.loc[m[:, 0]].reset_index(drop=True)
        m2 = a._metadata.loc[m[:, 1]].reset_index(drop=True)
        return IntervalSet(s, e, metadata=m1.join(m2))

    def union(self, a):
        """
        set union of IntervalSet

        Parameters
        ----------
        a : IntervalSet
            the IntervalSet to union self with

        Returns
        -------
        out: IntervalSet
            _
        """
        if len(self.metadata_columns):
            warnings.warn(
                "metadata incompatible with union method. dropping metadata from result",
                UserWarning,
            )
        start1 = self.values[:, 0]
        end1 = self.values[:, 1]
        start2 = a.values[:, 0]
        end2 = a.values[:, 1]
        s, e = jitunion(start1, end1, start2, end2)
        return IntervalSet(s, e)

    def set_diff(self, a):
        """
        set difference of IntervalSet

        Parameters
        ----------
        a : IntervalSet
            the IntervalSet to set-substract from self

        Returns
        -------
        out: IntervalSet
            _
        """
        start1 = self.values[:, 0]
        end1 = self.values[:, 1]
        start2 = a.values[:, 0]
        end2 = a.values[:, 1]
        s, e, m = jitdiff(start1, end1, start2, end2)
        m1 = self._metadata.loc[m].reset_index(drop=True)
        return IntervalSet(s, e, metadata=m1)

    def in_interval(self, tsd):
        """
        finds out in which element of the interval set each point in a time series fits.

        NaNs for those that don't fit an interval

        Parameters
        ----------
        tsd : Tsd
            The tsd to be binned

        Returns
        -------
        out: numpy.ndarray
            an array with the interval index labels for each time stamp (NaN) for timestamps not in IntervalSet
        """
        times = tsd.index.values
        starts = self.values[:, 0]
        ends = self.values[:, 1]

        return jitin_interval(times, starts, ends)

    def drop_short_intervals(self, threshold, time_units="s"):
        """
        Drops the short intervals in the interval set with duration shorter than `threshold`.

        Parameters
        ----------
        threshold : numeric
            Time threshold for "short" intervals
        time_units : None, optional
            The time units for the treshold ('us', 'ms', 's' [default])

        Returns
        -------
        out: IntervalSet
            A copied IntervalSet with the dropped intervals
        """
        threshold = TsIndex.format_timestamps(
            np.array([threshold], dtype=np.float64), time_units
        )[0]
        return self[(self.values[:, 1] - self.values[:, 0]) > threshold]

    def drop_long_intervals(self, threshold, time_units="s"):
        """
        Drops the long intervals in the interval set with duration longer than `threshold`.

        Parameters
        ----------
        threshold : numeric
            Time threshold for "long" intervals
        time_units : None, optional
            The time units for the treshold ('us', 'ms', 's' [default])

        Returns
        -------
        out: IntervalSet
            A copied IntervalSet with the dropped intervals
        """
        threshold = TsIndex.format_timestamps(
            np.array([threshold], dtype=np.float64), time_units
        )[0]
        return self[(self.values[:, 1] - self.values[:, 0]) < threshold]

    def as_units(self, units="s"):
        """
        returns a pandas DataFrame with time expressed in the desired unit

        Parameters
        ----------
        units : None, optional
            'us', 'ms', or 's' [default]

        Returns
        -------
        out: pandas.DataFrame
            DataFrame with adjusted times
        """

        data = self.values.copy()
        data = TsIndex.return_timestamps(data, units)
        if units == "us":
            data = data.astype(np.int64)

        df = pd.DataFrame(index=self.index, data=data, columns=self.columns)

        return df

    def merge_close_intervals(self, threshold, time_units="s"):
        """
        Merges intervals that are very close.

        Parameters
        ----------
        threshold : numeric
            time threshold for the closeness of the intervals
        time_units : None, optional
            time units for the threshold ('us', 'ms', 's' [default])

        Returns
        -------
        out: IntervalSet
            a copied IntervalSet with merged intervals

        """
        if len(self.metadata_columns):
            warnings.warn(
                "metadata incompatible with merge_close_intervals method. dropping metadata from result",
                UserWarning,
            )

        if len(self) == 0:
            return IntervalSet(start=[], end=[])

        threshold = TsIndex.format_timestamps(
            np.array((threshold,), dtype=np.float64).ravel(), time_units
        )[0]
        start = self.values[:, 0]
        end = self.values[:, 1]
        tojoin = (start[1:] - end[0:-1]) > threshold
        start = np.hstack((start[0], start[1:][tojoin]))
        end = np.hstack((end[0:-1][tojoin], end[-1]))

        return IntervalSet(start=start, end=end)

    def get_intervals_center(self, alpha=0.5):
        """
        Returns by default the centers of each intervals.

        It is possible to bias the midpoint by changing the alpha parameter between [0, 1]
        For each epoch:
        t = start + (end-start)*alpha

        Parameters
        ----------
        alpha : float, optional
            The midpoint within each interval.

        Returns
        -------
        Ts
            Timestamps object
        """
        time_series = importlib.import_module(".time_series", "pynapple.core")
        starts = self.values[:, 0]
        ends = self.values[:, 1]

        if not isinstance(alpha, float):
            raise RuntimeError("Parameter alpha should be float type")

        alpha = np.clip(alpha, 0, 1)
        t = starts + (ends - starts) * alpha
        return time_series.Ts(t=t, time_support=self)

    def as_dataframe(self):
        """
        Convert the `IntervalSet` object to a pandas.DataFrame object.

        Returns
        -------
        out: pandas.DataFrame
            _
        """
        df = pd.DataFrame(data=self.values, columns=["start", "end"])
        return pd.concat([df, self._metadata], axis=1)

    def save(self, filename):
        """
        Save IntervalSet object in npz format. The file will contain the starts and ends.

        The main purpose of this function is to save small/medium sized IntervalSet
        objects. For example, you determined some epochs for one session that you want to save
        to avoid recomputing them.

        You can load the object with `nap.load_file`. Keys are 'start', 'end' and 'type'.
        See the example below.

        Parameters
        ----------
        filename : str
            The filename

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> ep = nap.IntervalSet(start=[0, 10, 20], end=[5, 12, 33])
        >>> ep.save("my_ep.npz")

        To load you file, you can use the `nap.load_file` function :

        >>> ep = nap.load_file("my_path/my_ep.npz")
        >>> ep
           start   end
        0    0.0   5.0
        1   10.0  12.0
        2   20.0  33.0

        Raises
        ------
        RuntimeError
            If filename is not str, path does not exist or filename is a directory.
        """
        np.savez(
            check_filename(filename),
            start=self.values[:, 0],
            end=self.values[:, 1],
            type=np.array(["IntervalSet"], dtype=np.str_),
            _metadata=self._metadata.to_dict(),  # save metadata as dictionary
        )

        return

    def split(self, interval_size, time_units="s"):
        """Split `IntervalSet` to a new `IntervalSet` with each interval being of size `interval_size`.

        Used mostly for chunking very large dataset or looping throught multiple epoch of same duration.

        This function skips the epochs that are shorter than `interval_size`.

        Note that intervals are strictly non-overlapping in pynapple. One microsecond is removed from contiguous intervals.

        Parameters
        ----------
        interval_size : Number
            Description
        time_units : str, optional
            time units for the `interval_size` ('us', 'ms', 's' [default])

        Returns
        -------
        IntervalSet
            New `IntervalSet` with equal sized intervals

        Raises
        ------
        IOError
            If `interval_size` is not a Number or is below 0
            If `time_units` is not a string
        """
        if not isinstance(interval_size, Number):
            raise IOError("Argument interval_size should of type float or int")

        if not interval_size > 0:
            raise IOError("Argument interval_size should be strictly larger than 0")

        if not isinstance(time_units, str):
            raise IOError("Argument time_units should be of type str")

        if len(self) == 0:
            return IntervalSet(start=[], end=[])

        interval_size = TsIndex.format_timestamps(
            np.array((interval_size,), dtype=np.float64).ravel(), time_units
        )[0]

        interval_size = np.round(interval_size, nap_config.time_index_precision)

        durations = np.round(self.end - self.start, nap_config.time_index_precision)

        idxs = np.where(durations > interval_size)[0]
        size_tmp = (
            np.ceil((self.end[idxs] - self.start[idxs]) / interval_size)
        ).astype(int) + 1
        new_starts = np.full(size_tmp.sum() - size_tmp.shape[0], np.nan)
        new_ends = np.full(size_tmp.sum() - size_tmp.shape[0], np.nan)
        new_meta = np.full(size_tmp.sum() - size_tmp.shape[0], np.nan)
        i0 = 0
        for cnt, idx in enumerate(idxs):
            # repeat metainfo for each new interval
            new_meta[i0 : i0 + size_tmp[cnt] - 1] = idx
            new_starts[i0 : i0 + size_tmp[cnt] - 1] = np.arange(
                self.start[idx], self.end[idx], interval_size
            )
            new_ends[i0 : i0 + size_tmp[cnt] - 2] = new_starts[
                i0 + 1 : i0 + size_tmp[cnt] - 1
            ]
            new_ends[i0 + size_tmp[cnt] - 2] = self.end[idx]
            i0 += size_tmp[cnt] - 1
        new_starts = np.round(new_starts, nap_config.time_index_precision)
        new_ends = np.round(new_ends, nap_config.time_index_precision)

        durations = np.round(new_ends - new_starts, nap_config.time_index_precision)
        tokeep = durations >= interval_size
        new_starts = new_starts[tokeep]
        new_ends = new_ends[tokeep]
        new_meta = new_meta[tokeep]
        metadata = self._metadata.loc[new_meta].reset_index(drop=True)

        # Removing 1 microsecond to have strictly non-overlapping intervals for intervals coming from the same epoch
        new_ends -= 1e-6

        return IntervalSet(new_starts, new_ends, metadata=metadata)

    @add_meta_docstring("set_info")
    def set_info(self, metadata=None, **kwargs):
        """
        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> times = np.array([[0, 5], [10, 12], [20, 33]])
        >>> ep = nap.IntervalSet(times)

        To add metadata with a pandas.DataFrame:

        >>> import pandas as pd
        >>> metadata = pd.DataFrame(data=['left','right','left'], columns=['choice'])
        >>> ep.set_info(metadata)
        >>> ep
          index    start    end      choice
              0        0      5  |   left
              1       10     12  |   right
              2       20     33  |   left
        shape: (3, 2), time unit: sec.

        To add metadata with a dictionary:

        >>> metadata = {"reward": [1, 0, 1]}
        >>> ep.set_info(metadata)
        >>> ep
          index    start    end      choice      reward
              0        0      5  |   left             1
              1       10     12  |   right            0
              2       20     33  |   left             1
        shape: (3, 2), time unit: sec.

        To add metadata with a keyword argument (pd.Series, numpy.ndarray, list or tuple):

        >>> stim = pd.Series(data = [10, -23, 12])
        >>> ep.set_info(stim=stim)
        >>> ep
          index    start    end      choice      reward    stim
              0        0      5  |   left             1      10
              1       10     12  |   right            0     -23
              2       20     33  |   left             1      12
        shape: (3, 2), time unit: sec.

        To add metadata as an attribute:

        >>> ep.label = ["a", "b", "c"]
        >>> ep
          index    start    end      choice      reward  label
              0        0      5  |   left             1  a
              1       10     12  |   right            0  b
              2       20     33  |   left             1  c
        shape: (3, 2), time unit: sec.

        To add metadata as a key:

        >>> ep["error"] = [0, 0, 0]
        >>> ep
          index    start    end      choice      reward  label      error
              0        0      5  |   left             1  a             0
              1       10     12  |   right            0  b             0
              2       20     33  |   left             1  c             0
        shape: (3, 2), time unit: sec.

        Metadata can be overwritten:

        >>> ep.set_info(label=["x", "y", "z"])
        >>> ep
          index    start    end      choice      reward  label      error
              0        0      5  |   left             1  x             0
              1       10     12  |   right            0  y             0
              2       20     33  |   left             1  z             0
        shape: (3, 2), time unit: sec.
        """
        _MetadataMixin.set_info(self, metadata, **kwargs)

    @add_meta_docstring("get_info")
    def get_info(self, key):
        """
        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> times = np.array([[0, 5], [10, 12], [20, 33]])
        >>> metadata = {"l1": [1, 2, 3], "l2": ["x", "x", "y"]}
        >>> ep = nap.IntervalSet(tmp,metadata=metadata)

        To access a single metadata column:

        >>> ep.get_info("l1")
        0    1
        1    2
        2    3
        Name: l1, dtype: int64

        To access multiple metadata columns:

        >>> ep.get_info(["l1", "l2"])
           l1 l2
        0   1  x
        1   2  x
        2   3  y

        To access metadata of a single index:

        >>> ep.get_info(0)
        rate    0.667223
        l1             1
        l2             x
        Name: 0, dtype: object

        To access metadata of multiple indices:

        >>> ep.get_info([0, 1])
               rate  l1 l2
        0  0.667223   1  x
        1  1.334445   2  x

        To access metadata of a single index and column:

        >>> ep.get_info((0, "l1"))
        np.int64(1)

        To access metadata as an attribute:

        >>> ep.l1
        0    1
        1    2
        2    3
        Name: l1, dtype: int64

        To access metadata as a key:

        >>> ep["l1"]
        0    1
        1    2
        2    3
        Name: l1, dtype: int64

        Multiple metadata columns can be accessed as keys:

        >>> ep[["l1", "l2"]]
           l1 l2
        0   1  x
        1   2  x
        2   3  y
        """
        return _MetadataMixin.get_info(self, key)
