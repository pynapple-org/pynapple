"""
Abstract class for `core` time series.

"""

import abc
from numbers import Number

import numpy as np

from ._core_functions import _count, _restrict, _value_from
from .interval_set import IntervalSet
from .time_index import TsIndex
from .utils import check_filename, convert_to_numpy_array


def add_base_docstring(base_func, sep="\n"):
    base_doc = getattr(_Base, base_func).__doc__

    def _decorator(func):
        func.__doc__ = sep.join([base_doc, func.__doc__])
        return func

    return _decorator


class _Base(abc.ABC):
    """
    Abstract base class for time series and timestamps objects.
    Implement most of the shared functions across concrete classes `Ts`, `Tsd`, `TsdFrame`, `TsdTensor`
    """

    _initialized = False

    index: TsIndex
    """The time index of the time series"""

    rate: float
    """Frequency of the time series (Hz) computed over the time support"""

    time_support: IntervalSet
    """The time support of the time series"""

    def __init__(self, t, time_units="s", time_support=None):
        if isinstance(t, TsIndex):
            self.index = t
        else:
            self.index = TsIndex(convert_to_numpy_array(t, "t"), time_units)

        if time_support is not None:
            assert isinstance(
                time_support, IntervalSet
            ), "time_support should be an IntervalSet"

        # Restrict should occur in the inherited class
        if len(self.index):
            if isinstance(time_support, IntervalSet):
                self.time_support = time_support
            else:
                self.time_support = IntervalSet(start=self.index[0], end=self.index[-1])

            self.rate = self.index.shape[0] / np.sum(
                self.time_support.values[:, 1] - self.time_support.values[:, 0]
            )
        else:
            self.rate = np.nan
            self.time_support = IntervalSet(start=[], end=[])

    @abc.abstractmethod
    def _define_instance(self, time_index, time_support, values=None, **kwargs):
        """Return a new class instance.

        Pass "columns", "metadata" and other attributes of self
        to the new instance unless specified in kwargs.
        """
        pass

    @property
    def t(self):
        """The time index of the time series"""
        return self.index.values

    @property
    def start(self):
        """The first time index in the time series"""
        return self.start_time()

    @property
    def end(self):
        """The last time index in the time series"""
        return self.end_time()

    @property
    def shape(self):
        """The shape of the time series"""
        return self.index.shape

    def __repr__(self):
        return str(self.__class__)

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.index)

    def __setattr__(self, name, value):
        """Object is immutable"""
        if self._initialized:
            raise RuntimeError(
                "Changing directly attributes is not permitted for {}.".format(
                    self.nap_class
                )
            )
        else:
            object.__setattr__(self, name, value)

    @abc.abstractmethod
    def __getitem__(self, key, *args, **kwargs):
        """getter for time series"""
        pass

    def __setitem__(self, key, value):
        pass

    def times(self, units="s"):
        """
        The time index of the object, returned as np.double in the desired time units.

        Parameters
        ----------
        units : str, optional
            ('us', 'ms', 's' [default])

        Returns
        -------
        out: numpy.ndarray
            the time indexes
        """
        return self.index.in_units(units)

    def start_time(self, units="s"):
        """
        The first time index in the time series object

        Parameters
        ----------
        units : str, optional
            ('us', 'ms', 's' [default])

        Returns
        -------
        out: numpy.float64
            _
        """
        if len(self.index):
            return self.times(units=units)[0]
        else:
            return None

    def end_time(self, units="s"):
        """
        The last time index in the time series object

        Parameters
        ----------
        units : str, optional
            ('us', 'ms', 's' [default])

        Returns
        -------
        out: numpy.float64
            _
        """
        if len(self.index):
            return self.times(units=units)[-1]
        else:
            return None

    def value_from(self, data, ep=None, mode="closest"):
        """
        Replace the value with the closest value from Tsd/TsdFrame/TsdTensor argument

        Parameters
        ----------
        data : Tsd, TsdFrame or TsdTensor
            The object holding the values to replace.
        ep : IntervalSet (optional)
            The IntervalSet object to restrict the operation.
            If None, the time support of the tsd input object is used.
        mode: literal, either 'closest', 'before', 'after'
            If closest, replace value with value from Tsd/TsdFrame/TsdTensor, if before gets the
            first value before, if after the first value after.

        Returns
        -------
        out : Tsd, TsdFrame or TsdTensor
            Object with the new values

        Examples
        --------
        In this example, the ts object will receive the closest values in time from tsd.

        >>> import pynapple as nap
        >>> import numpy as np
        >>> t = np.unique(np.sort(np.random.randint(0, 1000, 100))) # random times
        >>> ts = nap.Ts(t=t, time_units='s')
        >>> tsd = nap.Tsd(t=np.arange(0,1000), d=np.random.rand(1000), time_units='s')
        >>> ep = nap.IntervalSet(start = 0, end = 500, time_units = 's')

        The variable ts is a timestamp object.
        The tsd object containing the values, for example the tracking data, and the epoch to restrict the operation.

        >>> newts = ts.value_from(tsd, ep, mode='closest')

        newts is the same size as ts restrict to ep.

        >>> print(len(ts.restrict(ep)), len(newts))
            52 52
        """
        if not isinstance(data, _Base) and not hasattr(data, "values"):
            raise TypeError(
                "First argument should be an instance of Tsd, TsdFrame or TsdTensor"
            )
        if ep is None:
            ep = data.time_support
        if not isinstance(ep, IntervalSet):
            raise TypeError("Argument ep should be of type IntervalSet or None")

        if mode not in ("closest", "before", "after"):
            raise ValueError(
                f'Argument ``mode`` should be "closest", "before", or "after". ``{mode}`` provided instead.'
            )

        time_array = self.index.values
        time_target_array = data.index.values
        data_target_array = data.values
        starts = ep.start
        ends = ep.end

        t, d = _value_from(
            time_array, time_target_array, data_target_array, starts, ends, mode=mode
        )

        time_support = IntervalSet(start=starts, end=ends)

        return data._define_instance(time_index=t, time_support=time_support, values=d)

    def count(self, bin_size=None, ep=None, time_units="s", dtype=None):
        """
        Count occurences of events within bin_size or within a set of bins defined as an IntervalSet.
        You can call this function in multiple ways :

        1. *tsd.count(bin_size=1, time_units = 'ms')*
        -> Count occurence of events within a 1 ms bin defined on the time support of the object.

        2. *tsd.count(1, ep=my_epochs)*
        -> Count occurent of events within a 1 second bin defined on the IntervalSet my_epochs.

        3. *tsd.count(ep=my_bins)*
        -> Count occurent of events within each epoch of the intervalSet object my_bins

        4. *tsd.count()*
        -> Count occurent of events within each epoch of the time support.

        bin_size should be seconds unless specified.
        If bin_size is used and no epochs is passed, the data will be binned based on the time support of the object.

        Parameters
        ----------
        bin_size : None or float, optional
            The bin size (default is second)
        ep : None or IntervalSet, optional
            IntervalSet to restrict the operation
        time_units : str, optional
            Time units of bin size ('us', 'ms', 's' [default])
        dtype: type, optional
            Data type for the count. Default is np.int64.

        Returns
        -------
        out: Tsd
            A Tsd object indexed by the center of the bins.

        Examples
        --------
        This example shows how to count events within bins of 0.1 second.

        >>> import pynapple as nap
        >>> import numpy as np
        >>> t = np.unique(np.sort(np.random.randint(0, 1000, 100)))
        >>> ts = nap.Ts(t=t, time_units='s')
        >>> bincount = ts.count(0.1)

        An epoch can be specified:

        >>> ep = nap.IntervalSet(start = 100, end = 800, time_units = 's')
        >>> bincount = ts.count(0.1, ep=ep)

        And bincount automatically inherit ep as time support:

        >>> bincount.time_support
            start    end
        0  100.0  800.0
        """

        if bin_size is not None:
            if isinstance(bin_size, int):
                bin_size = float(bin_size)
            if not isinstance(bin_size, float):
                raise TypeError("bin_size argument should be float or int.")

        if not isinstance(time_units, str) or time_units not in ["s", "ms", "us"]:
            raise ValueError("time_units argument should be 's', 'ms' or 'us'.")

        if ep is None:
            ep = self.time_support
        if not isinstance(ep, IntervalSet):
            raise TypeError("ep argument should be of type IntervalSet")

        if dtype is None:
            dtype = np.dtype(np.int64)
        else:
            try:
                dtype = np.dtype(dtype)
            except Exception:
                raise ValueError(f"{dtype} is not a valid numpy dtype.")

        starts = ep.start
        ends = ep.end

        if isinstance(bin_size, (float, int)):
            bin_size = TsIndex.format_timestamps(np.array([bin_size]), time_units)[0]

        time_array = self.index.values

        t, d = _count(time_array, starts, ends, bin_size, dtype=dtype)

        return self._define_instance(t, ep, values=d)

    def time_diff(self, align="center", epochs=None):
        """
        Computes the differences between subsequent timestamps.

        Parameters
        ----------
        align: str, optional
            Determines the time index of the resulting time differences:
             - "start" : the start of the interval between two timestamps.
             - "center" [default]: the center of the interval between two timestamps.
             - "end" : the end of the interval between two timestamps.
        epochs : IntervalSet, optional
            The epochs on which interspike intervals are computed.
            If None, the time support of the input is used.

        Returns
        -------
        Tsd
            The time differences.

        """
        if align not in ["start", "center", "end"]:
            raise RuntimeError("align should be 'start', 'center' or 'end'")

        if epochs is None:
            epochs = self.time_support
        else:
            if not isinstance(epochs, IntervalSet):
                raise TypeError("epochs should be an object of type IntervalSet")

        n = max(len(self) - 1, 0)
        new_d = np.empty(n)
        new_t = np.empty(n)

        start = 0
        alpha = 0.0 if align == "start" else 0.5 if align == "center" else 1.0
        for i in range(len(epochs)):
            tmp = self.get(epochs[i, 0], epochs[i, 1])

            if len(tmp) > 1:
                diff = tmp.index.values[1:] - tmp.index.values[:-1]
                new_d[start : start + len(tmp) - 1] = diff
                new_t[start : start + len(tmp) - 1] = (
                    tmp.index.values[:-1] + alpha * diff
                )
                start += len(tmp) - 1

        return self._define_instance(
            time_index=new_t[:start], time_support=epochs, values=new_d[:start]
        )

    def restrict(self, iset):
        """
        Restricts a time series object to a set of time intervals delimited by an IntervalSet object

        Parameters
        ----------
        iset : IntervalSet
            the IntervalSet object

        Returns
        -------
        Ts, Tsd, TsdFrame or TsdTensor
            Tsd object restricted to ep

        Examples
        --------
        The Ts object is restrict to the intervals defined by ep.

        >>> import pynapple as nap
        >>> import numpy as np
        >>> t = np.unique(np.sort(np.random.randint(0, 1000, 100)))
        >>> ts = nap.Ts(t=t, time_units='s')
        >>> ep = nap.IntervalSet(start=0, end=500, time_units='s')
        >>> newts = ts.restrict(ep)

        The time support of newts automatically inherit the epochs defined by ep.

        >>> newts.time_support
            start    end
        0    0.0  500.0

        """
        if not isinstance(iset, IntervalSet):
            raise TypeError("Argument should be IntervalSet")

        time_array = self.index.values
        starts = iset.start
        ends = iset.end

        idx = _restrict(time_array, starts, ends)
        data = None if not hasattr(self, "values") else self.values[idx]
        return self._define_instance(time_array[idx], iset, values=data)

    def copy(self):
        """Copy the data, index and time support"""
        data = getattr(self, "values", None)
        if data is not None:
            data = data.copy() if hasattr(data, "copy") else data[:].copy()
        return self._define_instance(self.index.copy(), self.time_support, values=data)

    def find_support(self, min_gap, time_units="s"):
        """
        find the smallest (to a min_gap resolution) IntervalSet containing all the times in the Tsd

        Parameters
        ----------
        min_gap : float or int
            minimal interval between timestamps
        time_units : str, optional
            Time units of min gap

        Returns
        -------
        IntervalSet
            Description
        """
        assert isinstance(min_gap, Number), "min_gap should be a float or int"
        min_gap = TsIndex.format_timestamps(np.array([min_gap]), time_units)[0]
        time_array = self.index.values

        starts = [time_array[0]]
        ends = []
        for i in range(len(time_array) - 1):
            if (time_array[i + 1] - time_array[i]) > min_gap:
                ends.append(time_array[i] + 1e-6)
                starts.append(time_array[i + 1])

        ends.append(time_array[-1] + 1e-6)

        return IntervalSet(start=starts, end=ends)

    def get(self, start, end=None, time_units="s"):
        """Slice the time series from `start` to `end` such that all the timestamps satisfy `start<=t<=end`.
        If `end` is None, only the timepoint closest to `start` is returned.

        By default, the time support doesn't change. If you want to change the time support, use the `restrict` function.

        Parameters
        ----------
        start : float or int
            The start (or closest time point if `end` is None)
        end : float or int or None
            The end
        """
        sl = self.get_slice(start, end, time_units)

        if end is None:
            sl = sl.start

        return self[sl]

    def get_slice(self, start, end=None, time_unit="s"):
        """
        Get a slice object from the time series data based on the start and end values such that all the timestamps satisfy `start<=t<=end`.
        If `end` is None, only the timepoint closest to `start` is returned.

        By default, the time support doesn't change. If you want to change the time support, use the `restrict` function.

        This function is equivalent of calling the `get` method.

        Parameters
        ----------
        start : int or float
            The starting value for the slice.
        end : int or float, optional
            The ending value for the slice. Defaults to None.
        time_unit : str, optional
            The time unit for the start and end values. Defaults to "s" (seconds).

        Returns
        -------
        slice : slice
            A slice determining the start and end indices, with unit step
            Slicing the array will be equivalent to calling get: `ts[s].t == ts.get(start, end).t` with `s` being the slice object.


        Raises
        ------
        ValueError
            - If start or end is not a number.
            - If start is greater than end.

        Examples
        --------
        >>> import pynapple as nap

        >>> ts = nap.Ts(t = [0, 1, 2, 3])

        >>> # slice over a range
        >>> start, end = 1.2, 2.6
        >>> print(ts.get_slice(start, end))  # returns `slice(2, 3, None)`
        >>> start, end = 1., 2.
        >>> print(ts.get_slice(start, end, mode="forward"))  # returns `slice(1, 3, None)`

        >>> # slice a single value
        >>> start = 1.2
        >>> print(ts.get_slice(start))  # returns `slice(1, 2, None)`
        >>> start = 2.
        >>> print(ts.get_slice(start)) # returns `slice(2, 3, None)`
        """
        mode = "closest_t" if end is None else "restrict"
        return self._get_slice(
            start, end=end, mode=mode, n_points=None, time_unit=time_unit
        )

    def _get_slice(
        self, start, end=None, mode="closest_t", n_points=None, time_unit="s"
    ):
        """
        Get a slice from the time series data based on the start and end values with the specified mode.

        For a given time t, mode `before_t` means you want the time point right before t to start the slice.
        Mode `after_t` means you want the time point right after t to start the slice.

        Parameters
        ----------
        start : int or float
            The starting value for the slice.
        end : int or float, optional
            The ending value for the slice. Defaults to None.
        mode : str, optional
            The mode for slicing. Can be "after_t", "before_t", "restrict", or "closest_t". Defaults to "closest_t".
        time_unit : str, optional
            The time unit for the start and end values. Defaults to "s" (seconds).
        n_points : int, optional
            Number of time point that will result from applying the slice. This parameter is used to
            calculate a step size for the slice.

        Returns
        -------
        slice : slice
            If end is not provided:
                - For mode == "before_t":
                    - An empty slice for start < self.t[0]
                    - slice(idx, idx+1) with self.t[idx] <= start < self.t[idx+1]
                - For mode == "after_t":
                    - An empty slice for start >= self.t[-1]
                    - slice(idx, idx+1) with self.t[idx-1] < start <= self.t[idx]
                - For mode == "closest_t":
                    - slice(idx, idx+1) with the closest index to start
                - For mode == "restrict":
                    - slice the indices such that start <= self.t[idx] <= end
            If end is provided:
                - For mode == "before_t":
                    - An empty slice if end < self.t[0]
                    - slice(idx_start, idx_end) with self.t[idx_start] <= start < self.t[idx_start+1] and
                    self.t[idx_end] <= end < self.t[idx_end+1]
                - For mode == "after_t":
                    - An empty slice if start > self.t[-1]
                     - slice(idx_start, idx_end) with self.t[idx_start-1] <= start < self.t[idx_start] and
                    self.t[idx_end-1] <= end < self.t[idx_end]
                - For mode == "closest":
                    - slice(idx_start, idx_end) with the closest indices to start and end
                - For mode == "restrict":
                    - An empty slice if start > self.t[-1] or end < self.t[0]
                    - slice(idx_start, idx_end) with self.t[idx_start] <= start <= self.t[idx_start+1] and
                    self.t[idx_end] <= end <= self.t[idx_end+1]

        Raises
        ------
        ValueError
            - If start or end is not a number.
            - If start is greater than end.

        """
        if not isinstance(start, Number):
            raise ValueError(
                f"'start' must be an int or a float. Type {type(start)} provided instead!"
            )

        if n_points is not None and not isinstance(n_points, int):
            raise TypeError(
                f"'n_points' must be of type int or None. Type {type(n_points)} provided instead!"
            )

        if end is None and n_points:
            raise ValueError("'n_points' can be used only when 'end' is specified!")

        if mode not in ["before_t", "after_t", "closest_t", "restrict"]:
            raise ValueError(
                "'mode' only accepts 'before_t', 'after_t', 'closest_t' or 'restrict'."
            )

        if mode == "restrict" and n_points:
            raise ValueError(
                "Fixing the number of time points is incompatible with 'restrict' mode."
            )

        # convert and get index for start
        start = TsIndex.format_timestamps(np.array([start]), time_unit)[0]

        # check end
        if end is not None and not isinstance(end, Number):
            raise ValueError(
                f"'end' must be an int or a float. Type {type(end)} provided instead!"
            )

        # get index of preceding time value
        idx_start = np.searchsorted(self.t, start, side="left")
        if idx_start == len(self.t) and mode != "restrict":
            idx_start -= 1  # make sure the index is not out of bound

        if mode == "before_t":
            # in order to get the index preceding start
            # subtract one except if self.t[idx_start] is exactly equal to start
            idx_start -= self.t[idx_start] > start
        elif mode == "closest_t":
            # subtract 1 if start is closer to the previous index
            di = self.t[idx_start] - start > np.abs(self.t[idx_start - 1] - start)
            idx_start -= di

        if end is None:
            if idx_start < 0:  # happens only on backwards if start < self.t[0]
                return slice(0, 0)
            elif (
                idx_start == len(self.t) - 1 and mode == "after_t"
            ):  # happens only on forward if start >= self.t[-1]
                return slice(idx_start, idx_start)
            return slice(idx_start, idx_start + 1)
        else:
            idx_start = max([0, idx_start])  # if taking a range set slice index to 0

        # convert and get index for end
        end = TsIndex.format_timestamps(np.array([end]), time_unit)[0]
        if start > end:
            raise ValueError("'start' should not precede 'end'.")

        idx_end = np.searchsorted(self.t, end, side="left")
        add_if_forward = 0
        if idx_end == len(self.t):
            idx_end -= 1  # make sure the index is not out of bound
            add_if_forward = 1  # add back the index if forward

        if mode == "before_t":
            # remove 1 if self.t[idx_end] is larger than end, except if idx_end is 0
            idx_end -= (self.t[idx_end] > end) - int(idx_end == 0)
        elif mode == "closest_t":
            # subtract 1 if end is closer to self.t[idx_end - 1]
            di = self.t[idx_end] - end > np.abs(self.t[idx_end - 1] - end)
            idx_end -= di
        elif mode == "after_t" and idx_end == len(self.t) - 1:
            idx_end += add_if_forward  # add one if idx_start < len(self.t)
        elif mode == "restrict":
            idx_end += int(self.t[idx_end] <= end)

        step = None
        if n_points:
            tot_tps = idx_end - idx_start
            if tot_tps > n_points:
                rounding = tot_tps % n_points
                step = tot_tps // n_points
                idx_end -= rounding

        return slice(idx_start, idx_end, step)

    def _get_filename(self, filename):
        """Check if the filename is valid and return the path

        Parameters
        ----------
        filename : str or Path
            The filename

        Returns
        -------
        Path
            The path to the file

        Raises
        ------
        RuntimeError
            If the filename is a directory or the parent does not exist
        """

        return check_filename(filename)

    @classmethod
    def _from_npz_reader(cls, file):
        """Load a time series object from a npz file interface.

        Parameters
        ----------
        file : NPZFile object
            opened npz file interface.

        Returns
        -------
        out : Ts or Tsd or TsdFrame or TsdTensor
            The time series object
        """
        kwargs = {
            key: file[key]
            for key in file.keys()
            if key not in ["start", "end", "type", "_metadata"]
        }
        iset = IntervalSet(start=file["start"], end=file["end"])
        ts = cls(time_support=iset, **kwargs)
        if "_metadata" in file:  # load metadata if it exists
            if file["_metadata"]:  # check if metadata is not empty
                m = file["_metadata"].item()
                # check if first field is a dictionary, meaning it was saved from a pandas.DataFrame
                if isinstance(next(iter(m.values())), dict):
                    import pandas as pd

                    m = pd.DataFrame.from_dict(m)

                ts.set_info(m)
        return ts
