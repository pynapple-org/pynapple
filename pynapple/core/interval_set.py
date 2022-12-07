# -*- coding: utf-8 -*-
# @Date:   2022-01-25 21:50:48
# @Last Modified by:   gviejo
# @Last Modified time: 2022-12-06 21:27:08

"""
"""

import warnings

import numpy as np
import pandas as pd
from numba import jit

from .jitted_functions import jitdiff, jitin_interval, jitintersect, jitunion
from .time_units import format_timestamps, return_timestamps, sort_timestamps

all_warnings = np.array(
    [
        "Some starts and ends are equal. Removing 1 microsecond!",
        "Some ends precede the relative start. Dropping them!",
        "Some starts precede the previous end. Joining them!",
        "Some epochs have no duration",
    ]
)


@jit(nopython=True)
def jitfix_iset(start, end):
    """
    0 - > "Some starts and ends are equal. Removing 1 microsecond!",
    1 - > "Some ends precede the relative start. Dropping them!",
    2 - > "Some starts precede the previous end. Joining them!",
    3 - > "Some epochs have no duration"

    Parameters
    ----------
    start : TYPE
        Description
    end : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    to_warn = np.zeros(4, dtype=np.bool_)

    m = start.shape[0]

    data = np.zeros((m, 2), dtype=np.float64)

    i = 0
    ct = 0

    while i < m:

        newstart = start[i]
        newend = end[i]

        while i < m:
            if end[i] == start[i]:
                to_warn[3] = True
                i += 1
            else:
                newstart = start[i]
                newend = end[i]
                break

        while i < m:
            if end[i] < start[i]:
                to_warn[1] = True
                i += 1
            else:
                newstart = start[i]
                newend = end[i]
                break

        while i < m - 1:

            if start[i + 1] < end[i]:
                to_warn[2] = True
                i += 1
                newend = max(end[i - 1], end[i])
            else:
                break

        if i < m - 1:
            if newend == start[i + 1]:
                to_warn[0] = True
                newend -= 1.0e-6

        data[ct, 0] = newstart
        data[ct, 1] = newend

        ct += 1
        i += 1

    data = data[0:ct]

    return (data, to_warn)


class IntervalSet(pd.DataFrame):
    # class IntervalSet():
    """
    A subclass of pandas.DataFrame representing a (irregular) set of time intervals in elapsed time, with relative operations
    """

    def __init__(self, start, end=None, time_units="s", **kwargs):
        """
        IntervalSet initializer

        If start and end and not aligned, meaning that \n
        1. len(start) != len(end)
        2. end[i] > start[i]
        3. start[i+1] > end[i]
        4. start and end are not sorted,

        IntervalSet will try to "fix" the data by eliminating some of the start and end data point

        Parameters
        ----------
        start : numpy.ndarray or number or pandas.DataFrame
            Beginning of intervals
        end : numpy.ndarray or number, optional
            Ends of intervals
        time_units : str, optional
            Time unit of the intervals ('us', 'ms', 's' [default])
        **kwargs
            Additional parameters passed ot pandas.DataFrame

        Returns
        -------
        IntervalSet
            _

        Raises
        ------
        RuntimeError
            Description
        ValueError
            If a pandas.DataFrame is passed, it should contains
            a column 'start' and a column 'end'.

        """

        if end is None:
            df = pd.DataFrame(start)
            if "start" not in df.columns or "end" not in df.columns:
                raise ValueError("wrong columns name")
            start = df["start"].values.astype(np.float64)
            end = df["end"].values.astype(np.float64)

            start = sort_timestamps(format_timestamps(start.ravel(), time_units))
            end = sort_timestamps(format_timestamps(end.ravel(), time_units))

            data, to_warn = jitfix_iset(start, end)
            if np.any(to_warn):
                msg = "\n".join(all_warnings[to_warn])
                warnings.warn(msg, stacklevel=2)
            super().__init__(data=data, columns=("start", "end"), **kwargs)
            self.r_cache = None
            self._metadata = ["nap_class"]
            self.nap_class = self.__class__.__name__
            return

        start = np.array(start).astype(np.float64)
        end = np.array(end).astype(np.float64)

        start = format_timestamps(np.array(start).ravel(), time_units)
        end = format_timestamps(np.array(end).ravel(), time_units)

        if len(start) != len(end):
            raise RuntimeError("Starts end ends are not of the same length")

        if not (np.diff(start) > 0).all():
            warnings.warn("start is not sorted.", stacklevel=2)
            start = np.sort(start)

        if not (np.diff(end) > 0).all():
            warnings.warn("end is not sorted.", stacklevel=2)
            end = np.sort(end)

        data, to_warn = jitfix_iset(start, end)

        if np.any(to_warn):
            msg = "\n".join(all_warnings[to_warn])
            warnings.warn(msg, stacklevel=2)

        super().__init__(data=data, columns=("start", "end"), **kwargs)
        self.r_cache = None
        # self._metadata = ["nap_class"]
        self.nap_class = self.__class__.__name__

    def __repr__(self):
        return self.as_units("s").__repr__()

    def __str__(self):
        return self.__repr__()

    def time_span(self):
        """
        Time span of the interval set.

        Returns
        -------
        out: IntervalSet
            an IntervalSet with a single interval encompassing the whole IntervalSet
        """
        s = self["start"][0]
        e = self["end"].iloc[-1]
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
        tot_l = (self["end"] - self["start"]).sum()
        return return_timestamps(np.array([tot_l]), time_units)[0]

    def intersect(self, a):
        """
        set intersection of IntervalSet

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
        s, e = jitintersect(start1, end1, start2, end2)
        return IntervalSet(s, e)

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
        s, e = jitdiff(start1, end1, start2, end2)
        return IntervalSet(s, e)

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
        Drops the short intervals in the interval set.

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
        threshold = format_timestamps(
            np.array([threshold], dtype=np.float64), time_units
        )[0]
        return self.loc[(self["end"] - self["start"]) > threshold].reset_index(
            drop=True
        )

    def drop_long_intervals(self, threshold, time_units="s"):
        """
        Drops the long intervals in the interval set.

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
        threshold = format_timestamps(
            np.array([threshold], dtype=np.float64), time_units
        )[0]
        return self.loc[(self["end"] - self["start"]) < threshold].reset_index(
            drop=True
        )

    def as_units(self, units="s"):
        """
        returns a DataFrame with time expressed in the desired unit

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
        data = return_timestamps(data, units)
        if units == "us":
            data = data.astype(np.int64)

        df = pd.DataFrame(index=self.index.values, data=data, columns=self.columns)

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
        if len(self) == 0:
            return IntervalSet(start=[], end=[])

        threshold = format_timestamps(
            np.array((threshold,), dtype=np.float64).ravel(), time_units
        )[0]
        start = self["start"].values
        end = self["end"].values
        tojoin = (start[1:] - end[0:-1]) > threshold
        start = np.hstack((start[0], start[1:][tojoin]))
        end = np.hstack((end[0:-1][tojoin], end[-1]))

        return IntervalSet(start=start, end=end)

    @property
    def _constructor(self):
        return IntervalSet
