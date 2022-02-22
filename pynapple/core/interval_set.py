# -*- coding: utf-8 -*-
# @Date:   2022-01-25 21:50:48
# @Last Modified by:   gviejo
# @Last Modified time: 2022-02-16 16:56:45

"""
"""

import warnings
import pandas as pd
import numpy as np
from .time_units import TimeUnits

def _join_helper(start, end):
    time = np.hstack((start, end))
    start_end = np.hstack((np.ones(len(time)//2, dtype=np.int32),
                          -1 * np.ones(len(time)//2, dtype=np.int32)))
    df = pd.DataFrame({'time': time, 'start_end': start_end})
    df.sort_values(by='time', inplace=True)
    df.reset_index(inplace=True, drop=True)
    df['cumsum'] = df['start_end'].cumsum()
    ix_stop = (df['cumsum']==0).to_numpy().nonzero()[0]
    ix_start = np.hstack((0, ix_stop[:-1]+1))
    start = df['time'][ix_start].values
    end = df['time'][ix_stop].values
    return start, end    


class IntervalSet(pd.DataFrame):
    # class IntervalSet():
    """
    A subclass of pandas.DataFrame representing a (irregular) set of time intervals in elapsed time,
    with relative operations
    """
    def __init__(self, start, end = None, time_units='s', expect_fix=False, **kwargs):
        """
        IntervalSet initializer
        
        If start and end and not aligned, meaning that len(start) == len(end),
        end[i] > start[i] and start[i+1] > end[i], or start and end are not sorted,
        will try to "fix" the data by eliminating some of the start and end data point
        
        Parameters
        ----------
        start : numpy.ndarray or number
            Beginning of intervals
        end : numpy.ndarray or number, optional
            Ends of intervals
        time_units : str, optional
            Time unit of the intervals ('us', 'ms', 's' [default])
        expect_fix : bool, optional
            if False, will give a warning when a fix is needed (default: False)
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
            if 'start' not in df.columns or 'end' not in df.columns:
                raise ValueError('wrong columns')
            start = df['start'].values
            end = df['end'].values
            start = TimeUnits.format_timestamps(start.ravel(), time_units,
                                                give_warning=False)
            end = TimeUnits.format_timestamps(end.ravel(), time_units,
                                              give_warning=False)            
            if (start[1:] < end[0:-1]).any():
                start, end = _join_helper(start, end)
                df = pd.DataFrame.from_dict({'start':start,'end':end})
            else:
                df = pd.DataFrame.from_dict({'start':start,'end':end})
            super().__init__(df, **kwargs)
            self.r_cache = None
            self._metadata = ['nap_class']
            self.nap_class = self.__class__.__name__
            return

        start = np.array(start).ravel()
        end = np.array(end).ravel()

        if len(start) != len(end):
            raise RuntimeError("Starts end ends are not of the same length")

        if not (np.diff(start) > 0).all():
            warnings.warn("start is not sorted.", stacklevel=2)
            start = np.sort(start)

        if not (np.diff(end) > 0).all():
            warnings.warn("end is not sorted.", stacklevel=2)
            end = np.sort(end)

        start = TimeUnits.format_timestamps(start.ravel(), time_units,
                                            give_warning=False)
        end = TimeUnits.format_timestamps(end.ravel(), time_units,
                                          give_warning=False)


        if (start[1:] == end[0:-1]).any():
            warnings.warn("Some starts and ends are equal. Removing 1 microsecond!", stacklevel=2)
            idx = np.where((start[1:] == end[0:-1]))[0]
            end[idx] -= 1.0e-6 # removing 1 microseconds because bounds are both closed
            idx2 = start != end
            start = start[idx2]
            end = end[idx2]

        if (start > end).any():
            warnings.warn("Some ends precede the relative start. Dropping them!", stacklevel=2)
            idx = start < end
            start = start[idx]
            end = end[idx]            

        if (start[1:] < end[0:-1]).any():
            warnings.warn("Some starts precede the previous end. Joining them!", stacklevel=2)
            time = np.hstack((start, end))
            start_end = np.hstack((np.ones(len(time)//2, dtype=np.int32),
                                  -1 * np.ones(len(time)//2, dtype=np.int32)))
            df = pd.DataFrame({'time': time, 'start_end': start_end})
            df.sort_values(by='time', inplace=True)
            df.reset_index(inplace=True, drop=True)
            df['cumsum'] = df['start_end'].cumsum()
            ix_stop = (df['cumsum']==0).to_numpy().nonzero()[0]
            ix_start = np.hstack((0, ix_stop[:-1]+1))
            start = df['time'][ix_start].values
            end = df['time'][ix_stop].values

        if (start == end).any():
            warnings.warn("Some epochs have no duration", stacklevel=2)
            idx = start != end
            start = start[idx]
            end = end[idx]


        data = np.vstack((start, end)).T

        super().__init__(data=data, columns=('start', 'end'), **kwargs)
        self.r_cache = None
        self._metadata = ['nap_class']
        self.nap_class = self.__class__.__name__

    def __repr__(self):
        return self.as_units('s').__repr__()
                
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
        s = self['start'][0]
        e = self['end'].iloc[-1]
        return IntervalSet(s, e)

    def tot_length(self, time_units='s'):
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
        tot_l = (self['end'] - self['start']).astype(np.float64).sum()
        return TimeUnits.return_timestamps(tot_l, time_units)

    def intersect(self, a):
        """
        set intersection of IntervalSet's
                
        Parameters
        ----------
        a : IntervalSet or list/tuple of IntervalSets
            the IntervalSet to intersect self with, or a tuple of
        
        Returns
        -------
        out: IntervalSet
            _
        """
        if isinstance(a, IntervalSet):
            a = [a]
        elif isinstance(a, (list, tuple)):
            a = list(a)

        i_sets = [self]
        i_sets.extend(a)
        n_sets = len(i_sets)
        time1 = [i_set['start'] for i_set in i_sets]
        time2 = [i_set['end'] for i_set in i_sets]
        time1.extend(time2)
        time = np.hstack(time1)

        start_end = np.hstack((np.ones(len(time)//2, dtype=np.int32),
                              -1 * np.ones(len(time)//2, dtype=np.int32)))

        df = pd.DataFrame({'time': time, 'start_end': start_end})
        df.sort_values(by='time', inplace=True)
        df.reset_index(inplace=True, drop=True)
        df['cumsum'] = df['start_end'].cumsum()
        # noinspection PyTypeChecker
        # ix = np.nonzero(df['cumsum'] == n_sets)[0]
        ###################################################################################
        ix = (df['cumsum']==n_sets).to_numpy().nonzero()[0]
        ###################################################################################
        start = df['time'][ix]
        # noinspection PyTypeChecker
        end = df['time'][ix+1]

        return IntervalSet(start, end)

    def union(self, a):
        """
        set union of IntervalSet's
                
        Parameters
        ----------
        a : IntervalSet or list/tuple of IntervalSets            
            the IntervalSet to union self with, or a tuple of 
        
        Returns
        -------
        out: IntervalSet
            _
        """
        if isinstance(a, IntervalSet):
            a = [a]
        elif isinstance(a, (list, tuple)):
            a = list(a)

        i_sets = [self]
        i_sets.extend(a)        
        time = np.hstack([i_set['start'] for i_set in i_sets] +
                         [i_set['end'] for i_set in i_sets])

        start_end = np.hstack((np.ones(len(time)//2, dtype=np.int32),
                              -1 * np.ones(len(time)//2, dtype=np.int32)))

        df = pd.DataFrame({'time': time, 'start_end': start_end})
        df.sort_values(by='time', inplace=True)
        df.reset_index(inplace=True, drop=True)
        df['cumsum'] = df['start_end'].cumsum()
        # noinspection PyTypeChecker
        #ix_stop = np.nonzero(df['cumsum'] == 0)[0]
        ###################################################################################
        ix_stop = (df['cumsum']==0).to_numpy().nonzero()[0]
        ###################################################################################
        ix_start = np.hstack((0, ix_stop[:-1]+1))
        start = df['time'][ix_start]
        stop = df['time'][ix_stop]

        return IntervalSet(start, stop)

    def set_diff(self, a):
        """
        set difference of IntervalSet
                
        Parameters
        ----------
        a : IntervalSet or tuple of IntervalSets
            the IntervalSet to set-substract from self
        
        Returns
        -------
        out: IntervalSet
            _
        """
        i_sets = (self, a)
        time = np.hstack([i_set['start'] for i_set in i_sets] +
                         [i_set['end'] for i_set in i_sets])
        start_end1 = np.hstack((np.ones(len(i_sets[0]), dtype=np.int32),
                                -1 * np.ones(len(i_sets[0]), dtype=np.int32)))
        start_end2 = np.hstack((-1 * np.ones(len(i_sets[1]), dtype=np.int32),
                                np.ones(len(i_sets[1]), dtype=np.int32)))
        start_end = np.hstack((start_end1, start_end2))
        df = pd.DataFrame({'time': time, 'start_end': start_end})
        df.sort_values(by='time', inplace=True)
        df.reset_index(inplace=True, drop=True)
        df['cumsum'] = df['start_end'].cumsum()
        # noinspection PyTypeChecker
        #ix = np.nonzero(df['cumsum'] == 1)[0] # CHANGED BY G.VIEJO 22/04/2020        
        ###################################################################################        
        ix = (df['cumsum']==1).to_numpy().nonzero()[0]        
        ################################################################################### 
        start = df['time'][ix]
        # noinspection PyTypeChecker
        end = df['time'][ix+1]        
        start = start.reset_index(drop=True)  
        end = end.reset_index(drop=True)  
        idx = start!=end

        return IntervalSet(start[idx], end[idx])

    def in_interval(self, tsd):
        """
        finds out in which element of the interval set each point in a time series fits.
        
        NaNs for those that don't fit an interval

        IntervalSet.
        
        Parameters
        ----------
        tsd : TSD
            The tsd to be binned
        
        Returns
        -------
        out: numpy.ndarray
            an array with the interval index labels for each time stamp (NaN) for timestamps not in IntervalSet
        """
        bins = self.values.ravel()
        # Because yes there is no funtion with both bounds closed as an option
        ix = np.array(pd.cut(tsd.index, bins, labels=np.arange(len(bins) - 1, dtype=np.float64)))
        ix2 = np.array(pd.cut(tsd.index, bins, labels=np.arange(len(bins) - 1, dtype=np.float64), right=False))
        ix3 = np.vstack((ix, ix2)).T
        # ix[np.floor(ix / 2) * 2 != ix] = np.NaN
        # ix = np.floor(ix/2)
        ix3[np.floor(ix3 / 2) * 2 != ix3] = np.NaN
        ix3 = np.floor(ix3/2)
        ix3[np.isnan(ix3[:,0]),0] = ix3[np.isnan(ix3[:,0]),1]
        return ix3[:,0]

    def drop_short_intervals(self, threshold, time_units='s'):
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
        threshold = TimeUnits.format_timestamps(np.array((threshold,), dtype=np.float64).ravel(), time_units)[0]
        return self.loc[(self['end']-self['start']) > threshold].reset_index(drop=True)

    def drop_long_intervals(self, threshold, time_units='s'):
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
        threshold = TimeUnits.format_timestamps(np.array((threshold,), dtype=np.float64).ravel(), time_units)[0]
        return self.loc[(self['end']-self['start']) < threshold].reset_index(drop=True)

    def as_units(self, units='s'):
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
        data = TimeUnits.return_timestamps(data, units)
        if units == 'us':
            data = data.astype(np.int64)

        df = pd.DataFrame(index = self.index.values, data=data, columns=self.columns)

        return df

    def merge_close_intervals(self, threshold, time_units='s'):
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

        threshold = TimeUnits.format_timestamps(np.array((threshold,), dtype=np.float64).ravel(), time_units)[0]
        start = self['start'].values
        end = self['end'].values
        tojoin = (start[1:] - end[0:-1]) > threshold
        start = np.hstack((start[0], start[1:][tojoin]))
        end = np.hstack((end[0:-1][tojoin], end[-1]))

        return IntervalSet(start = start, end = end)

    def store(self, the_store, key, **kwargs):
        data_to_store = pd.DataFrame(self)
        the_store[key] = data_to_store
        # noinspection PyProtectedMember
        metadata = {k: getattr(self, k) for k in self._metadata}
        the_store.put(key, data_to_store, metadata, **kwargs)

    @property
    def _constructor(self):
        return IntervalSet
