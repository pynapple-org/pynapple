# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-27 18:33:31
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-07-10 17:28:36

from calendar import c
import importlib
import os
import warnings

import numpy as np
import pandas as pd
from pandas.core.internals import BlockManager, SingleBlockManager

from .interval_set import IntervalSet
from .jitted_functions import (
    jitbin,
    jitbin_array,
    jitcount,
    jitrestrict,
    jitthreshold,
    jittsrestrict,
    jittsrestrict_with_count,
    jitvaluefrom,
    jitvaluefromtsdframe,
)
from .time_units import format_timestamps, return_timestamps, sort_timestamps

class TimeDataMixin:
    """Class that implements the common features that we want in Tsd and TsdFrames and 
    that can be handled in a single place. This is a mixin class, so it should be used
    only in conjunction with a subclass of pandas.Series or pandas.DataFrame.
    """
    
    def times(self, units="s"):
        """
        The time index of the Tsd, returned as np.double in the desired time units.

        Parameters
        ----------
        units : str, optional
            ('us', 'ms', 's' [default])

        Returns
        -------
        out: numpy.ndarray
            the time indexes
        """
        return return_timestamps(self.index.values, units=units)

    def start_time(self, units="s"):
        """
        The first time index in the Ts/Tsd object

        Parameters
        ----------
        units : str, optional
            ('us', 'ms', 's' [default])

        Returns
        -------
        out: numpy.float64
            _
        """
        return self.times(units=units)[0]

    def end_time(self, units="s"):
        """
        The last time index in the Ts/Tsd object

        Parameters
        ----------
        units : str, optional
            ('us', 'ms', 's' [default])

        Returns
        -------
        out: numpy.float64
            _
        """
        return self.times(units=units)[-1]
    
    def as_units(self, units="s"):
        """
        Returns a Series with time expressed in the desired unit.

        Parameters
        ----------
        units : str, optional
            ('us', 'ms', 's' [default])

        Returns
        -------
        pandas.Series
            the series object with adjusted times
        """
        if isinstance(self, TsdFrame):
            ss = self.as_dataframe()
        elif isinstance(self, Tsd):
            ss = self.as_series()
        
        t = self.index.values
        t = return_timestamps(t, units)
        if units == "us":
            t = t.astype(np.int64)
        ss.index = t
        ss.index.name = "Time (" + str(units) + ")"
        return ss

    def data(self):
        """
        The data in the Tsd object

        Returns
        -------
        out: numpy.ndarray
            _
        """
        return self.values
    
    def value_from(self, data, ep=None):
        """
        Replace the value with the closest value from Tsd/TsdFrame argument
        If data is TsdFrame, the output is also TsdFrame.

        Parameters
        ----------
        data : Tsd/TsdFrame
            The Tsd/TsdFrame object holding the values to replace.
        ep : IntervalSet (optional)
            The IntervalSet object to restrict the operation.
            If None, the time support of the tsd input object is used.

        Returns
        -------
        out : Tsd/TsdFrame
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

        The variable ts is a time series object containing only nan.
        The tsd object containing the values, for example the tracking data, and the epoch to restrict the operation.

        >>> newts = ts.value_from(tsd, ep)

        newts is the same size as ts restrict to ep.

        >>> print(len(ts.restrict(ep)), len(newts))
            52 52
        """
        if not isinstance(data, (Tsd, TsdFrame)):
            raise ValueError("The time series to align to should be Tsd/TsdFrame.")
        
        time_array = self.index.values
        
        if ep is None:
            ep = self.time_support    
        starts = ep.start.values
        ends = ep.end.values
        time_target_array = data.index.values
        data_target_array = data.values
       
        # TODO those should be refactores as "values_to" methods in the respective
        # classes, confusing to have this typecheck here:
        if isinstance(data, Tsd):
            t, d, ns, ne = jitvaluefrom(
                time_array, time_target_array, data_target_array, starts, ends
            )
            time_support = IntervalSet(start=ns, end=ne)

            return Tsd(t=t, d=d, time_support=time_support)

        if isinstance(data, TsdFrame):
            t, d, ns, ne = jitvaluefromtsdframe(
                time_array, time_target_array, data_target_array, starts, ends
            )
            time_support = IntervalSet(start=ns, end=ne)

            return TsdFrame(t=t, d=d, time_support=time_support, columns=data.columns)
        
    def bin_average(self, bin_size, ep=None, time_units="s"):
        """
        Bin the data by averaging points within bin_size
        bin_size should be seconds unless specified.
        If no epochs is passed, the data will be binned based on the time support.

        Parameters
        ----------
        bin_size : float
            The bin size (default is second)
        ep : None or IntervalSet, optional
            IntervalSet to restrict the operation
        time_units : str, optional
            Time units of bin size ('us', 'ms', 's' [default])

        Returns
        -------
        out: TsdFrame
            A TsdFrame object indexed by the center of the bins and holding the averaged data points.

        Examples
        --------
        This example shows how to bin data within bins of 0.1 second.

        >>> import pynapple as nap
        >>> import numpy as np
        >>> tsdframe = nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 3))
        >>> bintsdframe = tsdframe.bin_average(0.1)

        An epoch can be specified:

        >>> ep = nap.IntervalSet(start = 10, end = 80, time_units = 's')
        >>> bintsdframe = tsdframe.bin_average(0.1, ep=ep)

        And bintsdframe automatically inherit ep as time support:

        >>> bintsdframe.time_support
        >>>    start    end
        >>> 0  10.0     80.0
        """
        if not isinstance(ep, IntervalSet):
            ep = self.time_support

        bin_size = format_timestamps(np.array([bin_size]), time_units)[0]

        time_array = self.index.values
        data_array = self.values
        starts = ep.start.values
        ends = ep.end.values
        t, d = jitbin_array(time_array, data_array, starts, ends, bin_size)
        time_support = IntervalSet(start=starts, end=ends)
        if isinstance(self, TsdFrame):
            return TsdFrame(t=t, d=d, time_support=time_support, columns=self.columns)
        elif isinstance(self, Tsd):
            return Tsd(t=t, d=d, time_support=time_support)
        else:
            raise RuntimeError("The time series to bin should be Tsd/TsdFrame.")
        
    def restrict(self, ep):
        """
        Restricts a Tsd object to a set of time intervals delimited by an IntervalSet object

        Parameters
        ----------
        ep : IntervalSet
            the IntervalSet object

        Returns
        -------
        out: Tsd
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
        >>>    start    end
        >>> 0    0.0  500.0

        """
        time_array = self.index.values
        data_array = self.values
        starts = ep.start.values
        ends = ep.end.values
        t, d = jitrestrict(time_array, data_array, starts, ends)
        if isinstance(self, TsdFrame):
            return TsdFrame(t=t, d=d, time_support=ep, columns=self.columns)
        elif isinstance(self, Tsd):
            return Tsd(t=t, d=d, time_support=ep)

    def save(self, filename):
        """
        Save TsdFrame object in npz format. The file will contain the timestamps, the
        data and the time support.

        The main purpose of this function is to save small/medium sized time series
        objects. For example, you extracted several channels from your recording and
        filtered them. You can save the filtered channels as a npz to avoid
        reprocessing it.

        You can load the object with numpy.load. Keys are 't', 'd', 'start', 'end', 'type'
        and 'columns' for columns names.

        Parameters
        ----------
        filename : str
            The filename

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tsdframe = nap.TsdFrame(t=np.array([0., 1.]), d = np.array([[2, 3],[4,5]]), columns=['a', 'b'])
        >>> tsdframe.save("my_path/my_tsdframe.npz")

        Here I can retrieve my data with numpy directly:

        >>> file = np.load("my_path/my_tsdframe.npz")
        >>> print(list(file.keys()))
        ['t', 'd', 'start', 'end', 'columns', 'type']
        >>> print(file['t'])
        [0. 1.]

        It is then easy to recreate the Tsd object.
        >>> time_support = nap.IntervalSet(file['start'], file['end'])
        >>> nap.TsdFrame(t=file['t'], d=file['d'], time_support=time_support, columns=file['columns'])
                  a  b
        Time (s)
        0.0       2  3
        1.0       4  5


        Raises
        ------
        RuntimeError
            If filename is not str, path does not exist or filename is a directory.
        """
        if not isinstance(filename, str):
            raise RuntimeError("Invalid type; please provide filename as string")

        if os.path.isdir(filename):
            raise RuntimeError(
                "Invalid filename input. {} is directory.".format(filename)
            )

        if not filename.lower().endswith(".npz"):
            filename = filename + ".npz"

        dirname = os.path.dirname(filename)

        if len(dirname) and not os.path.exists(dirname):
            raise RuntimeError(
                "Path {} does not exist.".format(os.path.dirname(filename))
            )

        saving_kwargs = dict(t=self.index.values,
            start=self.time_support.start.values,
            end=self.time_support.end.values,
            type=np.array(["TsdFrame"], dtype=np.str_))
        
        if isinstance(self, TsdFrame):
            cols_name = self.columns.values
            if cols_name.dtype == np.dtype("O"):
                cols_name = cols_name.astype(str)
            saving_kwargs["columns"]=cols_name
        if not isinstance(self, Ts):
            saving_kwargs["d"]=self.values
        saving_kwargs["type"] = np.array([self.__class__.__name__], dtype=np.str_)
        
        np.savez(
            filename,
        **saving_kwargs
        )


        return
    
    def __str__(self):
        return self.__repr__()
        

    # def find_gaps(self, min_gap, method="absolute"):
    #     """
    #     finds gaps in a tsd larger than min_gap

    #     Parameters
    #     ----------
    #     min_gap : TYPE
    #         Description
    #     method : str, optional
    #         Description
    #     """
    #     print("TODO")
    #     return

    # def find_support(self, min_gap, method="absolute"):
    #     """
    #     find the smallest (to a min_gap resolution) IntervalSet containing all the times in the Tsd

    #     Parameters
    #     ----------
    #     min_gap : TYPE
    #         Description
    #     method : str, optional
    #         Description

    #     Returns
    #     -------
    #     TYPE
    #         Description
    #     """
    #     print("TODO")
    #     return
    

class Tsd(pd.Series, TimeDataMixin):
    """
    A subclass of pandas.Series specialized for neurophysiology time series.

    Tsd provides standardized time representation, plus various functions for manipulating times series.

    Attributes
    ----------
    rate : float
        Frequency of the time series (Hz) computed over the time support
    time_support : IntervalSet
        The time support of the time series
    """

    def __init__(self, d=None, t=None, time_units="s", time_support=None, **kwargs):
        """
        Tsd Initializer.

        Parameters
        ----------
        t : numpy.ndarray or pandas.Series
            An object transformable in a time series, or a pandas.Series equivalent (if d is None)
        d : numpy.ndarray, optional
            The data of the time series
        time_units : str, optional
            The time units in which times are specified ('us', 'ms', 's' [default])
        time_support : IntervalSet, optional
            The time support of the tsd object
        **kwargs
            Arguments that will be passed to the pandas.Series initializer.
        """
 
        if isinstance(d, SingleBlockManager):
            t = d.index.values 
            d = d.array
            
        elif isinstance(d, pd.Series):
            t = d.index.values
            d = d.values
            
        elif "index" in kwargs:  # if we are creating from _constructor
            t = kwargs.pop("index")

        t = np.array(t).astype(np.float64).flatten()
        t = format_timestamps(t, time_units)
        t = sort_timestamps(t)

        if len(t):
            if time_support is not None:
                starts = time_support.start.values
                ends = time_support.end.values
                if d is not None:
                    t, d = jitrestrict(t, d, starts, ends)
                    super().__init__(index=t, data=d)
                else:
                    t = jittsrestrict(t, starts, ends)
                    super().__init__(index=t, data=None, dtype=np.int8)
            else:
                time_support = IntervalSet(start=t[0], end=t[-1])
                if d is not None:
                    super().__init__(index=t, data=d)
                else:
                    super().__init__(index=t, data=d, dtype=np.float64)

            self.time_support = time_support
            duration = np.sum(time_support.end.values - time_support.start.values)
            self.rate = t.shape[0] / duration if duration else 0.0

        else:
            time_support = IntervalSet(pd.DataFrame(columns=["start", "end"]))
            super().__init__(index=t, data=d, dtype=np.float64)

            self.time_support = time_support
            self.rate = 0.0

        self.index.name = "Time (s)"

    def __repr__(self):
        return self.as_series().__repr__()

    def as_series(self):
        """
        Convert the Ts/Tsd object to a pandas.Series object.

        Returns
        -------
        out: pandas.Series
            _
        """
        return pd.Series(self, copy=True)


    def count(self, *args, **kwargs):
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
        >>>    start    end
        >>> 0  100.0  800.0
        """
        bin_size = None
        if "bin_size" in kwargs:
            bin_size = kwargs["bin_size"]
            if isinstance(bin_size, int):
                bin_size = float(bin_size)
            if not isinstance(bin_size, float):
                raise ValueError("bin_size argument should be float.")
        else:
            for a in args:
                if isinstance(a, (float, int)):
                    bin_size = float(a)

        time_units = "s"
        if "time_units" in kwargs:
            time_units = kwargs["time_units"]
            if not isinstance(time_units, str):
                raise ValueError("time_units argument should be 's', 'ms' or 'us'.")
        else:
            for a in args:
                if isinstance(a, str) and a in ["s", "ms", "us"]:
                    time_units = a

        ep = self.time_support
        if "ep" in kwargs:
            ep = kwargs["ep"]
            if not isinstance(ep, IntervalSet):
                raise ValueError("ep argument should be IntervalSet")
        else:
            for a in args:
                if isinstance(a, IntervalSet):
                    ep = a

        time_array = self.index.values
        starts = ep.start.values
        ends = ep.end.values

        if isinstance(bin_size, (float, int)):
            bin_size = float(bin_size)
            bin_size = format_timestamps(np.array([bin_size]), time_units)[0]
            t, d = jitcount(time_array, starts, ends, bin_size)
            time_support = IntervalSet(start=starts, end=ends)
            return Tsd(t=t, d=d, time_support=time_support)
        else:
            _, countin = jittsrestrict_with_count(time_array, starts, ends)
            t = starts + (ends - starts) / 2
            return Tsd(t=t, d=countin, time_support=ep)

    def bin_average(self, bin_size, ep=None, time_units="s"):
        """
        Bin the data by averaging points within bin_size
        bin_size should be seconds unless specified.
        If no epochs is passed, the data will be binned based on the time support.

        Parameters
        ----------
        bin_size : float
            The bin size (default is second)
        ep : None or IntervalSet, optional
            IntervalSet to restrict the operation
        time_units : str, optional
            Time units of bin size ('us', 'ms', 's' [default])

        Returns
        -------
        out: Tsd
            A Tsd object indexed by the center of the bins and holding the averaged data points.

        Examples
        --------
        This example shows how to bin data within bins of 0.1 second.

        >>> import pynapple as nap
        >>> import numpy as np
        >>> tsd = nap.Tsd(t=np.arange(100), d=np.random.rand(100))
        >>> bintsd = tsd.bin_average(0.1)

        An epoch can be specified:

        >>> ep = nap.IntervalSet(start = 10, end = 80, time_units = 's')
        >>> bintsd = tsd.bin_average(0.1, ep=ep)

        And bintsd automatically inherit ep as time support:

        >>> bintsd.time_support
        >>>    start    end
        >>> 0  10.0     80.0
        """
        if not isinstance(ep, IntervalSet):
            ep = self.time_support

        bin_size = format_timestamps(np.array([bin_size]), time_units)[0]

        time_array = self.index.values
        data_array = self.values
        starts = ep.start.values
        ends = ep.end.values
        t, d = jitbin(time_array, data_array, starts, ends, bin_size)
        time_support = IntervalSet(start=starts, end=ends)
        return Tsd(t=t, d=d, time_support=time_support)

    def threshold(self, thr, method="above"):
        """
        Apply a threshold function to the tsd to return a new tsd
        with the time support being the epochs above/below/>=/<= the threshold

        Parameters
        ----------
        thr : float
            The threshold value
        method : str, optional
            The threshold method (above/below/aboveequal/belowequal)

        Returns
        -------
        out: Tsd
            All the time points below/ above/greater than equal to/less than equal to the threshold

        Raises
        ------
        ValueError
            Raise an error if method is not 'below' or 'above'
        RuntimeError
            Raise an error if thr is too high/low and no epochs is found.

        Examples
        --------
        This example finds all epoch above 0.5 within the tsd object.

        >>> import pynapple as nap
        >>> tsd = nap.Tsd(t=np.arange(100), d=np.random.rand(100))
        >>> newtsd = tsd.threshold(0.5)

        The epochs with the times above/below the threshold can be accessed through the time support:

        >>> tsd = nap.Tsd(t=np.arange(100), d=np.arange(100), time_units='s')
        >>> tsd.threshold(50).time_support
        >>>    start   end
        >>> 0   50.5  99.0

        """
        time_array = self.index.values
        data_array = self.values
        starts = self.time_support.start.values
        ends = self.time_support.end.values
        if method not in ["above", "below", "aboveequal", "belowequal"]:
            raise ValueError(
                "Method {} for thresholding is not accepted.".format(method)
            )

        t, d, ns, ne = jitthreshold(time_array, data_array, starts, ends, thr, method)
        time_support = IntervalSet(start=ns, end=ne)
        return Tsd(t=t, d=d, time_support=time_support)

    def to_tsgroup(self):
        """
        Convert Tsd to a TsGroup by grouping timestamps with the same values.
        By default, the values are converted to integers.

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tsd = nap.Tsd(t = np.array([0, 1, 2, 3]), d = np.array([0, 2, 0, 1]))
        Time (s)
        0.0    0
        1.0    2
        2.0    0
        3.0    1
        dtype: int64

        >>> tsd.to_tsgroup()
        Index    rate
        -------  ------
            0    0.67
            1    0.33
            2    0.33

        The reverse operation can be done with the TsGroup.to_tsd function :

        >>> tsgroup.to_tsd()
        Time (s)
        0.0    0.0
        1.0    2.0
        2.0    0.0
        3.0    1.0
        dtype: float64

        Returns
        -------
        TsGroup
            Grouped timestamps

        """
        ts_group = importlib.import_module(".ts_group", "pynapple.core")
        t = self.index.values
        d = self.values.astype("int")
        idx = np.unique(d)

        group = {}
        for k in idx:
            group[k] = Ts(t=t[d == k], time_support=self.time_support)

        return ts_group.TsGroup(group, time_support=self.time_support)

    @property
    def _constructor(self):
        return Tsd
    
    @property
    def _constructor_expanddim(self):
        return TsdFrame


class TsdFrame(pd.DataFrame, TimeDataMixin):
    """
    A subclass of pandas.DataFrame specialized for neurophysiological time series.

    TsdFrame provides standardized time representation, plus various functions for manipulating times series with identical sampling frequency.

    Attributes
    ----------
    rate : float
        Frequency of the time series (Hz) computed over the time support
    time_support : IntervalSet
        The time support of the time series
    """

    def __init__(self, t, d=None, time_units="s", time_support=None, **kwargs):
        """
        TsdFrame initializer

        Parameters
        ----------
        t : numpy.ndarray or pandas.DataFrame
            the time index t,  or a pandas.DataFrame (if d is None)
        d : numpy.ndarray
            The data
        time_units : str, optional
            The time units in which times are specified ('us', 'ms', 's' [default]).
        time_support : IntervalSet, optional
            The time support of the TsdFrame object
        **kwargs
            Arguments that will be passed to the pandas.DataFrame initializer.
        """
        if isinstance(t, BlockManager):
            d = t.as_array()
            c = t.axes[0].values
            t = t.axes[1].values
        elif isinstance(t, pd.DataFrame):
            d = t.values
            c = t.columns.values
            t = t.index.values
        else:
            if "columns" in kwargs:
                c = kwargs["columns"]
            else:
                if isinstance(d, np.ndarray):
                    if len(d.shape) == 2:
                        c = np.arange(d.shape[1])
                    elif len(d.shape) == 1:
                        c = np.zeros(1)
                    else:
                        c = np.array([])
                else:
                    c = None

        t = t.astype(np.float64).flatten()
        t = format_timestamps(t, time_units)
        t = sort_timestamps(t)

        if len(t):
            if time_support is not None:
                starts = time_support.start.values
                ends = time_support.end.values
                if d is not None:
                    t, d = jitrestrict(t, d, starts, ends)
                    super().__init__(index=t, data=d, columns=c)
                else:
                    t = jittsrestrict(t, starts, ends)
                    super().__init__(index=t, data=None, columns=c)
            else:
                time_support = IntervalSet(start=t[0], end=t[-1])
                super().__init__(index=t, data=d, columns=c)

            self.rate = t.shape[0] / np.sum(
                time_support.values[:, 1] - time_support.values[:, 0]
            )

        else:
            time_support = IntervalSet(pd.DataFrame(columns=["start", "end"]))
            super().__init__(index=np.array([]), dtype=np.float64)
            self.rate = 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.time_support = time_support

        self.index.name = "Time (s)"

    def __repr__(self):
        return self.as_units("s").__repr__()

    @property
    def _constructor(self):
        return TsdFrame
    
    @property
    def _constructor_sliced(self):
        return Tsd

    def as_dataframe(self, copy=True):
        """
        Convert the TsdFrame object to a pandas.DataFrame object.

        Returns
        -------
        out: pandas.DataFrame
            _
        """
        return pd.DataFrame(self, copy=copy)


class Ts(Tsd):
    """
    A subclass of the Tsd object for a time series with only time index,
    By default, the values are set to nan.
    All the functions of a Tsd object are available in a Ts object.

    Attributes
    ----------
    rate : float
        Frequency of the time series (Hz) computed over the time support
    time_support : IntervalSet
        The time support of the time series
    """

    def __init__(self, t, time_units="s", time_support=None, **kwargs):
        """
        Ts Initializer

        Parameters
        ----------
        t : numpy.ndarray or pandas.Series
            An object transformable in a time series, or a pandas.Series equivalent (if d is None)
        time_units : str, optional
            The time units in which times are specified ('us', 'ms', 's' [default])
        time_support : IntervalSet, optional
            The time support of the Ts object
        **kwargs
            Arguments that will be passed to the pandas.Series initializer.
        """
        super().__init__(
            t=t,
            d=None,
            time_units=time_units,
            time_support=time_support,
            dtype=np.float64,
            **kwargs,
        )

    def __repr__(self):
        return self.as_series().fillna("").__repr__()
