# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-28 15:10:48
# @Last Modified by:   gviejo
# @Last Modified time: 2022-12-06 20:06:39


import warnings
from collections import UserDict

import numpy as np
import pandas as pd
from tabulate import tabulate

from .interval_set import IntervalSet
from .jitted_functions import jitcount, jittsrestrict, jitunion, jitunion_isets
from .time_series import Ts, TsdFrame
from .time_units import format_timestamps


def union_intervals(i_sets):
    """
    Helper to merge intervals from ts_group
    """
    n = len(i_sets)

    if n == 1:
        return i_sets[0]

    new_start = np.zeros(0)
    new_end = np.zeros(0)

    if n == 2:
        new_start, new_end = jitunion(
            i_sets[0].start.values,
            i_sets[0].end.values,
            i_sets[1].start.values,
            i_sets[1].end.values,
        )

    if n > 2:
        sizes = np.array([i_sets[i].shape[0] for i in range(n)])
        startends = np.zeros((np.sum(sizes), 2))
        ct = 0
        for i in range(sizes.shape[0]):
            startends[ct : ct + sizes[i], :] = i_sets[i].values
            ct += sizes[i]
        new_start, new_end = jitunion_isets(startends[:, 0], startends[:, 1])

    return IntervalSet(new_start, new_end)


class TsGroup(UserDict):
    """
    The TsGroup is a dictionnary-like object to hold multiple [`Ts`][pynapple.core.time_series.Ts] or [`Tsd`][pynapple.core.time_series.Tsd] objects with different time index.

    Attributes
    ----------
    time_support: IntervalSet
        The time support of the TsGroup
    rates : pandas.Series
        The rate of each element of the TsGroup
    """

    def __init__(
        self, data, time_support=None, time_units="s", bypass_check=False, **kwargs
    ):
        """
        TsGroup Initializer

        Parameters
        ----------
        data : dict
            Dictionnary containing Ts/Tsd objects
        time_support : IntervalSet, optional
            The time support of the TsGroup. Ts/Tsd objects will be restricted to the time support if passed.
            If no time support is specified, TsGroup will merge time supports from all the Ts/Tsd objects in data.
        time_units : str, optional
            Time units if data does not contain Ts/Tsd objects ('us', 'ms', 's' [default]).
        bypass_check: bool, optional
            To avoid checking that each element is within time_support.
            Useful to speed up initialization of TsGroup when Ts/Tsd objects have already been restricted beforehand
        **kwargs
            Meta-info about the Ts/Tsd objects. Can be either pandas.Series or numpy.ndarray.
            Note that the index should match the index of the input dictionnary.

        Raises
        ------
        RuntimeError
            Raise error if the union of time support of Ts/Tsd object is empty.
        """
        self._initialized = False

        self.index = np.sort(list(data.keys()))

        self._metadata = pd.DataFrame(index=self.index, columns=["rate"])

        # Transform elements to Ts/Tsd objects
        for k in self.index:
            if isinstance(data[k], (np.ndarray, list)):
                warnings.warn(
                    "Elements should not be passed as numpy array. Default time units is seconds when creating the Ts object.",
                    stacklevel=2,
                )
                data[k] = Ts(
                    t=data[k], time_support=time_support, time_units=time_units
                )

        # If time_support is passed, all elements of data are restricted prior to init
        if isinstance(time_support, IntervalSet):
            self.time_support = time_support
            if not bypass_check:
                data = {k: data[k].restrict(self.time_support) for k in self.index}
        else:
            # Otherwise do the union of all time supports
            time_support = union_intervals([data[k].time_support for k in self.index])
            if len(time_support) == 0:
                raise RuntimeError(
                    "Union of time supports is empty. Consider passing a time support as argument."
                )
            self.time_support = time_support
            if not bypass_check:
                data = {k: data[k].restrict(self.time_support) for k in self.index}

        UserDict.__init__(self, data)

        # Making the TsGroup non mutable
        self._initialized = True

        # Trying to add argument as metainfo
        self.set_info(**kwargs)

    """
    Base functions
    """

    def __setitem__(self, key, value):
        if self._initialized:
            raise RuntimeError("TsGroup object is not mutable.")

        self._metadata.loc[int(key), "rate"] = value.rate
        super().__setitem__(int(key), value)
        # if self.__contains__(key):
        #     raise KeyError("Key {} already in group index.".format(key))
        # else:
        # if isinstance(value, (Ts, Tsd)):
        #     self._metadata.loc[int(key), "rate"] = value.rate
        #     super().__setitem__(int(key), value)
        # elif isinstance(value, (np.ndarray, list)):
        #     warnings.warn(
        #         "Elements should not be passed as numpy array. Default time units is seconds when creating the Ts object.",
        #         stacklevel=2,
        #     )
        #     tmp = Ts(t=value, time_units="s")
        #     self._metadata.loc[int(key), "rate"] = tmp.rate
        #     super().__setitem__(int(key), tmp)
        # else:
        #     raise ValueError("Value with key {} is not an iterable.".format(key))

    def __getitem__(self, key):
        if key.__hash__:
            if self.__contains__(key):
                return self.data[key]
            else:
                raise KeyError("Can't find key {} in group index.".format(key))
        else:
            metadata = self._metadata.loc[key, self._metadata.columns.drop("rate")]
            return TsGroup(
                {k: self[k] for k in key}, time_support=self.time_support, **metadata
            )

    def __repr__(self):
        cols = self._metadata.columns.drop("rate")
        headers = ["Index", "rate"] + [c for c in cols]
        lines = []

        for i in self.data.keys():
            lines.append(
                [str(i), "%.2f" % self._metadata.loc[i, "rate"]]
                + [self._metadata.loc[i, c] for c in cols]
            )
        return tabulate(lines, headers=headers)

    def __str__(self):
        return self.__repr__()

    def keys(self):
        """
        Return index/keys of TsGroup

        Returns
        -------
        list
            List of keys
        """
        return list(self.data.keys())

    def items(self):
        """
        Return a list of key/object.

        Returns
        -------
        list
            List of tuples
        """
        return list(self.data.items())

    def values(self):
        """
        Return a list of all the Ts/Tsd objects in the TsGroup

        Returns
        -------
        list
            List of Ts/Tsd objects
        """
        return list(self.data.values())

    @property
    def rates(self):
        return self._metadata["rate"]

    #######################
    # Metadata
    #######################
    def set_info(self, *args, **kwargs):
        """
        Add metadata informations about the TsGroup.
        Metadata are saved as a DataFrame.

        Parameters
        ----------
        *args
            pandas.Dataframe or list of pandas.DataFrame
        **kwargs
            Can be either pandas.Series or numpy.ndarray
        Raises
        ------
        RuntimeError
            Raise an error if
                no column labels are found when passing simple arguments,
                indexes are not equals for a pandas series,
                not the same length when passing numpy array.

        Example
        -------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = { 0:nap.Ts(t=np.arange(0,200), time_units='s'),
        1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        }
        >>> tsgroup = nap.TsGroup(tmp)

        To add metadata with a pandas.DataFrame:

        >>> import pandas as pd
        >>> structs = pd.DataFrame(index = [0,1,2], data=['pfc','pfc','ca1'], columns=['struct'])
        >>> tsgroup.set_info(structs)
        >>> tsgroup
          Index    Freq. (Hz)  struct
        -------  ------------  --------
              0             1  pfc
              1             2  pfc
              2             4  ca1

        To add metadata with a pd.Series or numpy.ndarray:

        >>> hd = pd.Series(index = [0,1,2], data = [0,1,1])
        >>> tsgroup.set_info(hd=hd)
        >>> tsgroup
          Index    Freq. (Hz)  struct      hd
        -------  ------------  --------  ----
              0             1  pfc          0
              1             2  pfc          1
              2             4  ca1          1

        """
        if len(args):
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    if pd.Index.equals(self._metadata.index, arg.index):
                        self._metadata = self._metadata.join(arg)
                    else:
                        raise RuntimeError("Index are not equals")
                elif isinstance(arg, (pd.Series, np.ndarray)):
                    raise RuntimeError("Columns needs to be labelled for metadata")
        if len(kwargs):
            for k, v in kwargs.items():
                if isinstance(v, pd.Series):
                    if pd.Index.equals(self._metadata.index, v.index):
                        self._metadata[k] = v
                    else:
                        raise RuntimeError("Index are not equals")
                elif isinstance(v, np.ndarray):
                    if len(self._metadata) == len(v):
                        self._metadata[k] = v
                    else:
                        raise RuntimeError("Array is not the same length.")
        return

    def get_info(self, key):
        """
        Returns the metainfo located in one column.
        The key for the column frequency is "rate".

        Parameters
        ----------
        key : str
            One of the metainfo columns name

        Returns
        -------
        pandas.Series
            The metainfo
        """
        if key in ["freq", "frequency"]:
            key = "rate"
        return self._metadata[key]

    #################################
    # Generic functions of Tsd objects
    #################################
    def restrict(self, ep):
        """
        Restricts a TsGroup object to a set of time intervals delimited by an IntervalSet object

        Parameters
        ----------
        ep : IntervalSet
            the IntervalSet object

        Returns
        -------
        TsGroup
            TsGroup object restricted to ep

        Example
        -------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = { 0:nap.Ts(t=np.arange(0,200), time_units='s'),
        1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        }
        >>> tsgroup = nap.TsGroup(tmp)
        >>> ep = nap.IntervalSet(start=0, end=100, time_units='s')
        >>> newtsgroup = tsgroup.restrict(ep)

        All objects within the TsGroup automatically inherit the epochs defined by ep.

        >>> newtsgroup.time_support
           start    end
        0    0.0  100.0
        >>> newtsgroup[0].time_support
           start    end
        0    0.0  100.0
        """
        newgr = {}
        for k in self.index:
            newgr[k] = self.data[k].restrict(ep)
        cols = self._metadata.columns.drop("rate")

        return TsGroup(
            newgr, time_support=ep, bypass_check=True, **self._metadata[cols]
        )

    def value_from(self, tsd, ep=None):
        """
        Replace the value of each Ts/Tsd object within the Ts group with the closest value from tsd argument

        Parameters
        ----------
        tsd : Tsd
            The Tsd object holding the values to replace
        ep : IntervalSet
            The IntervalSet object to restrict the operation.
            If None, the time support of the tsd input object is used.

        Returns
        -------
        TsGroup
            TsGroup object with the new values

        Example
        -------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = { 0:nap.Ts(t=np.arange(0,200), time_units='s'),
        1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        }
        >>> tsgroup = nap.TsGroup(tmp)
        >>> ep = nap.IntervalSet(start=0, end=100, time_units='s')

        The variable tsd is a time series object containing the values to assign, for example the tracking data:

        >>> tsd = nap.Tsd(t=np.arange(0,100), d=np.random.rand(100), time_units='s')
        >>> ep = nap.IntervalSet(start = 0, end = 100, time_units = 's')
        >>> newtsgroup = tsgroup.value_from(tsd, ep)

        """
        if ep is None:
            ep = tsd.time_support

        newgr = {}
        for k in self.data:
            newgr[k] = self.data[k].value_from(tsd, ep)

        cols = self._metadata.columns.drop("rate")
        return TsGroup(newgr, time_support=ep, **self._metadata[cols])

    def count(self, bin_size, ep=None, time_units="s"):
        """
        Count occurences of events within bin_size.
        bin_size should be seconds unless specified.
        If no epochs is passed, the data will be binned based on the time support.

        Parameters
        ----------
        bin_size : float
            The bin size (default is second)
        ep : IntervalSet, optional
            IntervalSet to restrict the operation
            If None, the time support of self is used.
        time_units : str, optional
            Time units of bin size ('us', 'ms', 's' [default])

        Returns
        -------
        TsdFrame
            A TsdFrame with the columns being the index of each item in the TsGroup.

        Example
        -------
        This example shows how to count events within bins of 0.1 second for the first 100 seconds.

        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = { 0:nap.Ts(t=np.arange(0,200), time_units='s'),
        1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        }
        >>> tsgroup = nap.TsGroup(tmp)
        >>> ep = nap.IntervalSet(start=0, end=100, time_units='s')
        >>> bincount = tsgroup.count(0.1, ep)
        >>> bincount
                  0  1  2
        Time (s)
        0.05      0  0  0
        0.15      0  0  0
        0.25      0  0  1
        0.35      0  0  0
        0.45      0  0  0
        ...      .. .. ..
        99.55     0  1  1
        99.65     0  0  0
        99.75     0  0  1
        99.85     0  0  0
        99.95     1  1  1
        [1000 rows x 3 columns]

        """
        if not isinstance(ep, IntervalSet):
            ep = self.time_support

        bin_size = format_timestamps(np.array([bin_size]), time_units)[0]

        time_index = []
        for i in ep.index:
            bins = np.arange(ep.start[i], ep.end[i] + bin_size, bin_size)
            t = bins[0:-1] + np.diff(bins) / 2
            t = jittsrestrict(
                t, np.array([ep.loc[i, "start"]]), np.array([ep.loc[i, "end"]])
            )
            time_index.append(t)

        time_index = np.hstack(time_index)

        n = len(self.index)

        count = np.zeros((time_index.shape[0], n), dtype=np.int64)

        starts = ep.start.values
        ends = ep.end.values

        for i in range(n):
            count[:, i] = jitcount(
                self.data[self.index[i]].index.values, starts, ends, bin_size
            )[1]

        toreturn = TsdFrame(t=time_index, d=count, time_support=ep, columns=self.index)
        return toreturn

    """
    Special slicing of metadata
    """

    def getby_threshold(self, key, thr, op=">"):
        """
        Return a TsGroup with all Ts/Tsd objects with values above threshold for metainfo under key.

        Parameters
        ----------
        key : str
            One of the metainfo columns name
        thr : float
            THe value for thresholding
        op : str, optional
            The type of operation. Possibilities are '>', '<', '>=' or '<='.

        Returns
        -------
        TsGroup
            The new TsGroup

        Raises
        ------
        RuntimeError
            Raise eror is operation is not recognized.

        Example
        -------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = { 0:nap.Ts(t=np.arange(0,200), time_units='s'),
        1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        }
        >>> tsgroup = nap.TsGroup(tmp)
          Index    Freq. (Hz)
        -------  ------------
              0             1
              1             2
              2             4

        This exemple shows how to get a new TsGroup with all elements for which the metainfo frequency is above 1.
        >>> newtsgroup = tsgroup.getby_threshold('freq', 1, op = '>')
          Index    Freq. (Hz)
        -------  ------------
              1             2
              2             4

        """
        if op == ">":
            ix = list(self._metadata.index[self._metadata[key] > thr])
            return self[ix]
        elif op == "<":
            ix = list(self._metadata.index[self._metadata[key] < thr])
            return self[ix]
        elif op == ">=":
            ix = list(self._metadata.index[self._metadata[key] >= thr])
            return self[ix]
        elif op == "<=":
            ix = list(self._metadata.index[self._metadata[key] <= thr])
            return self[ix]
        else:
            raise RuntimeError("Operation {} not recognized.".format(op))

    def getby_intervals(self, key, bins):
        """
        Return a list of TsGroup binned.

        Parameters
        ----------
        key : str
            One of the metainfo columns name
        bins : numpy.ndarray or list
            The bin intervals

        Returns
        -------
        list
            A list of TsGroup

        Example
        -------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = { 0:nap.Ts(t=np.arange(0,200), time_units='s'),
        1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        }
        >>> tsgroup = nap.TsGroup(tmp, alpha = np.arange(3))
          Index    Freq. (Hz)    alpha
        -------  ------------  -------
              0             1        0
              1             2        1
              2             4        2

        This exemple shows how to bin the TsGroup according to one metainfo key.
        >>> newtsgroup, bincenter = tsgroup.getby_intervals('alpha', [0, 1, 2])
        >>> newtsgroup
        [  Index    Freq. (Hz)    alpha
         -------  ------------  -------
               0             1        0,
           Index    Freq. (Hz)    alpha
         -------  ------------  -------
               1             2        1]

        By default, the function returns the center of the bins.
        >>> bincenter
        array([0.5, 1.5])
        """
        idx = np.digitize(self._metadata[key], bins) - 1
        groups = self._metadata.index.groupby(idx)
        ix = np.unique(list(groups.keys()))
        ix = ix[ix >= 0]
        ix = ix[ix < len(bins) - 1]
        xb = bins[0:-1] + np.diff(bins) / 2
        sliced = [self[list(groups[i])] for i in ix]
        return sliced, xb[ix]

    def getby_category(self, key):
        """
        Return a list of TsGroup grouped by category.

        Parameters
        ----------
        key : str
            One of the metainfo columns name

        Returns
        -------
        dict
            A dictionnary of TsGroup

        Example
        -------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = { 0:nap.Ts(t=np.arange(0,200), time_units='s'),
        1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        }
        >>> tsgroup = nap.TsGroup(tmp, group = [0,1,1])
          Index    Freq. (Hz)    group
        -------  ------------  -------
              0             1        0
              1             2        1
              2             4        1

        This exemple shows how to group the TsGroup according to one metainfo key.
        >>> newtsgroup = tsgroup.getby_category('group')
        >>> newtsgroup
        {0:   Index    Freq. (Hz)    group
         -------  ------------  -------
               0             1        0,
         1:   Index    Freq. (Hz)    group
         -------  ------------  -------
               1             2        1
               2             4        1}
        """
        groups = self._metadata.groupby(key).groups
        sliced = {k: self[list(groups[k])] for k in groups.keys()}
        return sliced
