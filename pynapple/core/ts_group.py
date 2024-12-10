"""

The class `TsGroup` helps group objects with different timestamps 
(i.e. timestamps of spikes of a population of neurons).

"""

import warnings
from collections import UserDict
from collections.abc import Hashable

import numpy
import numpy as np
import pandas as pd
from tabulate import tabulate

from ._core_functions import _count
from ._jitted_functions import jitunion, jitunion_isets
from .base_class import _Base
from .config import nap_config
from .interval_set import IntervalSet
from .metadata_class import _MetadataMixin, add_meta_docstring
from .time_index import TsIndex
from .time_series import Ts, Tsd, TsdFrame, _BaseTsd, is_array_like
from .utils import (
    _convert_iter_to_str,
    _get_terminal_size,
    check_filename,
    convert_to_numpy_array,
)


def _union_intervals(i_sets):
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
            i_sets[0].start,
            i_sets[0].end,
            i_sets[1].start,
            i_sets[1].end,
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


class TsGroup(UserDict, _MetadataMixin):
    """
    Dictionary-like object to group objects with different timestamps (for example timestamps of spikes of a population of neurons).

    Parameters
    ----------
    data : dict or iterable
        Dictionary or iterable of Ts/Tsd objects. The keys should be integer-convertible; if a non-dict iterator is
        passed, its values will be used to create a dict with integer keys.
    time_support : IntervalSet, optional
        The time support of the TsGroup. Ts/Tsd objects will be restricted to the time support if passed.
        If no time support is specified, TsGroup will merge time supports from all the Ts/Tsd objects in data.
    time_units : str, optional
        Time units if data does not contain Ts/Tsd objects ('us', 'ms', 's' [default]).
    bypass_check: bool, optional
        To avoid checking that each element is within time_support.
        Useful to speed up initialization of TsGroup when Ts/Tsd objects have already been restricted beforehand
    metadata: pd.DataFrame or dict, optional
        Metadata associated with each Ts/Tsd object. Metadata names are pulled from DataFrame columns or dictionary keys.
        The length of the metadata should match the number of Ts/Tsd objects.
    **kwargs
        Meta-info about the Ts/Tsd objects. Can be either pandas.Series, numpy.ndarray, list or tuple
        The index should match the index of the input dictionary if pandas Series.
        NOTE: This method of initializing metadata is deprecated and will be removed in a future version of Pynapple.

    Raises
    ------
    RuntimeError
        Raise error if the union of time support of Ts/Tsd object is empty.
    ValueError
        - If a key cannot be converted to integer.
        - If a key was a floating point with non-negligible decimal part.
        - If the converted keys are not unique, i.e. {1: ts_2, "2": ts_2} is valid,
            {1: ts_2, "1": ts_2}  is invalid.

    Examples
    --------
    Initialize a TsGroup as a dictionary of Ts/Tsd objects:

    >>> import pynapple as nap
    >>> import numpy as np
    >>> data = {
    ...    0: nap.Ts(np.arange(100)),
    ...    1: nap.Ts(np.arange(0, 100, 2)),
    ...    2: nap.Ts(np.arange(0, 100, 3)),
    ... }
    >>> tsgroup = nap.TsGroup(data)
    >>> tsgroup
      Index     rate
    -------  -------
          0  1.0101
          1  0.50505
          2  0.34343

    Initialize a TsGroup as a list of Ts/Tsd objects:

    >>> data = [
    ...    nap.Ts(np.arange(100)),
    ...    nap.Ts(np.arange(0, 100, 2)),
    ...    nap.Ts(np.arange(0, 100, 3)),
    ... ]
    >>> tsgroup = nap.TsGroup(data)
    >>> tsgroup
      Index     rate
    -------  -------
          0  1.0101
          1  0.50505
          2  0.34343

    Initialize a TsGroup as a list of array (throws UserWarning):

    >>> data = [
    ...    np.arange(100),
    ...    np.arange(0, 100, 2),
    ...    np.arange(0, 100, 3),
    ... ]
    >>> tsgroup = nap.TsGroup(data)
    >>> tsgroup
      Index     rate
    -------  -------
          0  1.0101
          1  0.50505
          2  0.34343

    Initialize a TsGroup with metadata:

    >>> data = {
    ...    0: nap.Ts(np.arange(100)),
    ...    1: nap.Ts(np.arange(0, 100, 2)),
    ...    2: nap.Ts(np.arange(0, 100, 3)),
    ... }
    >>> metadata = {"label": ["A", "B", "C"]}
    >>> tsgroup = nap.TsGroup(data, metadata=metadata)
    >>> tsgroup
      Index     rate  label
    -------  -------  -------
          0  1.0101   A
          1  0.50505  B
          2  0.34343  C

    Initialize a TsGroup with metadata as a pandas DataFrame:

    >>> data = {
    ...    0: nap.Ts(np.arange(100)),
    ...    1: nap.Ts(np.arange(0, 100, 2)),
    ...    2: nap.Ts(np.arange(0, 100, 3)),
    ... }
    >>> metadata = pd.DataFrame(data=["A", "B", "C"], columns=["label"])
    >>> tsgroup = nap.TsGroup(data, metadata=metadata)
    >>> tsgroup
      Index     rate  label
    -------  -------  -------
          0  1.0101   A
          1  0.50505  B
          2  0.34343  C

    """

    index: np.ndarray
    """The index of the TsGroup, indicating the keys of each member"""

    time_support: IntervalSet
    """The time support of the TsGroup, indicating the time intervals where the TsGroup is defined"""

    nap_class: str
    """The pynapple class name"""

    def __init__(
        self,
        data,
        time_support=None,
        time_units="s",
        bypass_check=False,
        metadata=None,
        **kwargs,
    ):
        # Check input type
        if time_units not in ["s", "ms", "us"]:
            raise ValueError("Argument time_units should be 's', 'ms' or 'us'")
        if not isinstance(bypass_check, bool):
            raise TypeError("Argument bypass_check should be of type bool")
        passed_time_support = False

        if isinstance(time_support, IntervalSet):
            passed_time_support = True
        else:
            if time_support is not None:
                raise TypeError("Argument time_support should be of type IntervalSet")
            else:
                passed_time_support = False

        # set directly in __dict__ to avoid infinite recursion in __setattr__
        self.__dict__["_initialized"] = False

        if not isinstance(data, dict):
            data = dict(enumerate(data))

        # convert all keys to integer
        try:
            keys = [int(k) for k in data.keys()]
        except Exception:
            raise ValueError("All keys must be convertible to integer.")

        # check that there were no floats with decimal points in keys.
        # i.e. 0.5 is not a valid key
        if not all(np.allclose(keys[j], float(k)) for j, k in enumerate(data.keys())):
            raise ValueError("All keys must have integer value!}")

        # check that we have the same num of unique keys
        # {"0":val, 0:val} would be a problem...
        if len(keys) != len(np.unique(keys)):
            raise ValueError("Two dictionary keys contain the same integer value!")

        data = {keys[j]: data[k] for j, k in enumerate(data.keys())}
        self.index = np.sort(keys)

        # Make sure data dict and index are ordered the same
        data = {k: data[k] for k in self.index}

        # initialize metadata
        _MetadataMixin.__init__(self)

        # Transform elements to Ts/Tsd objects
        for k in self.index:
            if not isinstance(data[k], _Base):
                if isinstance(data[k], list) or is_array_like(data[k]):
                    warnings.warn(
                        "Elements should not be passed as {}. Default time units is seconds when creating the Ts object.".format(
                            type(data[k])
                        ),
                        stacklevel=2,
                    )
                    data[k] = Ts(
                        t=convert_to_numpy_array(data[k], "key {}".format(k)),
                        time_support=time_support,
                        time_units=time_units,
                    )

        # If time_support is passed, all elements of data are restricted prior to init
        if passed_time_support:
            self.time_support = time_support
            if not bypass_check:
                data = {k: data[k].restrict(self.time_support) for k in self.index}
        else:
            # Otherwise do the union of all time supports
            time_support = _union_intervals([data[k].time_support for k in self.index])
            if len(time_support) == 0:
                raise RuntimeError(
                    "Union of time supports is empty. Consider passing a time support as argument."
                )
            self.time_support = time_support
            if not bypass_check:
                data = {k: data[k].restrict(self.time_support) for k in self.index}

        UserDict.__init__(self, data)
        self.nap_class = self.__class__.__name__
        # grab current attributes before adding metadata
        self._class_attributes = self.__dir__()
        self._class_attributes.append("_class_attributes")  # add this property

        # Making the TsGroup non mutable
        self._initialized = True

        # Trying to add argument as metainfo
        if len(kwargs):
            warnings.warn(
                "initializing metadata with variable keyword arguments may be unsupported in a future version of Pynapple. Instead, initialize using the metadata argument.",
                FutureWarning,
            )
        self.set_info(metadata, **kwargs)

    """
    Base functions
    """

    def __setattr__(self, name, value):
        # necessary setter to allow metadata to be set as an attribute
        if self._initialized:
            if name in self._class_attributes:
                raise AttributeError(
                    f"Cannot set attribute: '{name}' is a reserved attribute. Use 'set_info()' to set '{name}' as metadata."
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
        if not self._initialized:
            self._metadata.loc[int(key), "rate"] = float(value.rate)
            super().__setitem__(int(key), value)
        else:
            _MetadataMixin.__setitem__(self, key, value)

    def __getitem__(self, key):
        # Standard dict keys are Hashable
        if isinstance(key, Hashable):
            if self.__contains__(key):
                return self.data[key]
            elif key in self._metadata.columns:
                return _MetadataMixin.__getitem__(self, key)
            else:
                raise KeyError(r"Key {} not in group index.".format(key))
        elif isinstance(key, list) and all(isinstance(k, str) for k in key):
            # index multiple metadata columns
            return _MetadataMixin.__getitem__(self, key)

        # array boolean are transformed into indices
        # note that raw boolean are hashable, and won't be
        # tsd == tsg.to_tsd()
        elif np.asarray(key).dtype == bool:
            key = np.asarray(key)
            if key.ndim != 1:
                raise IndexError("Only 1-dimensional boolean indices are allowed!")
            if len(key) != self.__len__():
                raise IndexError(
                    "Boolean index length must be equal to the number of Ts in the group! "
                    f"The number of Ts is {self.__len__()}, but the bolean array"
                    f"has length {len(key)} instead!"
                )
            key = self.index[key]

        keys_not_in = list(filter(lambda x: x not in self.index, key))

        if len(keys_not_in):
            raise KeyError(r"Key {} not in group index.".format(keys_not_in))

        return self._ts_group_from_keys(key)

    def _ts_group_from_keys(self, keys):
        metadata = self._metadata.loc[
            np.sort(keys), self._metadata.columns.drop("rate")
        ]
        return TsGroup(
            {k: self[k] for k in keys},
            time_support=self.time_support,
            metadata=metadata,
        )

    def __repr__(self):
        # Start by determining how many columns and rows.
        # This can be unique for each object
        cols, rows = _get_terminal_size()
        max_cols = np.maximum(cols // 12, 5)
        max_rows = np.maximum(rows - 10, 2)

        # By default, the first three columns should always show.
        col_names = self._metadata.columns.drop("rate")
        headers = ["Index", "rate"] + [c for c in col_names][0:max_cols]
        end = ["..."] if len(headers) > max_cols else []
        headers += end

        if len(self) > max_rows:
            n_rows = max_rows // 2
            ends = np.array([end] * n_rows)
            table = np.vstack(
                (
                    np.hstack(
                        (
                            self.index[0:n_rows, None],
                            np.round(self._metadata[["rate"]].values[0:n_rows], 5),
                            _convert_iter_to_str(
                                self._metadata[col_names].values[0:n_rows, 0:max_cols]
                            ),
                            ends,
                        ),
                        dtype=object,
                    ),
                    np.array(
                        [["..." for _ in range(2 + len(col_names[0:max_cols]))] + end],
                        dtype=object,
                    ),
                    np.hstack(
                        (
                            self.index[-n_rows:, None],
                            np.round(self._metadata[["rate"]].values[-n_rows:], 5),
                            _convert_iter_to_str(
                                self._metadata[col_names].values[-n_rows:, 0:max_cols]
                            ),
                            ends,
                        ),
                        dtype=object,
                    ),
                )
            )
        else:
            ends = np.array([end] * len(self))
            table = np.hstack(
                (
                    self.index[:, None],
                    np.round(self._metadata[["rate"]].values, 5),
                    _convert_iter_to_str(
                        self._metadata[col_names].values[:, 0:max_cols]
                    ),
                    ends,
                ),
                dtype=object,
            )

        return tabulate(table, headers=headers)

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
        """
        Return the rates of each element of the group in Hz
        """
        return self._metadata["rate"]

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

        Examples
        --------
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

        Examples
        --------
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

    def count(self, *args, dtype=None, **kwargs):
        """
        Count occurences of events within bin_size or within a set of bins defined as an IntervalSet.
        You can call this function in multiple ways :

        1. *tsgroup.count(bin_size=1, time_units = 'ms')*
        -> Count occurence of events within a 1 ms bin defined on the time support of the object.

        2. *tsgroup.count(1, ep=my_epochs)*
        -> Count occurent of events within a 1 second bin defined on the IntervalSet my_epochs.

        3. *tsgroup.count(ep=my_bins)*
        -> Count occurent of events within each epoch of the intervalSet object my_bins

        4. *tsgroup.count()*
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
        out: TsdFrame
            A TsdFrame with the columns being the index of each item in the TsGroup.

        Examples
        --------
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

        if dtype:
            try:
                dtype = np.dtype(dtype)
            except Exception:
                raise ValueError(f"{dtype} is not a valid numpy dtype.")

        starts = ep.start
        ends = ep.end

        if isinstance(bin_size, (float, int)):
            bin_size = float(bin_size)
            bin_size = TsIndex.format_timestamps(np.array([bin_size]), time_units)[0]

        # Call it on first element to pre-allocate the array
        if len(self) >= 1:
            time_index, d = _count(
                self.data[self.index[0]].index.values,
                starts,
                ends,
                bin_size,
                dtype=dtype,
            )

            count = np.zeros((len(time_index), len(self.index)), dtype=dtype)
            count[:, 0] = d

            for i in range(1, len(self.index)):
                count[:, i] = _count(
                    self.data[self.index[i]].index.values,
                    starts,
                    ends,
                    bin_size,
                    dtype=dtype,
                )[1]

            return TsdFrame(t=time_index, d=count, time_support=ep, columns=self.index)
        else:
            time_index, _ = _count(np.array([]), starts, ends, bin_size, dtype=dtype)
            return TsdFrame(
                t=time_index, d=np.empty((len(time_index), 0)), time_support=ep
            )

    def to_tsd(self, *args):
        """
        Convert TsGroup to a Tsd. The timestamps of the TsGroup are merged together and sorted.

        Parameters
        ----------
        *args
            string, list, numpy.ndarray or pandas.Series

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tsgroup = nap.TsGroup({0:nap.Ts(t=np.array([0, 1])), 5:nap.Ts(t=np.array([2, 3]))})
        Index    rate
        -------  ------
        0       1
        5       1

        By default, the values of the Tsd is the index of the timestamp in the TsGroup:

        >>> tsgroup.to_tsd()
        Time (s)
        0.0    0.0
        1.0    0.0
        2.0    5.0
        3.0    5.0
        dtype: float64

        Values can be inherited from the metadata of the TsGroup by giving the key of the corresponding columns.

        >>> tsgroup.set_info( phase=np.array([np.pi, 2*np.pi]) ) # assigning a phase to my 2 elements of the TsGroup
        >>> tsgroup.to_tsd("phase")
        Time (s)
        0.0    3.141593
        1.0    3.141593
        2.0    6.283185
        3.0    6.283185
        dtype: float64

        Values can also be passed directly to the function from a list, numpy.ndarray or pandas.Series of values as long as the length matches :

        >>> tsgroup.to_tsd([-1, 1])
        Time (s)
        0.0   -1.0
        1.0   -1.0
        2.0    1.0
        3.0    1.0
        dtype: float64

        The reverse operation can be done with the Tsd.to_tsgroup function :

        >>> my_tsd
        Time (s)
        0.0    0.0
        1.0    0.0
        2.0    5.0
        3.0    5.0
        dtype: float64
        >>> my_tsd.to_tsgroup()
          Index    rate
        -------  ------
              0       1
              5       1

        Returns
        -------
        Tsd

        Raises
        ------
        RuntimeError
            "Metadata indices do not match" : if pandas.Series indexes don't match the TsGroup indexes
            "Values is not the same length" : if numpy.ndarray/list object is not the same size as the TsGroup object
            "Key not in metadata of TsGroup" : if string argument does not match any column names of the metadata,
            "Unknown argument format" ; if argument is not a string, list, numpy.ndarray or pandas.Series

        """
        if len(args):
            if isinstance(args[0], pd.Series):
                if pd.Index.equals(self._metadata.index, args[0].index):
                    _values = args[0].values.flatten()
                else:
                    raise RuntimeError("Index are not equals")
            elif isinstance(args[0], (np.ndarray, list)):
                if len(self._metadata) == len(args[0]):
                    _values = np.array(args[0])
                else:
                    raise RuntimeError("Values is not the same length.")
            elif isinstance(args[0], str):
                if args[0] in self._metadata.columns:
                    _values = self._metadata[args[0]].values
                else:
                    raise RuntimeError(
                        "Key {} not in metadata of TsGroup".format(args[0])
                    )
            else:
                possible_keys = []
                for k, d in self._metadata.dtypes.items():
                    if "int" in str(d) or "float" in str(d):
                        possible_keys.append(k)
                raise RuntimeError(
                    "Unknown argument format. Must be pandas.Series, numpy.ndarray or a string from one of the following values : [{}]".format(
                        ", ".join(possible_keys)
                    )
                )
        else:
            _values = self.index

        nt = 0
        for n in self.index:
            nt += len(self[n])

        times = np.zeros(nt)
        data = np.zeros(nt)
        k = 0
        for n, v in zip(self.index, _values):
            kl = len(self[n])
            times[k : k + kl] = self[n].index
            data[k : k + kl] = v
            k += kl

        idx = np.argsort(times)
        toreturn = Tsd(t=times[idx], d=data[idx], time_support=self.time_support)

        return toreturn

    def get(self, start, end=None, time_units="s"):
        """Slice the `TsGroup` object from `start` to `end` such that all the timestamps within the group satisfy `start<=t<=end`.
        If `end` is None, only the timepoint closest to `start` is returned.

        By default, the time support doesn't change. If you want to change the time support, use the `restrict` function.

        Parameters
        ----------
        start : float or int
            The start (or closest time point if `end` is None)
        end : float or int or None
            The end
        """
        newgr = {}
        for k in self.index:
            newgr[k] = self.data[k].get(start, end, time_units)
        cols = self._metadata.columns.drop("rate")

        return TsGroup(
            newgr,
            time_support=self.time_support,
            bypass_check=True,
            **self._metadata[cols],
        )

    #################################
    # Special slicing of metadata
    #################################

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

        Examples
        --------
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

        Examples
        --------

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
            A dictionary of TsGroup

        Examples
        --------

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

    @staticmethod
    def merge_group(
        *tsgroups, reset_index=False, reset_time_support=False, ignore_metadata=False
    ):
        """
        Merge multiple TsGroup objects into a single TsGroup object.

        Parameters
        ----------
        *tsgroups : TsGroup
            The TsGroup objects to merge
        reset_index : bool, optional
            If True, the keys will be reset to range(len(data))
            If False, the keys of the TsGroup objects should be non-overlapping and will be preserved
        reset_time_support : bool, optional
            If True, the merged TsGroup will merge time supports from all the Ts/Tsd objects in data
            If False, the time support of the TsGroup objects should be the same and will be preserved
        ignore_metadata : bool, optional
            If True, the merged TsGroup will not have any metadata columns other than 'rate'
            If False, all metadata columns should be the same and all metadata will be concatenated

        Returns
        -------
        TsGroup
            A TsGroup of merged objects

        Raises
        ------
        TypeError
            If the input objects are not TsGroup objects
        ValueError
            If `ignore_metadata=False` but metadata columns are not the same
            If `reset_index=False` but keys overlap
            If `reset_time_support=False` but time supports are not the same

        """
        is_tsgroup = [isinstance(tsg, TsGroup) for tsg in tsgroups]
        if not all(is_tsgroup):
            not_tsgroup_index = [i + 1 for i, boo in enumerate(is_tsgroup) if not boo]
            raise TypeError(f"Input at positions {not_tsgroup_index} are not TsGroup!")

        if len(tsgroups) == 1:
            print("Only one TsGroup object provided, no merge needed.")
            return tsgroups[0]

        tsg1 = tsgroups[0]
        items = tsg1.items()
        keys = set(tsg1.keys())
        metadata = tsg1._metadata

        for i, tsg in enumerate(tsgroups[1:]):
            if not ignore_metadata:
                if tsg1.metadata_columns != tsg.metadata_columns:
                    raise ValueError(
                        f"TsGroup at position {i+2} has different metadata columns from previous TsGroup objects. "
                        "Set `ignore_metadata=True` to bypass the check."
                    )
                metadata = pd.concat([metadata, tsg._metadata], axis=0)

            if not reset_index:
                key_overlap = keys.intersection(tsg.keys())
                if key_overlap:
                    raise ValueError(
                        f"TsGroup at position {i+2} has overlapping keys {key_overlap} with previous TsGroup objects. "
                        "Set `reset_index=True` to bypass the check."
                    )
                keys.update(tsg.keys())

            if reset_time_support:
                time_support = None
            else:
                if not np.allclose(
                    tsg1.time_support.as_units("s").to_numpy(),
                    tsg.time_support.as_units("s").to_numpy(),
                    atol=10 ** (-nap_config.time_index_precision),
                    rtol=0,
                ):
                    raise ValueError(
                        f"TsGroup at position {i+2} has different time support from previous TsGroup objects. "
                        "Set `reset_time_support=True` to bypass the check."
                    )
                time_support = tsg1.time_support

            items.extend(tsg.items())

        if reset_index:
            metadata.index = range(len(metadata))
            data = {i: ts[1] for i, ts in enumerate(items)}
        else:
            data = dict(items)

        if ignore_metadata:
            return TsGroup(data, time_support=time_support, bypass_check=False)
        else:
            cols = metadata.columns.drop("rate")
            return TsGroup(
                data,
                time_support=time_support,
                bypass_check=False,
                metadata=metadata[cols],
            )

    def merge(
        self,
        *tsgroups,
        reset_index=False,
        reset_time_support=False,
        ignore_metadata=False,
    ):
        """
        Merge the TsGroup object with other TsGroup objects.
        Common uses include adding more neurons/channels (supposing each Ts/Tsd corresponds to data from a neuron/channel) or adding more trials (supposing each Ts/Tsd corresponds to data from a trial).

        Parameters
        ----------
        *tsgroups : TsGroup
            The TsGroup objects to merge with
        reset_index : bool, optional
            If True, the keys will be reset to range(len(data))
            If False, the keys of the TsGroup objects should be non-overlapping and will be preserved
        reset_time_support : bool, optional
            If True, the merged TsGroup will merge time supports from all the Ts/Tsd objects in data
            If False, the time support of the TsGroup objects should be the same and will be preserved
        ignore_metadata : bool, optional
            If True, the merged TsGroup will not have any metadata columns other than 'rate'
            If False, all metadata columns should be the same and all metadata will be concatenated

        Returns
        -------
        TsGroup
            A TsGroup of merged objects

        Raises
        ------
        TypeError
            If the input objects are not TsGroup objects
        ValueError
            If `ignore_metadata=False` but metadata columns are not the same
            If `reset_index=False` but keys overlap
            If `reset_time_support=False` but time supports are not the same

        Examples
        --------

        >>> import pynapple as nap
        >>> time_support_a = nap.IntervalSet(start=-1, end=1, time_units='s')
        >>> time_support_b = nap.IntervalSet(start=-5, end=5, time_units='s')

        >>> dict1 = {0: nap.Ts(t=[-1, 0, 1], time_units='s')}
        >>> tsgroup1 = nap.TsGroup(dict1, time_support=time_support_a)

        >>> dict2 = {10: nap.Ts(t=[-1, 0, 1], time_units='s')}
        >>> tsgroup2 = nap.TsGroup(dict2, time_support=time_support_a)

        >>> dict3 = {0: nap.Ts(t=[-.1, 0, .1], time_units='s')}
        >>> tsgroup3 = nap.TsGroup(dict3, time_support=time_support_a)

        >>> dict4 = {10: nap.Ts(t=[-1, 0, 1], time_units='s')}
        >>> tsgroup4 = nap.TsGroup(dict2, time_support=time_support_b)

        Merge with default options if have the same time support and non-overlapping indexes:

        >>> tsgroup_12 = tsgroup1.merge(tsgroup2)
        >>> tsgroup_12
        Index    rate
        -------  ------
             0     1.5
            10     1.5

        Set `reset_index=True` if indexes are overlapping:

        >>> tsgroup_13 = tsgroup1.merge(tsgroup3, reset_index=True)
        >>> tsgroup_13
        Index    rate
        -------  ------
              0     1.5
              1     1.5

        Set `reset_time_support=True` if time supports are different:

        >>> tsgroup_14 = tsgroup1.merge(tsgroup4, reset_time_support=True)
        >>> tsgroup_14
        >>> tsgroup_14.time_support
        Index    rate
        -------  ------
              0     0.3
             10     0.3

                    start    end
            0       -5      5
            shape: (1, 2), time unit: sec.

        See Also
        --------
        [`TsGroup.merge_group`](./#pynapple.core.ts_group.TsGroup.merge_group)
        """
        return TsGroup.merge_group(
            self,
            *tsgroups,
            reset_index=reset_index,
            reset_time_support=reset_time_support,
            ignore_metadata=ignore_metadata,
        )

    def save(self, filename):
        """
        Save TsGroup object in npz format. The file will contain the timestamps,
        the data (if group of Tsd), group index, the time support and the metadata

        The main purpose of this function is to save small/medium sized TsGroup
        objects.

        The function will "flatten" the TsGroup by sorting all the timestamps
        and assigning to each the corresponding index. Typically, a TsGroup like
        this :

        >>> TsGroup({
            0 : Tsd(t=[0, 2, 4], d=[1, 2, 3])
            1 : Tsd(t=[1, 5], d=[5, 6])})

        will be saved as npz with the following keys:


        >>> {
            't' : [0, 1, 2, 4, 5],
            'd' : [1, 5, 2, 3, 5],
            'index' : [0, 1, 0, 0, 1],
            'start' : [0],
            'end' : [5],
            'keys' : [0, 1],
            'type' : 'TsGroup'
        }

        Metadata are saved by columns with the column name as the npz key. To avoid
        potential conflicts, make sure the columns name of the metadata are different
        from ['t', 'd', 'start', 'end', 'index', 'keys']

        You can load the object with `nap.load_file`. Default keys are 't', 'd'(optional),
        'start', 'end', 'index', 'keys' and 'type'.
        See the example below.

        Parameters
        ----------
        filename : str
            The filename

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tsgroup = nap.TsGroup({
            0 : nap.Ts(t=np.array([0.0, 2.0, 4.0])),
            6 : nap.Ts(t=np.array([1.0, 5.0]))
            },
            group = np.array([0, 1]),
            location = np.array(['right foot', 'left foot'])
            )
        >>> tsgroup
          Index    rate    group  location
        -------  ------  -------  ----------
              0     0.6        0  right foot
              6     0.4        1  left foot
        >>> tsgroup.save("my_tsgroup.npz")

        To get back to pynapple, you can use the `nap.load_file` function :

        >>> tsgroup = nap.load_file("my_tsgroup.npz")
        >>> tsgroup
          Index    rate    group  location
        -------  ------  -------  ----------
              0     0.6        0  right foot
              6     0.4        1  left foot

        Raises
        ------
        RuntimeError
            If filename is not str, path does not exist or filename is a directory.
        """
        filename = check_filename(filename)

        dicttosave = {"type": np.array(["TsGroup"], dtype=np.str_)}
        # don't save rate in metadata since it will be re-added when loading
        dicttosave["_metadata"] = self._metadata.drop(columns="rate").to_dict()

        # are these things that still need to be enforced?
        # for k in self._metadata.columns:
        #     if k not in ["t", "d", "start", "end", "index", "keys"]:
        #         tmp = self._metadata[k].values
        #         if tmp.dtype == np.dtype("O"):
        #             tmp = tmp.astype(np.str_)
        #         dicttosave[k] = tmp

        # We can't use to_tsd here in case tsgroup contains Tsd and not only Ts.
        nt = 0
        for n in self.index:
            nt += len(self[n])

        times = np.zeros(nt)
        data = np.full(nt, np.nan)
        index = np.zeros(nt, dtype=np.int64)
        k = 0
        for n in self.index:
            kl = len(self[n])
            times[k : k + kl] = self[n].index
            if isinstance(self[n], _BaseTsd):
                data[k : k + kl] = self[n].values
            index[k : k + kl] = int(n)
            k += kl

        idx = np.argsort(times)
        times = times[idx]
        index = index[idx]

        dicttosave["t"] = times
        dicttosave["index"] = index
        if not np.all(np.isnan(data)):
            dicttosave["d"] = data[idx]
        dicttosave["keys"] = np.array(self.keys())
        dicttosave["start"] = self.time_support.start
        dicttosave["end"] = self.time_support.end

        np.savez(filename, **dicttosave)

        return

    @classmethod
    def _from_npz_reader(cls, file):
        """
        Load a Tsd object from a npz file.

        Parameters
        ----------
        file : str
            The opened npz file

        Returns
        -------
        Tsd
            The Tsd object
        """

        times = file["t"]
        index = file["index"]
        has_data = "d" in file.keys()
        time_support = IntervalSet(file["start"], file["end"])

        if has_data:
            data = file["data"]

        if "keys" in file.keys():
            keys = file["keys"]
        else:
            keys = np.unique(index)

        group = {}
        for key in keys:
            filtering_index = index == key
            t = times[filtering_index]

            if has_data:
                group[key] = Tsd(
                    t=t,
                    d=data[filtering_index],
                    time_support=time_support,
                )
            else:
                group[key] = Ts(t=t, time_support=time_support)

        tsgroup = cls(group, time_support=time_support, bypass_check=True)

        # do we need to enforce that these keys are not in metadata?
        # not_info_keys = {"start", "end", "t", "index", "d", "rate", "keys"}

        if "_metadata" in file:  # load metadata if it exists
            if file["_metadata"]:  # check that metadata is not empty
                metainfo = pd.DataFrame.from_dict(file["_metadata"].item())
                tsgroup.set_info(metainfo)
        return tsgroup

    @add_meta_docstring("set_info")
    def set_info(self, metadata=None, **kwargs):
        """
        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = {0:nap.Ts(t=np.arange(0,200), time_units='s'),
        ... 1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        ... 2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        ... }
        >>> tsgroup = nap.TsGroup(tmp)

        To add metadata with a pandas.DataFrame:

        >>> import pandas as pd
        >>> structs = pd.DataFrame(index = [0,1,2], data=['pfc','pfc','ca1'], columns=['struct'])
        >>> tsgroup.set_info(structs)
        >>> tsgroup
          Index     rate  struct
        -------  -------  --------
              0  0.66722  pfc
              1  1.33445  pfc
              2  4.00334  ca1

        To add metadata with a dictionary:

        >>> coords = {"coords": [[0,0],[0,1],[1,0]]}
        >>> tsgroup.set_info(coords)
        >>> tsgroup
          Index     rate  struct    coords
        -------  -------  --------  --------
              0  0.66722  pfc       [0, 0]
              1  1.33445  pfc       [0, 1]
              2  4.00334  ca1       [1, 0]

        To add metadata with a keyword argument (pd.Series, numpy.ndarray, list or tuple):

        >>> hd = pd.Series(index = [0,1,2], data = [0,1,1])
        >>> tsgroup.set_info(hd=hd)
        >>> tsgroup
          Index     rate  struct    coords      hd
        -------  -------  --------  --------  ----
              0  0.66722  pfc       [0, 0]       0
              1  1.33445  pfc       [0, 1]       1
              2  4.00334  ca1       [1, 0]       1

        To add metadata as an attribute:

        >>> tsgroup.label = ["a", "b", "c"]
        >>> tsgroup
          Index     rate  struct    coords      hd  label
        -------  -------  --------  --------  ----  -------
              0  0.66722  pfc       [0, 0]       0  a
              1  1.33445  pfc       [0, 1]       1  b
              2  4.00334  ca1       [1, 0]       1  c

        To add metadata as a key:

        >>> tsgroup["type"] = ["multi", "multi", "single"]
        >>> tsgroup
          Index     rate  struct    coords      hd  label    type
        -------  -------  --------  --------  ----  -------  ------
              0  0.66722  pfc       [0, 0]       0  a        multi
              1  1.33445  pfc       [0, 1]       1  b        multi
              2  4.00334  ca1       [1, 0]       1  c        single

        Metadata can be overwritten:

        >>> tsgroup.set_info(label=["x", "y", "z"])
        >>> tsgroup
          Index     rate  struct    coords      hd  label    type
        -------  -------  --------  --------  ----  -------  ------
              0  0.66722  pfc       [0, 0]       0  x        multi
              1  1.33445  pfc       [0, 1]       1  y        multi
              2  4.00334  ca1       [1, 0]       1  z        single

        """
        _MetadataMixin.set_info(self, metadata, **kwargs)

    @add_meta_docstring("get_info")
    def get_info(self, key):
        """
        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = {0:nap.Ts(t=np.arange(0,200), time_units='s'),
        ... 1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        ... 2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        ... }
        >>> metadata = {"l1": [1, 2, 3], "l2": ["x", "x", "y"]}
        >>> tsgroup = nap.TsGroup(tmp,metadata=metadata)

        To access a single metadata column:

        >>> tsgroup.get_info("l1")
        0    1
        1    2
        2    3
        Name: l1, dtype: int64

        To access multiple metadata columns:

        >>> tsgroup.get_info(["l1", "l2"])
           l1 l2
        0   1  x
        1   2  x
        2   3  y

        To access metadata of a single index:

        >>> tsgroup.get_info(0)
        rate    0.667223
        l1             1
        l2             x
        Name: 0, dtype: object

        To access metadata of multiple indices:

        >>> tsgroup.get_info([0, 1])
               rate  l1 l2
        0  0.667223   1  x
        1  1.334445   2  x

        To access metadata of a single index and column:

        >>> tsgroup.get_info((0, "l1"))
        np.int64(1)

        To access metadata as an attribute:

        >>> tsgroup.l1
        0    1
        1    2
        2    3
        Name: l1, dtype: int64

        To access metadata as a key:

        >>> tsgroup["l1"]
        0    1
        1    2
        2    3
        Name: l1, dtype: int64

        Multiple metadata columns can be accessed as keys:

        >>> tsgroup[["l1", "l2"]]
           l1 l2
        0   1  x
        1   2  x
        2   3  y
        """
        return _MetadataMixin.get_info(self, key)
