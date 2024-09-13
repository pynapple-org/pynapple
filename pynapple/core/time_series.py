"""
    
    Pynapple time series are containers specialized for neurophysiological time series.

    They provides standardized time representation, plus various functions for manipulating times series with identical sampling frequency.

    Multiple time series object are avaible depending on the shape of the data.

    - `TsdTensor` : for data with of more than 2 dimensions, typically movies.
    - `TsdFrame` : for column-based data. It can be easily converted to a pandas.DataFrame. Columns can be labelled and selected similar to pandas.
    - `Tsd` : One-dimensional time series. It can be converted to a pandas.Series.
    - `Ts` : For timestamps data only.

    Most of the same functions are available through all classes. Objects behaves like numpy.ndarray. Slicing can be done the same way for example 
    `tsd[0:10]` returns the first 10 rows. Similarly, you can call any numpy functions like `np.mean(tsd, 1)`.
"""

import abc
import importlib
import warnings
from numbers import Number

import numpy as np
import pandas as pd
from numpy.lib.mixins import NDArrayOperatorsMixin
from scipy import signal
from tabulate import tabulate

from ._core_functions import _bin_average, _convolve, _dropna, _restrict, _threshold
from .base_class import Base
from .interval_set import IntervalSet
from .time_index import TsIndex
from .utils import (
    _concatenate_tsd,
    _get_terminal_size,
    _split_tsd,
    _TsdFrameSliceHelper,
    convert_to_array,
    is_array_like,
)


def _get_class(data):
    """Select the right time series object and return the class

    Parameters
    ----------
    data : numpy.ndarray
        The data to hold in the time series object

    Returns
    -------
    Class
        The class
    """
    if data.ndim == 1:
        return Tsd
    elif data.ndim == 2:
        return TsdFrame
    else:
        return TsdTensor


class BaseTsd(Base, NDArrayOperatorsMixin, abc.ABC):
    """
    Abstract base class for time series objects.
    Implement most of the shared functions across concrete classes `Tsd`, `TsdFrame`, `TsdTensor`
    """

    def __init__(self, t, d, time_units="s", time_support=None, load_array=True):
        super().__init__(t, time_units, time_support)

        if load_array or isinstance(d, np.ndarray):
            self.values = convert_to_array(d, "d")
        else:
            if not is_array_like(d):
                raise TypeError(
                    "Data should be array-like, i.e. be indexable, iterable and, have attributes "
                    "`shape`, `ndim` and, `dtype`)."
                )
            self.values = d

        assert len(self.index) == len(
            self.values
        ), "Length of values {} does not match length of index {}".format(
            len(self.values), len(self.index)
        )

        if isinstance(time_support, IntervalSet) and len(self.index):
            starts = time_support.start
            ends = time_support.end
            idx = _restrict(self.index.values, starts, ends)
            t = self.index.values[idx]
            d = self.values[idx]

            self.index = TsIndex(t)
            self.values = d
            self.rate = self.index.shape[0] / np.sum(
                time_support.values[:, 1] - time_support.values[:, 0]
            )

        self.dtype = self.values.dtype

    def __setitem__(self, key, value):
        """setter for time series"""
        try:
            self.values.__setitem__(key, value)
        except IndexError:
            raise IndexError

    def __getattr__(self, name):
        """Allow numpy functions to be attached as attributes of Tsd objects"""
        if hasattr(np, name):
            np_func = getattr(np, name)

            def method(*args, **kwargs):
                return np_func(self, *args, **kwargs)

            return method

        raise AttributeError(
            "Time series object does not have the attribute {}".format(name)
        )

    @property
    def d(self):
        return self.values

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def size(self):
        return self.values.size

    def __array__(self, dtype=None):
        return self.values.astype(dtype)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        # print("In __array_ufunc__")
        # print("     ufunc = ", ufunc)
        # print("     method = ", method)
        # print("     args = ", args)
        # for inp in args:
        #     print(type(inp))
        # print("     kwargs = ", kwargs)

        if method == "__call__":
            new_args = []
            n_object = 0
            for a in args:
                if isinstance(a, self.__class__):
                    new_args.append(a.values)
                    n_object += 1
                else:
                    new_args.append(a)

            # Meant to prevent addition of two Tsd for example
            if n_object > 1:
                return NotImplemented
            else:
                out = ufunc(*new_args, **kwargs)

            if isinstance(out, np.ndarray) or is_array_like(out):
                if out.shape[0] == self.index.shape[0]:
                    kwargs = {}
                    if hasattr(self, "columns"):
                        kwargs["columns"] = self.columns
                    return _get_class(out)(
                        t=self.index, d=out, time_support=self.time_support, **kwargs
                    )
                else:
                    return out
            else:
                return out
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func in [
            np.sort,
            np.lexsort,
            np.sort_complex,
            np.partition,
            np.argpartition,
        ]:
            return NotImplemented

        if hasattr(np.fft, func.__name__):
            return NotImplemented

        if func in [np.split, np.array_split, np.dsplit, np.hsplit, np.vsplit]:
            return _split_tsd(func, *args, **kwargs)

        if func in [np.concatenate, np.vstack, np.hstack, np.dstack]:
            return _concatenate_tsd(func, *args, **kwargs)

        new_args = []
        for a in args:
            if isinstance(a, self.__class__):
                new_args.append(a.values)
            else:
                new_args.append(a)

        out = func._implementation(*new_args, **kwargs)

        if isinstance(out, np.ndarray) or is_array_like(out):
            # # if dims increased in any case, we can't return safely a time series
            # if out.ndim > self.ndim:
            #     return out
            if out.shape[0] == self.index.shape[0]:
                kwargs = {}
                if hasattr(self, "columns"):
                    kwargs["columns"] = self.columns
                return _get_class(out)(
                    t=self.index, d=out, time_support=self.time_support, **kwargs
                )
            else:
                return out
        else:
            return out

    def as_array(self):
        """
        Return the data.

        Returns
        -------
        out: array-like
            _
        """
        return self.values

    def data(self):
        """
        Return the data.

        Returns
        -------
        out: array-like
            _
        """
        return self.values

    def to_numpy(self):
        """
        Return the data as a numpy.ndarray.

        Mostly useful for matplotlib plotting when calling `plot(tsd)`.
        """
        return np.asarray(self.values)

    def copy(self):
        """Copy the data, index and time support"""
        return self.__class__(
            t=self.index.copy(), d=self.values[:].copy(), time_support=self.time_support
        )

    def value_from(self, data, ep=None):
        """
        Replace the value with the closest value from Tsd/TsdFrame/TsdTensor argument

        Parameters
        ----------
        data : Tsd, TsdFrame or TsdTensor
            The object holding the values to replace.
        ep : IntervalSet (optional)
            The IntervalSet object to restrict the operation.
            If None, the time support of the tsd input object is used.

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

        The variable ts is a time series object containing only nan.
        The tsd object containing the values, for example the tracking data, and the epoch to restrict the operation.

        >>> newts = ts.value_from(tsd, ep)

        newts has the same size of ts restrict to ep.

        >>> print(len(ts.restrict(ep)), len(newts))
            52 52
        """
        assert isinstance(
            data, BaseTsd
        ), "First argument should be an instance of Tsd, TsdFrame or TsdTensor"

        t, d, time_support, kwargs = super().value_from(data, ep)
        return data.__class__(t=t, d=d, time_support=time_support, **kwargs)

    def count(self, *args, dtype=None, **kwargs):
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
        t, d, ep = super().count(*args, dtype=dtype, **kwargs)
        return Tsd(t=t, d=d, time_support=ep)

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
        out: Tsd, TsdFrame, TsdTensor
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

        bin_size = TsIndex.format_timestamps(np.array([bin_size]), time_units)[0]

        time_array = self.index.values
        data_array = self.values
        starts = ep.start
        ends = ep.end

        t, d = _bin_average(time_array, data_array, starts, ends, bin_size)

        kwargs = {}
        if hasattr(self, "columns"):
            kwargs["columns"] = self.columns

        return self.__class__(t=t, d=d, time_support=ep, **kwargs)

    def dropna(self, update_time_support=True):
        """Drop every rows containing NaNs. By default, the time support is updated to start and end around the time points that are non NaNs.
        To change this behavior, you can set update_time_support=False.

        Parameters
        ----------
        update_time_support : bool, optional

        Returns
        -------
        Tsd, TsdFrame or TsdTensor
            The time series without the NaNs
        """
        assert isinstance(update_time_support, bool)

        time_array = self.index.values
        data_array = self.values
        starts = self.time_support.start
        ends = self.time_support.end

        t, d, starts, ends = _dropna(
            time_array, data_array, starts, ends, update_time_support, self.ndim
        )

        if update_time_support:
            if is_array_like(starts) and is_array_like(ends):
                ep = IntervalSet(starts, ends)
            else:
                ep = None
        else:
            ep = self.time_support

        kwargs = {}
        if hasattr(self, "columns"):
            kwargs["columns"] = self.columns

        return self.__class__(t=t, d=d, time_support=ep, **kwargs)

    def convolve(self, array, ep=None, trim="both"):
        """Return the discrete linear convolution of the time series with a one dimensional sequence.

        A parameter ep can control the epochs for which the convolution will apply. Otherwise the convolution is made over the time support.

        This function assume a constant sampling rate of the time series.

        The only mode supported is full. The returned object is trimmed to match the size of the original object. The parameter trim controls which side the trimming operates. Default is 'both'.

        See the numpy documentation here : https://numpy.org/doc/stable/reference/generated/numpy.convolve.html

        Parameters
        ----------
        array : array-like
            1-D or 2-D array with kernel(s) to be used for convolution.
            First dimension is assumed to be time.
        ep : None, optional
            The epochs to apply the convolution
        trim : str, optional
            The side on which to trim the output of the convolution ('left', 'right', 'both' [default])

        Returns
        -------
        Tsd, TsdFrame or TsdTensor
            The convolved time series
        """
        if not is_array_like(array):
            raise IOError(
                "Input should be a numpy array (or jax array if pynajax is installed)."
            )

        if len(array) == 0:
            raise IOError("Input array is length 0")

        if array.ndim > 2:
            raise IOError("Array should be 1 or 2 dimension.")

        if trim not in ["both", "left", "right"]:
            raise IOError("Unknow argument. trim should be 'both', 'left' or 'right'.")

        time_array = self.index.values
        data_array = self.values

        if ep is None:
            ep = self.time_support
            starts = ep.start
            ends = ep.end
        else:
            if not isinstance(ep, IntervalSet):
                raise IOError("ep should be an object of type IntervalSet")
            starts = ep.start
            ends = ep.end
            idx = _restrict(time_array, starts, ends)
            time_array = time_array[idx]
            data_array = data_array[idx]

        new_data_array = _convolve(time_array, data_array, starts, ends, array, trim)

        kwargs_dict = dict(time_support=ep)

        nap_class = _get_class(new_data_array)

        if isinstance(self, TsdFrame) and array.ndim == 1:  # keep columns
            kwargs_dict["columns"] = self.columns

        return nap_class(t=time_array, d=new_data_array, **kwargs_dict)

    def smooth(self, std, windowsize=None, time_units="s", size_factor=100, norm=True):
        """Smooth a time series with a gaussian kernel.

        `std` is the standard deviation of the gaussian kernel in units of time.
        If only `std` is passed, the function will compute the standard deviation and size in number
        of time points automatically based on the sampling rate of the time series.
        For example, if the time series `tsd` has a sample rate of 100 Hz and `std` is 50 ms,
        the standard deviation will be converted to an integer through
        `tsd.rate * std = int(100 * 0.05) = 5`.

        If `windowsize` is None, the function will select a kernel size as 100 times
        the std in number of time points. This behavior can be controlled with the
        parameter `size_factor`.

        `norm` set to True normalizes the gaussian kernel to sum to 1.

        In the following example, a time series `tsd` with a sampling rate of 100 Hz
        is convolved with a gaussian kernel. The standard deviation is
        0.05 second and the windowsize is 2 second. When instantiating the gaussian kernel
        from scipy, it corresponds to parameters `M = 200` and `std=5`

            >>> tsd.smooth(std=0.05, windowsize=2, time_units='s', norm=False)

        This line is equivalent to :

            >>> from scipy.signal.windows import gaussian
            >>> kernel = gaussian(M = 200, std=5)
            >>> tsd.convolve(window)

        It is generally a good idea to visualize the kernel before applying any convolution.

        See the scipy documentation for the [gaussian window](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.gaussian.html)

        Parameters
        ----------
        std : Number
            Standard deviation in units of time
        windowsize : Number
            Size of the gaussian window in units of time.
        time_units : str, optional
            The time units in which std and windowsize are specified ('us', 'ms', 's' [default]).
        size_factor : int, optional
            How long should be the kernel size as a function of the standard deviation. Default is 100.
            Bypassed if windowsize is used.
        norm : bool, optional
            Whether to normalized the gaussian kernel or not. Default is `True`.

        Returns
        -------
        Tsd, TsdFrame, TsdTensor
            Time series convolved with a gaussian kernel

        """
        if not isinstance(std, (int, float)):
            raise IOError("std should be type int or float")
        if not isinstance(size_factor, int):
            raise IOError("size_factor should be of type int")
        if not isinstance(norm, bool):
            raise IOError("norm should be of type boolean")
        if not isinstance(time_units, str):
            raise IOError("time_units should be of type str")

        std = TsIndex.format_timestamps(np.array([std]), time_units)[0]
        std_size = int(self.rate * std)

        if windowsize is not None:
            if not isinstance(windowsize, Number):
                raise IOError("windowsize should be type int or float")
            windowsize = TsIndex.format_timestamps(np.array([windowsize]), time_units)[
                0
            ]
            M = int(self.rate * windowsize)
        else:
            M = std_size * size_factor

        if M % 2 == 0:
            M += 1

        window = signal.windows.gaussian(M=M, std=std_size)

        if norm:
            window = window / window.sum()

        return self.convolve(window)

    def interpolate(self, ts, ep=None, left=None, right=None):
        """Wrapper of the numpy linear interpolation method. See [numpy interpolate](https://numpy.org/doc/stable/reference/generated/numpy.interp.html)
        for an explanation of the parameters.
        The argument ts should be Ts, Tsd, TsdFrame, TsdTensor to ensure interpolating from sorted timestamps in the right unit,

        Parameters
        ----------
        ts : Ts, Tsd, TsdFrame or TsdTensor
            The object holding the timestamps
        ep : IntervalSet, optional
            The epochs to use to interpolate. If None, the time support of Tsd is used.
        left : None, optional
            Value to return for ts < tsd[0], default is tsd[0].
        right : None, optional
            Value to return for ts > tsd[-1], default is tsd[-1].
        """
        if not isinstance(ts, Base):
            raise IOError(
                "First argument should be an instance of Ts, Tsd, TsdFrame or TsdTensor"
            )

        if left is not None and not isinstance(left, Number):
            raise IOError("Argument left should be of type float or int")

        if right is not None and not isinstance(right, Number):
            raise IOError("Argument right should be of type float or int")

        if ep is None:
            ep = self.time_support
        else:
            if not isinstance(ep, IntervalSet):
                raise IOError("ep should be an object of type IntervalSet")

        new_t = ts.restrict(ep).index

        new_shape = (
            len(new_t) if self.values.ndim == 1 else (len(new_t),) + self.shape[1:]
        )
        new_d = np.full(new_shape, np.nan)

        start = 0
        for i in range(len(ep)):
            t = ts.get(ep[i, 0], ep[i, 1])
            tmp = self.get(ep[i, 0], ep[i, 1])

            if len(t) and len(tmp):
                if self.values.ndim == 1:
                    new_d[start : start + len(t)] = np.interp(
                        t.index.values,
                        tmp.index.values,
                        tmp.values,
                        left=left,
                        right=right,
                    )
                else:
                    interpolated_values = np.apply_along_axis(
                        lambda row: np.interp(
                            t.index.values,
                            tmp.index.values,
                            row,
                            left=left,
                            right=right,
                        ),
                        0,
                        tmp.values,
                    )
                    new_d[start : start + len(t), ...] = interpolated_values

            start += len(t)
        kwargs_dict = dict(time_support=ep)
        if hasattr(self, "columns"):
            kwargs_dict["columns"] = self.columns
        return self.__class__(t=new_t, d=new_d, **kwargs_dict)


class TsdTensor(BaseTsd):
    """
    TsdTensor

    Attributes
    ----------
    rate : float
        Frequency of the time series (Hz) computed over the time support
    time_support : IntervalSet
        The time support of the time series
    """

    def __init__(
        self, t, d, time_units="s", time_support=None, load_array=True, **kwargs
    ):
        """
        TsdTensor initializer

        Parameters
        ----------
        t : numpy.ndarray
            the time index t
        d : numpy.ndarray
            The data
        time_units : str, optional
            The time units in which times are specified ('us', 'ms', 's' [default]).
        time_support : IntervalSet, optional
            The time support of the TsdFrame object
        load_array : bool, optional
            Whether the data should be converted to a numpy (or jax) array. Useful when passing a memory map object like zarr.
            Default is True. Does not apply if `d` is already a numpy array.

        """
        super().__init__(t, d, time_units, time_support, load_array)

        assert (
            self.values.ndim >= 3
        ), "Data should have more than 2 dimensions. If ndim < 3, use TsdFrame or Tsd object"

        self.nap_class = self.__class__.__name__
        self._initialized = True

    def __repr__(self):
        headers = ["Time (s)", ""]
        bottom = "dtype: {}".format(self.dtype) + ", shape: {}".format(self.shape)

        max_rows = 2
        rows = _get_terminal_size()[1]
        max_rows = np.maximum(rows - 10, 2)

        if len(self):

            def create_str(array):
                if array.ndim == 1:
                    if len(array) > 2:
                        return np.array2string(
                            np.array([array[0], array[-1]]),
                            precision=6,
                            separator=" ... ",
                        )
                    else:
                        return np.array2string(array, precision=6, separator=", ")
                else:
                    return "[" + create_str(array[0]) + " ...]"

            _str_ = []
            if self.shape[0] > max_rows:
                n_rows = max_rows // 2
                for i, array in zip(self.index[0:n_rows], self.values[0:n_rows]):
                    _str_.append([i.__repr__(), create_str(array)])
                _str_.append(["...", ""])
                for i, array in zip(
                    self.index[-n_rows:],
                    self.values[self.values.shape[0] - n_rows : self.values.shape[0]],
                ):
                    _str_.append([i.__repr__(), create_str(array)])
            else:
                for i, array in zip(self.index, self.values):
                    _str_.append([i.__repr__(), create_str(array)])

            return tabulate(_str_, headers=headers, colalign=("left",)) + "\n" + bottom

        else:
            return tabulate([], headers=headers) + "\n" + bottom

    def __getitem__(self, key, *args, **kwargs):
        output = self.values.__getitem__(key)
        if isinstance(key, tuple):
            index = self.index.__getitem__(key[0])
        else:
            index = self.index.__getitem__(key)

        if isinstance(index, Number):
            index = np.array([index])

        if all(is_array_like(a) for a in [index, output]):
            if output.shape[0] == index.shape[0]:
                if output.ndim == 1:
                    return Tsd(t=index, d=output, time_support=self.time_support)
                elif output.ndim == 2:
                    return TsdFrame(
                        t=index, d=output, time_support=self.time_support, **kwargs
                    )
                else:
                    return TsdTensor(t=index, d=output, time_support=self.time_support)

            else:
                return output
        else:
            return output

    def save(self, filename):
        """
        Save TsdTensor object in npz format. The file will contain the timestamps, the
        data and the time support.

        The main purpose of this function is to save small/medium sized time series
        objects. For example, you extracted several channels from your recording and
        filtered them. You can save the filtered channels as a npz to avoid
        reprocessing it.

        You can load the object with `nap.load_file`. Keys are 't', 'd', 'start', 'end', 'type'
        and 'columns' for columns names.

        Parameters
        ----------
        filename : str
            The filename

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tsdtensor = nap.TsdTensor(t=np.array([0., 1.]), d = np.zeros((2,3,4)))
        >>> tsdtensor.save("my_path/my_tsdtensor.npz")

        To load you file, you can use the `nap.load_file` function :

        >>> tsdtensor = nap.load_file("my_path/my_tsdtensor.npz")

        Raises
        ------
        RuntimeError
            If filename is not str, path does not exist or filename is a directory.
        """
        filename = self._get_filename(filename)

        np.savez(
            filename,
            t=self.index.values,
            d=self.values,
            start=self.time_support.start,
            end=self.time_support.end,
            type=np.array([self.nap_class], dtype=np.str_),
        )

        return


class TsdFrame(BaseTsd):
    """
    TsdFrame

    Attributes
    ----------
    rate : float
        Frequency of the time series (Hz) computed over the time support
    time_support : IntervalSet
        The time support of the time series
    """

    def __init__(
        self,
        t,
        d=None,
        time_units="s",
        time_support=None,
        columns=None,
        load_array=True,
    ):
        """
        TsdFrame initializer
        A pandas.DataFrame can be passed directly

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
        columns : iterables
            Column names
        load_array : bool, optional
            Whether the data should be converted to a numpy (or jax) array. Useful when passing a memory map object like zarr.
            Default is True. Does not apply if `d` is already a numpy array.
        """

        c = columns

        if isinstance(t, pd.DataFrame):
            d = t.values
            c = t.columns.values
            t = t.index.values
        else:
            assert d is not None, "Missing argument d when initializing TsdFrame"

        super().__init__(t, d, time_units, time_support, load_array)

        assert self.values.ndim <= 2, "Data should be 1 or 2 dimensional."

        if self.values.ndim == 1:
            self.values = np.expand_dims(self.values, 1)

        if c is None or len(c) != self.values.shape[1]:
            c = np.arange(self.values.shape[1], dtype="int")
        else:
            assert (
                len(c) == self.values.shape[1]
            ), "Number of columns should match the second dimension of d"

        self.columns = pd.Index(c)
        self.nap_class = self.__class__.__name__
        self._initialized = True

    @property
    def loc(self):
        return _TsdFrameSliceHelper(self)

    def __repr__(self):
        headers = ["Time (s)"] + [str(k) for k in self.columns]
        bottom = "dtype: {}".format(self.dtype) + ", shape: {}".format(self.shape)

        cols, rows = _get_terminal_size()
        max_cols = np.maximum(cols // 100, 5)
        max_rows = np.maximum(rows - 10, 2)

        if self.shape[1] > max_cols:
            headers = headers[0 : max_cols + 1] + ["..."]

        def round_if_float(x):
            if isinstance(x, float):
                return np.round(x, 5)
            else:
                return x

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(self):
                table = []
                end = ["..."] if self.shape[1] > max_cols else []
                if len(self) > max_rows:
                    n_rows = max_rows // 2
                    for i, array in zip(
                        self.index[0:n_rows], self.values[0:n_rows, 0:max_cols]
                    ):
                        table.append([i] + [round_if_float(k) for k in array] + end)
                    table.append(["..."])
                    for i, array in zip(
                        self.index[-n_rows:],
                        self.values[
                            self.values.shape[0] - n_rows : self.values.shape[0],
                            0:max_cols,
                        ],
                    ):
                        table.append([i] + [round_if_float(k) for k in array] + end)
                    return (
                        tabulate(table, headers=headers, colalign=("left",))
                        + "\n"
                        + bottom
                    )
                else:
                    for i, array in zip(self.index, self.values[:, 0:max_cols]):
                        table.append([i] + [round_if_float(k) for k in array] + end)
                    return (
                        tabulate(table, headers=headers, colalign=("left",))
                        + "\n"
                        + bottom
                    )
            else:
                return tabulate([], headers=headers) + "\n" + bottom

    def __setitem__(self, key, value):
        try:
            if isinstance(key, str):
                new_key = self.columns.get_indexer([key])
                self.values.__setitem__((slice(None, None, None), new_key[0]), value)
            elif hasattr(key, "__iter__") and all([isinstance(k, str) for k in key]):
                new_key = self.columns.get_indexer(key)
                self.values.__setitem__((slice(None, None, None), new_key), value)
            else:
                self.values.__setitem__(key, value)
        except IndexError:
            raise IndexError

    def __getitem__(self, key, *args, **kwargs):
        if (
            isinstance(key, str)
            or hasattr(key, "__iter__")
            and all([isinstance(k, str) for k in key])
        ):
            return self.loc[key]
        else:
            output = self.values.__getitem__(key)
            columns = self.columns

            if isinstance(key, tuple):
                index = self.index.__getitem__(key[0])
                if len(key) == 2:
                    columns = self.columns.__getitem__(key[1])
            else:
                index = self.index.__getitem__(key)

            if isinstance(index, Number):
                index = np.array([index])

            if all(is_array_like(a) for a in [index, output]):
                if output.shape[0] == index.shape[0]:
                    # if isinstance(columns, pd.Index):
                    #     if not pd.api.types.is_integer_dtype(columns):
                    kwargs["columns"] = columns

                    return _get_class(output)(
                        t=index, d=output, time_support=self.time_support, **kwargs
                    )
                else:
                    return output
            else:
                return output

    def as_dataframe(self):
        """
        Convert the TsdFrame object to a pandas.DataFrame object.

        Returns
        -------
        out: pandas.DataFrame
            _
        """
        return pd.DataFrame(
            index=self.index.values, data=self.values, columns=self.columns
        )

    def as_units(self, units="s"):
        """
        Returns a DataFrame with time expressed in the desired unit.

        Parameters
        ----------
        units : str, optional
            ('us', 'ms', 's' [default])

        Returns
        -------
        pandas.DataFrame
            the series object with adjusted times
        """
        t = self.index.in_units(units)
        if units == "us":
            t = t.astype(np.int64)

        df = pd.DataFrame(index=t, data=self.values)
        df.index.name = "Time (" + str(units) + ")"
        df.columns = self.columns.copy()
        return df

    def save(self, filename):
        """
        Save TsdFrame object in npz format. The file will contain the timestamps, the
        data and the time support.

        The main purpose of this function is to save small/medium sized time series
        objects. For example, you extracted several channels from your recording and
        filtered them. You can save the filtered channels as a npz to avoid
        reprocessing it.

        You can load the object with `nap.load_file`. Keys are 't', 'd', 'start', 'end', 'type'
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

        To load you file, you can use the `nap.load_file` function :

        >>> tsdframe = nap.load_file("my_path/my_tsdframe.npz")
        >>> tsdframe
                  a  b
        Time (s)
        0.0       2  3
        1.0       4  5


        Raises
        ------
        RuntimeError
            If filename is not str, path does not exist or filename is a directory.
        """
        filename = self._get_filename(filename)

        cols_name = self.columns
        if cols_name.dtype == np.dtype("O"):
            cols_name = cols_name.astype(str)

        np.savez(
            filename,
            t=self.index.values,
            d=self.values[:],
            start=self.time_support.start,
            end=self.time_support.end,
            columns=cols_name,
            type=np.array(["TsdFrame"], dtype=np.str_),
        )

        return


class Tsd(BaseTsd):
    """
    A container around numpy.ndarray specialized for neurophysiology time series.

    Tsd provides standardized time representation, plus various functions for manipulating times series.

    Attributes
    ----------
    rate : float
        Frequency of the time series (Hz) computed over the time support
    time_support : IntervalSet
        The time support of the time series
    """

    def __init__(
        self, t, d=None, time_units="s", time_support=None, load_array=True, **kwargs
    ):
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
        load_array : bool, optional
            Whether the data should be converted to a numpy (or jax) array. Useful when passing a memory map object like zarr.
            Default is True. Does not apply if `d` is already a numpy array.
        """
        if isinstance(t, pd.Series):
            d = t.values
            t = t.index.values
        else:
            assert d is not None, "Missing argument d when initializing Tsd"

        super().__init__(t, d, time_units, time_support, load_array)

        assert self.values.ndim == 1, "Data should be 1 dimensional"

        self.nap_class = self.__class__.__name__
        self._initialized = True

    def __repr__(self):
        headers = ["Time (s)", ""]
        bottom = "dtype: {}".format(self.dtype) + ", shape: {}".format(self.shape)

        max_rows = 2
        rows = _get_terminal_size()[1]
        max_rows = np.maximum(rows - 10, 2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(self):
                if len(self) > max_rows:
                    n_rows = max_rows // 2
                    table = []
                    for i, v in zip(self.index[0:n_rows], self.values[0:n_rows]):
                        table.append([i, v])
                    table.append(["..."])
                    for i, v in zip(
                        self.index[-n_rows:],
                        self.values[
                            self.values.shape[0] - n_rows : self.values.shape[0]
                        ],
                    ):
                        table.append([i, v])

                    return (
                        tabulate(table, headers=headers, colalign=("left",))
                        + "\n"
                        + bottom
                    )
                else:
                    return (
                        tabulate(
                            np.vstack((self.index, self.values)).T,
                            headers=headers,
                            colalign=("left",),
                        )
                        + "\n"
                        + bottom
                    )
            else:
                return tabulate([], headers=headers) + "\n" + bottom

    def __getitem__(self, key, *args, **kwargs):
        output = self.values.__getitem__(key)
        if isinstance(key, tuple):
            index = self.index.__getitem__(key[0])
        else:
            index = self.index.__getitem__(key)

        if isinstance(index, Number):
            index = np.array([index])

        if all(is_array_like(a) for a in [index, output]):
            if output.shape[0] == index.shape[0]:
                return _get_class(output)(
                    t=index, d=output, time_support=self.time_support, **kwargs
                )
            else:
                return output
        else:
            return output

    def as_series(self):
        """
        Convert the Ts/Tsd object to a pandas.Series object.

        Returns
        -------
        out: pandas.Series
            _
        """
        return pd.Series(
            index=self.index.values, data=self.values, copy=True, dtype="float64"
        )

    def as_units(self, units="s"):
        """
        Returns a pandas Series with time expressed in the desired unit.

        Parameters
        ----------
        units : str, optional
            ('us', 'ms', 's' [default])

        Returns
        -------
        pandas.Series
            the series object with adjusted times
        """
        ss = self.as_series()
        t = self.index.in_units(units)
        if units == "us":
            t = t.astype(np.int64)
        ss.index = t
        ss.index.name = "Time (" + str(units) + ")"
        return ss

    def threshold(self, thr, method="above"):
        """
        Apply a threshold function to the tsd to return a new tsd
        with the time support being the epochs above/below/>=/<= the threshold

        Parameters
        ----------
        thr : float
            The threshold value
        method : str, optional
            The threshold method ("above"[default], "below", "aboveequal", "belowequal")

        Returns
        -------
        out: Tsd
            All the time points below/ above/greater than equal to/less than equal to the threshold

        Raises
        ------
        ValueError
            Raise an error if method is unknown.
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
        if method not in ["above", "below", "aboveequal", "belowequal"]:
            raise ValueError(
                "Method {} for thresholding is not accepted.".format(method)
            )

        time_array = self.index.values
        data_array = self.values
        starts = self.time_support.start
        ends = self.time_support.end

        t, d, ns, ne = _threshold(time_array, data_array, starts, ends, thr, method)
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

        return ts_group.TsGroup(
            group, time_support=self.time_support, bypass_check=True
        )

    def save(self, filename):
        """
        Save Tsd object in npz format. The file will contain the timestamps, the
        data and the time support.

        The main purpose of this function is to save small/medium sized time series
        objects. For example, you extracted one channel from your recording and
        filtered it. You can save the filtered channel as a npz to avoid
        reprocessing it.

        You can load the object with `nap.load_file`. Keys are 't', 'd', 'start', 'end' and 'type'.
        See the example below.

        Parameters
        ----------
        filename : str
            The filename

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tsd = nap.Tsd(t=np.array([0., 1.]), d = np.array([2, 3]))
        >>> tsd.save("my_path/my_tsd.npz")

        To load you file, you can use the `nap.load_file` function :

        >>> tsd = nap.load_file("my_path/my_tsd.npz")
        >>> tsd
        Time (s)
        0.0    2
        1.0    3
        dtype: int64

        Raises
        ------
        RuntimeError
            If filename is not str, path does not exist or filename is a directory.
        """
        filename = self._get_filename(filename)
        np.savez(
            filename,
            t=self.index.values,
            d=self.values,
            start=self.time_support.start,
            end=self.time_support.end,
            type=np.array([self.nap_class], dtype=np.str_),
        )

        return


class Ts(Base):
    """
    Timestamps only object for a time series with only time index,

    Attributes
    ----------
    rate : float
        Frequency of the time series (Hz) computed over the time support
    time_support : IntervalSet
        The time support of the time series
    """

    def __init__(self, t, time_units="s", time_support=None):
        """
        Ts Initializer

        Parameters
        ----------
        t : numpy.ndarray or pandas.Series
            An object transformable in timestamps, or a pandas.Series equivalent (if d is None)
        time_units : str, optional
            The time units in which times are specified ('us', 'ms', 's' [default])
        time_support : IntervalSet, optional
            The time support of the Ts object
        """
        super().__init__(t, time_units, time_support)

        if isinstance(time_support, IntervalSet) and len(self.index):
            starts = time_support.start
            ends = time_support.end
            idx = _restrict(self.index.values, starts, ends)
            self.index = TsIndex(self.index.values[idx])
            self.rate = self.index.shape[0] / np.sum(
                time_support.values[:, 1] - time_support.values[:, 0]
            )

        self.nap_class = self.__class__.__name__
        self._initialized = True

    def __repr__(self):
        upper = "Time (s)"

        max_rows = 2
        rows = _get_terminal_size()[1]
        max_rows = np.maximum(rows - 10, 2)

        if len(self) > max_rows:
            n_rows = max_rows // 2
            _str_ = "\n".join(
                [i.__repr__() for i in self.index[0:n_rows]]
                + ["..."]
                + [i.__repr__() for i in self.index[-n_rows:]]
            )
        else:
            _str_ = "\n".join([i.__repr__() for i in self.index])

        bottom = "shape: {}".format(len(self.index))
        return "\n".join((upper, _str_, bottom))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            index = self.index.__getitem__(key[0])
        else:
            index = self.index.__getitem__(key)

        if isinstance(index, Number):
            index = np.array([index])

        return Ts(t=index, time_support=self.time_support)

    def as_series(self):
        """
        Convert the Ts/Tsd object to a pandas.Series object.

        Returns
        -------
        out: pandas.Series
            _
        """
        return pd.Series(index=self.index.values, dtype="object")

    def as_units(self, units="s"):
        """
        Returns a pandas Series with time expressed in the desired unit.

        Parameters
        ----------
        units : str, optional
            ('us', 'ms', 's' [default])

        Returns
        -------
        pandas.Series
            the series object with adjusted times
        """
        t = self.index.in_units(units)
        if units == "us":
            t = t.astype(np.int64)
        ss = pd.Series(index=t, dtype="object")
        ss.index.name = "Time (" + str(units) + ")"
        return ss

    def value_from(self, data, ep=None):
        """
        Replace the value with the closest value from Tsd/TsdFrame/TsdTensor argument

        Parameters
        ----------
        data : Tsd, TsdFrame or TsdTensor
            The object holding the values to replace.
        ep : IntervalSet (optional)
            The IntervalSet object to restrict the operation.
            If None, the time support of the tsd input object is used.

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

        The variable ts is a time series object containing only nan.
        The tsd object containing the values, for example the tracking data, and the epoch to restrict the operation.

        >>> newts = ts.value_from(tsd, ep)

        newts is the same size as ts restrict to ep.

        >>> print(len(ts.restrict(ep)), len(newts))
            52 52
        """
        assert isinstance(
            data, BaseTsd
        ), "First argument should be an instance of Tsd, TsdFrame or TsdTensor"

        t, d, time_support, kwargs = super().value_from(data, ep)

        return data.__class__(t, d, time_support=time_support, **kwargs)

    def count(self, *args, dtype=None, **kwargs):
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
        t, d, ep = super().count(*args, dtype=dtype, **kwargs)
        return Tsd(t=t, d=d, time_support=ep)

    def fillna(self, value):
        """
        Similar to pandas fillna function.

        Parameters
        ----------
        value : Number
            Value for filling

        Returns
        -------
        Tsd


        """
        assert isinstance(value, Number), "Only a scalar can be passed to fillna"
        d = np.empty(len(self))
        d.fill(value)
        return Tsd(t=self.index, d=d, time_support=self.time_support)

    def save(self, filename):
        """
        Save Ts object in npz format. The file will contain the timestamps and
        the time support.

        The main purpose of this function is to save small/medium sized timestamps
        object.

        You can load the object with `nap.load_file`. Keys are 't', 'start' and 'end' and 'type'.
        See the example below.

        Parameters
        ----------
        filename : str
            The filename

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> ts = nap.Ts(t=np.array([0., 1., 1.5]))
        >>> ts.save("my_path/my_ts.npz")

        To load you file, you can use the `nap.load_file` function :

        >>> ts = nap.load_file("my_path/my_ts.npz")
        >>> ts
        Time (s)
        0.0
        1.0
        1.5

        Raises
        ------
        RuntimeError
            If filename is not str, path does not exist or filename is a directory.
        """
        filename = self._get_filename(filename)

        np.savez(
            filename,
            t=self.index.values,
            start=self.time_support.start,
            end=self.time_support.end,
            type=np.array(["Ts"], dtype=np.str_),
        )

        return
