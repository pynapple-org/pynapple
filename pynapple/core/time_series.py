"""

Pynapple time series are containers specialized for neurophysiological time series.

They provides standardized time representation, plus various functions for manipulating times series with identical sampling frequency.

Multiple time series object are available depending on the shape of the data.

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

from ._core_functions import (
    _bin_average,
    _convolve,
    _count,
    _dropna,
    _restrict,
    _threshold,
)
from .base_class import _Base, add_base_docstring
from .interval_set import IntervalSet
from .metadata_class import _MetadataMixin, add_meta_docstring, add_or_convert_metadata
from .time_index import TsIndex
from .utils import (
    _concatenate_tsd,
    _convert_iter_to_str,
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


def _initialize_tsd_output(
    input_object, values, time_index=None, time_support=None, kwargs=None
):
    """
    Initialize the output object for time series data, ensuring proper alignment of time indices
    and handling metadata when applicable.

    Parameters
    ----------
    input_object : Tsd | TsdFrame | TsdTensor
        Input object, typically a `Tsd`, `TsdFrame` or `TsdTensor`, used to extract time indices and metadata
        if not provided explicitly.
    values : array-like
        Output data, which can be a NumPy array or an array-like object compatible with `Tsd` or `TsdFrame`.
    time_index : array-like, optional
        Time indices for the output data. If not provided, the indices are extracted from `input_object`.
    time_support : IntervalSet, optional
        Time support (epoch) for the output. If not provided, the time support is extracted from `input_object`.
    kwargs : dict, optional
        Additional keyword arguments for constructing the output object. Supports `columns` and `metadata`
        for `TsdFrame` objects.

    Returns
    -------
    object
        Initialized TSD object (`Tsd`, `TsdFrame`, or equivalent) with the specified time indices, data,
        and metadata, or the original `values` object if it is not array-like.

    Notes
    -----
    - If output is a `TsdFrame` and `input_object` is also a `TsdFrame` with matching shapes, the columns and metadata
      are propagated from input to output, unless explicitly provided in `kwargs`.
    - If `time_index` and `time_support` are not provided, they are extracted from `input_object`.

    Examples
    --------
    # Example usage with TsdFrame
    out = _initialize_tsd_output(input_object=tsd_frame, values=data_array, time_index=time_array, time_support=epoch, kwargs={"columns": cols})

    # Example usage with NumPy array
    out = _initialize_tsd_output(input_object=tsd_obj, values=numpy_array)
    """
    kwargs = kwargs if kwargs is not None else {}

    if isinstance(values, np.ndarray) or is_array_like(values):
        # if time and ep are passed use them, otherwise strip from inp
        time_index = input_object.index if time_index is None else time_index

        if time_index.shape[0] == values.shape[0]:
            # grab time support
            time_support = (
                input_object.time_support if time_support is None else time_support
            )

            cls = _get_class(values)

            # if out will be a tsdframe implement kwargs logic
            if cls is TsdFrame:
                # get eventual setting
                cols = kwargs.get("columns", None)
                metadata = kwargs.get("metadata", None)

                # if input is tsdframe and has the shape grab metadata and cols if
                # not already passed in kwargs
                if isinstance(input_object, TsdFrame) and (
                    values.shape[1] == input_object.shape[1]
                ):
                    cols = (
                        cols
                        if cols is not None
                        else getattr(input_object, "columns", None)
                    )
                    metadata = (
                        metadata
                        if metadata is not None
                        else getattr(input_object, "metadata", None)
                    )

                # update the kwargs
                kwargs.update({"columns": cols, "metadata": metadata})

            return cls(t=time_index, d=values, time_support=time_support, **kwargs)

    return values


class _BaseTsd(_Base, NDArrayOperatorsMixin, abc.ABC):
    """
    Abstract base class for time series objects.
    Implement most of the shared functions across concrete classes `Tsd`, `TsdFrame`, `TsdTensor`
    """

    values: np.ndarray
    """An array of the time series data"""

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

    def _define_instance(self, time_index, time_support, values=None, **kwargs):
        """
        Define a new class instance.

        Optional parameters for initialization are either passed to the function or are grabbed from self.
        """
        return _initialize_tsd_output(
            self,
            values,
            time_index=time_index,
            time_support=time_support,
            kwargs=kwargs,
        )

    def __setitem__(self, key, value):
        """setter for time series"""
        if isinstance(key, _BaseTsd):
            key = key.d
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
        return np.asarray(self.values, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
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

            return _initialize_tsd_output(self, out)
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
        return _initialize_tsd_output(self, out)

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

        return _initialize_tsd_output(self, d, time_index=t, time_support=ep)

    def dropna(self, update_time_support=True):
        """Drop every row containing NaNs. By default, the time support is updated to start and end around the time points that are non NaNs.
        To change this behavior, you can set update_time_support=False.

        Parameters
        ----------
        update_time_support : bool, optional

        Returns
        -------
        Tsd, TsdFrame or TsdTensor
            The time series without the NaNs
        """
        if not isinstance(update_time_support, bool):
            raise TypeError("Argument update_time_support should be of type bool")

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

        return _initialize_tsd_output(self, d, time_index=t, time_support=ep)

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

        return _initialize_tsd_output(
            self, new_data_array, time_index=time_array, time_support=ep
        )

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

    def decimate(self, down, order=8, filter_type="iir", ep=None):
        """
        Downsample the time series by an integer factor after an antialiasing filter.

        As default, applies an antialiasing Chebyshev type I filter and downsample.
        The filter is a low pass-filter, if ``filter_type`` is set to "fir",
        applies a 30 point filter with Hamming window.

        Parameters
        ----------
        down : int
            The down-sampling factor.
        order : int, optional
            The order of the filter. Default is 8.
        filter_type : literal, "iir" or "fir".
            The filter type. Default is "iir".
        ep: IntervalSet
            The epoch over which applying the decimate algorithm.

        Example
        -------
        .. plot::
            :include-source: True
            :caption: Downsample with decimate

            >>> import pynapple as nap
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> noisy_data = np.random.rand(100) + np.sin(np.linspace(0, 2 * np.pi, 100))
            >>> tsd = nap.Tsd(t=np.arange(100), d=noisy_data)
            >>> new_tsd = tsd.decimate(down=4)
            >>> plt.plot(tsd, color="k", label="original") #doctest: +ELLIPSIS
            [<matplotlib.lines.Line2D at ...
            >>> plt.plot(new_tsd, color="r", marker="o", label="decimate") #doctest: +ELLIPSIS
            [<matplotlib.lines.Line2D at ...
            >>> plt.plot(tsd[::4], color="g", marker="o", label="naive downsample") #doctest: +ELLIPSIS
            [<matplotlib.lines.Line2D at ...
            >>> plt.legend() #doctest: +ELLIPSIS
            <matplotlib.legend.Legend at ...
            >>> plt.show()

        """
        if not isinstance(down, int):
            raise IOError(
                "Invalid value for 'down': Parameter 'down' should be of type int."
            )
        if not isinstance(order, int) or order <= 0:
            raise IOError(
                "Invalid value for 'order': Parameter 'order' should be a positive int."
            )
        if filter_type not in ["fir", "iir"]:
            raise IOError("'filter_type' should be one of 'fir', 'iir'.")

        if ep is None:
            ep = self.time_support

        if not isinstance(ep, IntervalSet):
            raise IOError("ep should be an object of type IntervalSet")

        # apply scipy filter
        _, n_vals = _count(self.t, ep.start, ep.end, dtype=np.int32)
        deltas = np.ceil(n_vals / down).astype(int)
        tot_samples = np.sum(deltas)
        new_data = np.zeros((tot_samples, *self.d.shape[1:]))
        new_time = np.zeros(tot_samples)
        i_start = 0
        for e, d in zip(ep, deltas):
            slc = self.get_slice(start=e.start[0], end=e.end[0])
            # TODO: remove conversion when add jax backend is available
            new_data[i_start : i_start + d] = signal.decimate(
                np.asarray(self.d[slc]),
                down,
                n=order,
                ftype=filter_type,
                axis=0,
                zero_phase=True,
            )
            new_time[i_start : i_start + d] = self.t[slc][::down]
            i_start += d

        return _initialize_tsd_output(
            self, new_data, time_index=new_time, time_support=ep
        )

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
        if not isinstance(ts, _Base):
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
            t = ts.t[ts.get_slice(ep[i, 0], ep[i, 1])]
            xp = self.t[self.get_slice(ep[i, 0], ep[i, 1])]
            fp = self.d[self.get_slice(ep[i, 0], ep[i, 1])]
            if len(t) and len(xp):
                if self.values.ndim == 1:
                    new_d[start : start + len(t)] = np.interp(
                        t,
                        xp,
                        fp,
                        left=left,
                        right=right,
                    )
                else:
                    interpolated_values = np.apply_along_axis(
                        lambda row: np.interp(
                            t,
                            xp,
                            row,
                            left=left,
                            right=right,
                        ),
                        0,
                        fp,
                    )
                    new_d[start : start + len(t), ...] = interpolated_values

            start += len(t)

        return _initialize_tsd_output(self, new_d, time_index=new_t, time_support=ep)

    def derivative(self, ep=None):
        """Computes the derivative of the time series with respect to time. Wraps numpy.gradient.

        Parameters
        ----------
        ep : IntervalSet, optional
            The epochs to calculate derivatives. If None, the time support of Tsd is used.

        Returns
        -------
        Tsd, TsdFrame or TsdTensor
            The derivative of the time series.

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tsd = nap.Tsd(t=np.arange(5), d=np.arange(0, 10, 2))
        >>> tsd_derivative = tsd.derivative()
        >>> tsd_derivative
        Time (s)
        ----------  --
        0            2
        1            2
        2            2
        3            2
        4            2
        dtype: float64, shape: (5,)
        """
        if ep is None:
            ep = self.time_support
        else:
            if not isinstance(ep, IntervalSet):
                raise IOError("ep should be an object of type IntervalSet")

        new_t = self.restrict(ep).index

        new_shape = (
            len(new_t) if self.values.ndim == 1 else (len(new_t),) + self.shape[1:]
        )
        new_d = np.full(new_shape, np.nan)

        start = 0
        for i in range(len(ep)):
            tmp = self.get(ep[i, 0], ep[i, 1])

            if len(tmp):
                if self.values.ndim == 1:
                    new_d[start : start + len(tmp)] = np.gradient(
                        tmp.values, tmp.index.values
                    )
                else:
                    # Compute gradient along first axis (time)
                    new_d[start : start + len(tmp)] = np.gradient(
                        tmp.values, tmp.index.values, axis=0
                    )

            start += len(tmp)

        return _initialize_tsd_output(self, new_d, time_index=new_t, time_support=ep)

    def to_trial_tensor(self, ep, align="start", padding_value=np.nan):
        """
        Return trial-based tensor from an IntervalSet object. The shape of the tensor array is
        (shape of time series, number of trials, number of  time points)

        The `align` parameter controls how the time series are aligned. If `align="start"`, the time
        series are aligned to the start of each trial. If `align="end"`, the time series are aligned
        to the end of each trial.

        If trials have uneven durations, the returned array is padded. The parameter `padding_value`
        determine which value is used to pad the array. Default is NaN.

        Parameters
        ----------
        ep : IntervalSet
            Epochs holding the trials. Each interval can be of unequal size.
        align: str, optional
            How to align the time series ('start' [default], 'end')
        padding_value: Number, optional
            How to pad the array if unequal intervals. Default is np.nan.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tsdframe = nap.TsdFrame(t=np.arange(100), d=np.arange(200).reshape(2,100).T)
        >>> ep = nap.IntervalSet(start=np.arange(20, 100, 20), end=np.arange(20, 100, 20) + np.arange(2, 10, 2))
        >>> print(ep)
          index    start    end
              0       20     22
              1       40     44
              2       60     66
              3       80     88
        shape: (4, 2), time unit: sec.

        Create a trial-based tensor by slicing `tsdframe` for each interval of `ep`.

        >>> tensor = tsdframe.to_trial_tensor(ep)
        >>> tensor
        array([[[ 20.,  21.,  22.,  nan,  nan,  nan,  nan,  nan,  nan],
                [ 40.,  41.,  42.,  43.,  44.,  nan,  nan,  nan,  nan],
                [ 60.,  61.,  62.,  63.,  64.,  65.,  66.,  nan,  nan],
                [ 80.,  81.,  82.,  83.,  84.,  85.,  86.,  87.,  88.]],
               [[120., 121., 122.,  nan,  nan,  nan,  nan,  nan,  nan],
                [140., 141., 142., 143., 144.,  nan,  nan,  nan,  nan],
                [160., 161., 162., 163., 164., 165., 166.,  nan,  nan],
                [180., 181., 182., 183., 184., 185., 186., 187., 188.]]])
        """
        if not isinstance(ep, IntervalSet):
            raise RuntimeError("Argument ep should be of type IntervalSet")
        if align not in ["start", "end"]:
            raise RuntimeError("align should be 'start' or 'end'")

        slices = [self.get_slice(s, e) for s, e in ep.values]
        lengths = list(map(lambda sl: sl.stop - sl.start, slices))
        n_t = max(lengths)
        output = np.ones(shape=(len(ep), n_t, *self.shape[1:])) * padding_value
        if align == "start":
            for i, sl in enumerate(slices):
                output[i, 0 : lengths[i]] = self.values[sl]
        if align == "end":
            for i, sl in enumerate(slices):
                output[i, n_t - lengths[i] :] = self.values[sl]

        if output.ndim > 2:
            output = np.moveaxis(output, source=[0, 1], destination=[-2, -1])

        return output


class TsdTensor(_BaseTsd):
    """
    Container for neurophysiological time series with more than 2 dimensions (for example movies).

    Attributes
    ----------
    rate : float
        Frequency of the time series (Hz) computed over the time support
    time_support : IntervalSet
        The time support of the time series

    Examples
    --------
    Initialize a TsdTensor:

    >>> import pynapple as nap
    >>> import numpy as np
    >>> t = np.arange(10)
    >>> d = np.random.randn(10, 2, 3)
    >>> tsdtensor = nap.TsdTensor(t=t, d=d)
    >>> tsdtensor
    Time (s)
    ----------  -------------------------------
    0           [[-1.493178 ... -1.281017] ...]
    1           [[0.230829 ... 0.437679] ...]
    2           [[-0.462031 ...  0.344506] ...]
    3           [[0.497019 ... 0.469494] ...]
    4           [[0.065921 ... 1.012917] ...]
    5           [[0.158534 ... 1.455523] ...]
    6           [[-2.567728 ...  0.61182 ] ...]
    7           [[0.940799 ... 0.109203] ...]
    8           [[2.340077 ... 0.21885 ] ...]
    9           [[-0.306175 ... -0.447414] ...]
    dtype: float64, shape: (10, 2, 3)

    Initialize a TsdTensor with `time_support`:

    >>> t = np.arange(10)
    >>> d = np.random.randn(10, 2, 3)
    >>> time_support = nap.IntervalSet(start=0, end=4)
    >>> tsdtensor = nap.TsdTensor(t=t, d=d, time_support=time_support)
    >>> tsdtensor
    Time (s)
    ----------  -------------------------------
    0           [[-1.493178 ... -1.281017] ...]
    1           [[0.230829 ... 0.437679] ...]
    2           [[-0.462031 ...  0.344506] ...]
    3           [[0.497019 ... 0.469494] ...]
    4           [[0.065921 ... 1.012917] ...]
    dtype: float64, shape: (5, 2, 3)

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
            Default is True. Does not apply if `d` is already a numpy array  or a numpy memory map.

        """
        super().__init__(t, d, time_units, time_support, load_array)

        if not self.values.ndim >= 3:
            raise RuntimeError(
                "Data should have more than 2 dimensions. If ndim < 3, use TsdFrame or Tsd object"
            )

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
                    _str_.append([i, create_str(array)])
                _str_.append(["...", ""])
                for i, array in zip(
                    self.index[-n_rows:],
                    self.values[self.values.shape[0] - n_rows : self.values.shape[0]],
                ):
                    _str_.append([i, create_str(array)])
            else:
                for i, array in zip(self.index, self.values):
                    _str_.append([i, create_str(array)])

            return tabulate(_str_, headers=headers, colalign=("left",)) + "\n" + bottom

        else:
            return tabulate([], headers=headers) + "\n" + bottom

    def __getitem__(self, key):
        if isinstance(key, Tsd):
            if not np.issubdtype(key.dtype, np.bool_):
                raise ValueError(
                    "When indexing with a Tsd, it must contain boolean values"
                )
            output = self.values[key.values]
            index = self.index[key.values]
        elif isinstance(key, tuple):
            if any(
                isinstance(k, Tsd) and not np.issubdtype(k.dtype, np.bool_) for k in key
            ):
                raise ValueError(
                    "When indexing with a Tsd, it must contain boolean values"
                )
            key = tuple(k.values if isinstance(k, Tsd) else k for k in key)
            output = self.values.__getitem__(key)
            index = self.index.__getitem__(key[0])
        else:
            output = self.values.__getitem__(key)
            index = self.index.__getitem__(key)

        if isinstance(index, Number):
            index = np.array([index])
        return _initialize_tsd_output(self, output, time_index=index)

    @add_base_docstring("time_diff")
    def time_diff(self, align="center", epochs=None):
        """
        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tsdtensor = nap.TsdTensor(t=[1, 3, 5, 6, 8, 12], d=np.ones((6,6,6)))
        >>> epochs = nap.IntervalSet(start=2, end=9, time_units='s')
        >>> tsd_time_diffs = tsdtensor.time_diff(align="center", epochs=epochs)
        >>> tsd_time_diffs
        Time (s)
        ----------  --
        4            2
        5.5          1
        7            2
        dtype: float64, shape: (3,)
        """
        return _Base.time_diff(self, align, epochs)

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


class TsdFrame(_BaseTsd, _MetadataMixin):
    """
    Column-based container for neurophysiological time series.
    A pandas.DataFrame can be passed directly.

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
        Default is True. Does not apply if `d` is already a numpy array or a numpy memory map.
    metadata: pd.DataFrame or dict, optional
        Metadata associated with data columns. Metadata names are pulled from DataFrame columns or dictionary keys.
        The length of the metadata should match the number of data columns.
        If a DataFrame is passed, the index should match the columns of the TsdFrame.

    Examples
    --------
    Initialize a TsdFrame:

    >>> import pynapple as nap
    >>> import numpy as np
    >>> t = np.arange(100)
    >>> d = np.ones((100, 3))
    >>> tsdframe = nap.TsdFrame(t=t, d=d)
    >>> tsdframe
    Time (s)    0    1    2
    ----------  ---  ---  ---
    0.0         1.0  1.0  1.0
    1.0         1.0  1.0  1.0
    2.0         1.0  1.0  1.0
    3.0         1.0  1.0  1.0
    4.0         1.0  1.0  1.0
    ...         ...  ...  ...
    95.0        1.0  1.0  1.0
    96.0        1.0  1.0  1.0
    97.0        1.0  1.0  1.0
    98.0        1.0  1.0  1.0
    99.0        1.0  1.0  1.0
    dtype: float64, shape: (100, 3)

    Initialize a TsdFrame with column names:

    >>> tsdframe = nap.TsdFrame(t=t, d=d, columns=['A', 'B', 'C'])
    >>> tsdframe
    Time (s)    A    B    C
    ----------  ---  ---  ---
    0.0         1.0  1.0  1.0
    1.0         1.0  1.0  1.0
    2.0         1.0  1.0  1.0
    3.0         1.0  1.0  1.0
    4.0         1.0  1.0  1.0
    ...         ...  ...  ...
    95.0        1.0  1.0  1.0
    96.0        1.0  1.0  1.0
    97.0        1.0  1.0  1.0
    98.0        1.0  1.0  1.0
    99.0        1.0  1.0  1.0
    dtype: float64, shape: (100, 3)

    Initialize a TsdFrame with metadata:

    >>> metadata = {"color": ["red", "blue", "green"], "depth": [1, 2, 3]}
    >>> tsdframe = nap.TsdFrame(t=t, d=d, columns=["A", "B", "C"], metadata=metadata)
    >>> tsdframe
    Time (s)    A         B         C
    ----------  --------  --------  --------
    0.0         1.0       1.0       1.0
    1.0         1.0       1.0       1.0
    2.0         1.0       1.0       1.0
    3.0         1.0       1.0       1.0
    4.0         1.0       1.0       1.0
    ...         ...       ...       ...
    95.0        1.0       1.0       1.0
    96.0        1.0       1.0       1.0
    97.0        1.0       1.0       1.0
    98.0        1.0       1.0       1.0
    99.0        1.0       1.0       1.0
    Metadata
    --------    --------  --------  --------
    color       red       blue      green
    depth       1         2         3
    <BLANKLINE>
    dtype: float64, shape: (100, 3)

    Initialize a TsdFrame with a pandas DataFrame:

    >>> import pandas as pd
    >>> data = pd.DataFrame(index=t, columns=["A", "B", "C"], data=d)
    >>> metadata = pd.DataFrame(
    ...    index=["A", "B", "C"],
    ...    columns=["color", "depth"],
    ...    data=[["red", 1], ["blue", 2], ["green", 3]],
    ... )
    >>> tsdframe = nap.TsdFrame(data, metadata=metadata)
    >>> tsdframe
    Time (s)    A         B         C
    ----------  --------  --------  --------
    0.0         1.0       1.0       1.0
    1.0         1.0       1.0       1.0
    2.0         1.0       1.0       1.0
    3.0         1.0       1.0       1.0
    4.0         1.0       1.0       1.0
    ...         ...       ...       ...
    95.0        1.0       1.0       1.0
    96.0        1.0       1.0       1.0
    97.0        1.0       1.0       1.0
    98.0        1.0       1.0       1.0
    99.0        1.0       1.0       1.0
    Metadata
    --------    --------  --------  --------
    color       red       blue      green
    depth       1         2         3
    <BLANKLINE>
    dtype: float64, shape: (100, 3)
    """

    columns: pd.Index
    """Data column names of the TsdFrame"""

    def __init__(
        self,
        t,
        d=None,
        time_units="s",
        time_support=None,
        columns=None,
        load_array=True,
        metadata=None,
    ):
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
        # initialize metadata for class attributes
        _MetadataMixin.__init__(self)
        # to test compatibility with pandas
        # self._metadata = pd.DataFrame(index=self.metadata_index)
        # get current list of attributes
        self._class_attributes = self.__dir__()
        self._class_attributes.append("_class_attributes")
        # set metadata
        self._initialized = True
        self.set_info(metadata)

    @property
    def loc(self):
        # add deprecation warning
        # warnings.warn(
        #     "'loc' will be deprecated in a future version. Use bracket indexing instead.",
        #     DeprecationWarning,
        # )
        return _TsdFrameSliceHelper(self)

    def __repr__(self):
        # Start by determining how many columns and rows.
        # This can be unique for each object
        cols, rows = _get_terminal_size()
        max_cols = np.maximum(cols // 100, 5)
        max_rows = np.maximum(rows - 10, 2)

        # Computing headers and bottom
        headers = ["Time (s)"] + [str(k) for k in self.columns]
        bottom = f"dtype: {self.dtype}, shape: {self.shape}"

        if self.shape[1] > max_cols:
            headers = headers[0 : max_cols + 1] + ["..."]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(self):
                end = ["..."] if self.shape[1] > max_cols else []
                if len(self) > max_rows:
                    n_rows = max_rows // 2
                    ends = np.array([end] * n_rows)
                    table = np.vstack(
                        (
                            np.hstack(
                                (
                                    self.index[0:n_rows, None],
                                    np.round(self.values[0:n_rows, 0:max_cols], 5),
                                    ends,
                                ),
                                dtype=object,
                            ),
                            np.array(
                                [
                                    ["..."]
                                    + ["..."] * np.minimum(max_cols, self.shape[1])
                                    + end
                                ],
                                dtype=object,
                            ),
                            np.hstack(
                                (
                                    self.index[-n_rows:, None],
                                    np.round(self.values[-n_rows:, 0:max_cols], 5),
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
                            np.round(self.values[:, 0:max_cols], 5),
                            ends,
                        ),
                        dtype=object,
                    )
            else:
                table = np.ndarray(shape=(0, self.shape[1] + 1))
                end = []

        # Adding metadata if any.
        try:
            metadata = self._metadata
            row_names = metadata.columns
        except Exception:
            # Necessary for backward compatibility when saving IntervalSet as pickle
            row_names = []

        if len(row_names):
            n_rows = len(row_names)
            max_metarows = 2

            if n_rows > max_metarows:
                row_names = np.hstack((row_names[0:2]))[:, None]
                last_row = ["..."]
                n_rows = max_metarows
            else:
                row_names = np.array(row_names)[:, None]  # Vertically
                last_row = []  # np.ndarray(shape=(0, table.shape[1]))

            ends = np.array([end] * n_rows)

            formatted_metadata = np.array(
                [
                    _convert_iter_to_str(
                        self._metadata.iloc[0 : table.shape[1] - len(end) - 1][
                            self._metadata.columns[k]
                        ]
                    )
                    for k in range(
                        n_rows
                    )  # Which correspond to columns in the metadata object
                ]
            )

            # Make metadata table
            try:
                tmp = np.hstack(
                    (
                        row_names,
                        formatted_metadata,
                        ends,
                    ),
                    dtype=object,
                )
                mtable = np.vstack(
                    (
                        np.array([["Metadata"] + [" "] * (tmp.shape[1] - 1)]),
                        tmp,
                    ),
                    dtype=object,
                )

                if len(last_row):
                    mtable = np.vstack(
                        (mtable, last_row * mtable.shape[1]), dtype=object
                    )

            except ValueError:
                mtable = np.empty(shape=(0, 0))

            if table.shape[1] == mtable.shape[1]:
                table = np.vstack((table, mtable))

        if len(table):
            return tabulate(table, headers=headers, colalign=("left",)) + "\n" + bottom
        else:
            return tabulate([], headers=headers) + "\n" + bottom

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
            super().__setattr__(name, value)

    @add_or_convert_metadata
    def __getattr__(self, name):
        # TsdFrame needs a custom __getattr__ to override default inherited from BaseTsd

        # avoid infinite recursion when pickling due to
        # self._metadata.column having attributes '__reduce__', '__reduce_ex__'
        if name in ("__getstate__", "__setstate__", "__reduce__", "__reduce_ex__"):
            raise AttributeError(name)

        # try:
        #     metadata = self._metadata
        # except (AttributeError, RecursionError):
        #     metadata = {}  # pd.DataFrame(index=self.columns)
        metadata = self._metadata

        if name == "_metadata":
            return metadata
        elif name in metadata.keys():
            return _MetadataMixin.__getattr__(self, name)
        else:
            return super().__getattr__(name)

    def __setitem__(self, key, value):
        if isinstance(key, Tsd):
            try:
                assert np.issubdtype(key.dtype, np.bool_)
            except AssertionError:
                raise ValueError(
                    "When indexing with a Tsd, it must contain boolean values"
                )
            key = key.d
        try:
            if isinstance(key, str):
                if key in self.columns:
                    new_key = self.columns.get_indexer([key])
                    self.values.__setitem__(
                        (slice(None, None, None), new_key[0]), value
                    )
                else:
                    _MetadataMixin.__setitem__(self, key, value)
            elif hasattr(key, "__iter__") and all([isinstance(k, str) for k in key]):
                new_key = self.columns.get_indexer(key)
                self.values.__setitem__((slice(None, None, None), new_key), value)
            else:
                self.values.__setitem__(key, value)
        except IndexError:
            raise IndexError

    @add_or_convert_metadata
    def __getitem__(self, key, *args, **kwargs):
        if isinstance(key, tuple):
            key = tuple(k.values if hasattr(k, "values") else k for k in key)
        if isinstance(key, Tsd):
            try:
                assert np.issubdtype(key.dtype, np.bool_)
            except AssertionError:
                raise ValueError(
                    "When indexing with a Tsd, it must contain boolean values"
                )
            key = key.d
        elif isinstance(key, str):
            if key in self.columns:
                with warnings.catch_warnings():
                    # ignore deprecated warning for loc
                    warnings.simplefilter("ignore")
                    return self.loc[key]
            else:
                return _MetadataMixin.__getitem__(self, key)
        elif hasattr(key, "__iter__") and all([isinstance(k, str) for k in key]):
            if all(k in self.columns for k in key):
                with warnings.catch_warnings():
                    # ignore deprecated warning for loc
                    warnings.simplefilter("ignore")
                    return self.loc[key]
            else:
                return _MetadataMixin.__getitem__(self, key)
        else:
            output = self.values.__getitem__(key)
            columns = self.columns

            if isinstance(key, tuple):
                index = self.index.__getitem__(key[0])
                if len(key) == 2:
                    columns = self.columns.__getitem__(key[1])
            else:
                index = self.index.__getitem__(key)

            # if isinstance(index, Number):
            #     index = np.array([index])

            if all(is_array_like(a) for a in [index, output]):
                if isinstance(key, tuple):
                    if (
                        len(index) == 1
                        and output.ndim == 1
                        and not isinstance(key[1], (int, np.integer))
                    ):
                        output = output[None, :]
                    elif (
                        (output.ndim == 1)
                        and isinstance(key[1], (list, np.ndarray))
                        and (len(columns) == 1)
                    ):
                        # reshape output of single column if column key is a list or array
                        output = output[:, None]
                # if getting a row (1 dim implied)
                elif isinstance(key, Number):
                    output = output[None, :]

                kwargs["columns"] = columns
                kwargs["metadata"] = self._metadata.loc[columns]
                return _initialize_tsd_output(
                    self, output, time_index=index, kwargs=kwargs
                )
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

    @add_base_docstring("time_diff")
    def time_diff(self, align="center", epochs=None):
        """
        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tsdframe = nap.TsdFrame(t=[1, 3, 5, 6, 8, 12], d=np.ones((6, 2)))
        >>> epochs = nap.IntervalSet(start=2, end=9, time_units='s')
        >>> tsd_time_diffs = tsdframe.time_diff(align="center", epochs=epochs)
        >>> tsd_time_diffs
        Time (s)
        ----------  --
        4            2
        5.5          1
        7            2
        dtype: float64, shape: (3,)
        """
        return _Base.time_diff(self, align, epochs)

    # @add_or_convert_metadata
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
            _metadata=dict(self._metadata),  # save metadata as dictionary
        )

        return

    @add_meta_docstring("set_info")
    def set_info(self, metadata=None, **kwargs):
        """
        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tsdframe = nap.TsdFrame(t=np.arange(5), d=np.ones((5, 3)), columns=["a", "b", "c"])

        To add metadata with a pandas.DataFrame:

        >>> import pandas as pd
        >>> metadata = pd.DataFrame(index=tsdframe.columns, data=["red", "blue", "green"], columns=["color"])
        >>> tsdframe.set_info(metadata)
        >>> tsdframe
        Time (s)    a         b         c
        ----------  --------  --------  --------
        0.0         1.0       1.0       1.0
        1.0         1.0       1.0       1.0
        2.0         1.0       1.0       1.0
        3.0         1.0       1.0       1.0
        4.0         1.0       1.0       1.0
        Metadata
        --------    --------  --------  --------
        color       red       blue      green
        <BLANKLINE>
        dtype: float64, shape: (5, 3)

        To add metadata with a dictionary:

        >>> metadata = {"xpos": [10, 20, 30]}
        >>> tsdframe.set_info(metadata)
        >>> tsdframe
        Time (s)    a         b         c
        ----------  --------  --------  --------
        0.0         1.0       1.0       1.0
        1.0         1.0       1.0       1.0
        2.0         1.0       1.0       1.0
        3.0         1.0       1.0       1.0
        4.0         1.0       1.0       1.0
        Metadata
        --------    --------  --------  --------
        color       red       blue      green
        xpos        10        20        30
        <BLANKLINE>
        dtype: float64, shape: (5, 3)

        To add metadata with a keyword arument (pd.Series, numpy.ndarray, list or tuple):

        >>> ypos = pd.Series(index=tsdframe.columns, data = [10, 10, 10])
        >>> tsdframe.set_info(ypos=ypos)
        >>> tsdframe
        Time (s)    a         b         c
        ----------  --------  --------  --------
        0.0         1.0       1.0       1.0
        1.0         1.0       1.0       1.0
        2.0         1.0       1.0       1.0
        3.0         1.0       1.0       1.0
        4.0         1.0       1.0       1.0
        Metadata
        --------    --------  --------  --------
        color       red       blue      green
        xpos        10        20        30
        ypos        10        10        10
        <BLANKLINE>
        dtype: float64, shape: (5, 3)

        To add metadata as an attribute:

        >>> tsdframe.label = ["a", "b", "c"]
        >>> tsdframe
        Time (s)    a         b         c
        ----------  --------  --------  --------
        0.0         1.0       1.0       1.0
        1.0         1.0       1.0       1.0
        2.0         1.0       1.0       1.0
        3.0         1.0       1.0       1.0
        4.0         1.0       1.0       1.0
        Metadata
        --------    --------  --------  --------
        color       red       blue      green
        xpos        10        20        30
        ypos        10        10        10
        label       a         b         c
        <BLANKLINE>
        dtype: float64, shape: (5, 3)

        To add metadata as a key:

        >>> tsdframe["region"] = ["M1", "M1", "M2"]
        >>> tsdframe
        Time (s)    a         b         c
        ----------  --------  --------  --------
        0.0         1.0       1.0       1.0
        1.0         1.0       1.0       1.0
        2.0         1.0       1.0       1.0
        3.0         1.0       1.0       1.0
        4.0         1.0       1.0       1.0
        Metadata
        --------    --------  --------  --------
        color       red       blue      green
        xpos        10        20        30
        ypos        10        10        10
        label       a         b         c
        region      M1        M1        M2
        <BLANKLINE>
        dtype: float64, shape: (5, 3)

        Metadata can be overwritten:

        >>> tsdframe.set_info(label=["x", "y", "z"])
        >>> tsdframe
        Time (s)    a         b         c
        ----------  --------  --------  --------
        0.0         1.0       1.0       1.0
        1.0         1.0       1.0       1.0
        2.0         1.0       1.0       1.0
        3.0         1.0       1.0       1.0
        4.0         1.0       1.0       1.0
        Metadata
        --------    --------  --------  --------
        color       red       blue      green
        xpos        10        20        30
        ypos        10        10        10
        label       x         y         z
        region      M1        M1        M2
        <BLANKLINE>
        dtype: float64, shape: (5, 3)
        """
        _MetadataMixin.set_info(self, metadata, **kwargs)

    @add_meta_docstring("get_info")
    def get_info(self, key):
        """
        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> metadata = {"l1": [1, 2, 3], "l2": ["x", "x", "y"]}
        >>> tsdframe = nap.TsdFrame(t=np.arange(5), d=np.ones((5, 3)), metadata=metadata)
        >>> print(tsdframe)
        Time (s)    0    1    2
        ----------  ---  ---  ---
        0.0         1.0  1.0  1.0
        1.0         1.0  1.0  1.0
        2.0         1.0  1.0  1.0
        3.0         1.0  1.0  1.0
        4.0         1.0  1.0  1.0
        Metadata
        ----------  ---  ---  ---
        l1          1    2    3
        l2          x    x    y
        dtype: float64, shape: (5, 3)

        To access a single metadata row (transposed to column):

        >>> tsdframe.get_info("l1")
        array([1, 2, 3])

        To access multiple metadata rows (transposed to columns):

        >>> tsdframe.get_info(["l1", "l2"])
             l1    l2
        0    1     x
        1    2     x
        2    3     y

        To access metadata as an attribute:

        >>> tsdframe.l1
        array([1, 2, 3])

        To access metadata as a key:

        >>> tsdframe["l1"]
        array([1, 2, 3])

        Multiple metadata columns can be accessed as keys:

        >>> tsdframe[["l1", "l2"]]
             l1    l2
        0    1     x
        1    2     x
        2    3     y
        """
        return _MetadataMixin.get_info(self, key)

    @add_meta_docstring("drop_info")
    def drop_info(self, key):
        """
        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> metadata = {"l1": [1, 2, 3], "l2": ["x", "x", "y"], "l3": [4, 5, 6]}
        >>> tsdframe = nap.TsdFrame(t=np.arange(5), d=np.ones((5, 3)), metadata=metadata)
        >>> print(tsdframe)
        Time (s)    0    1    2
        ----------  ---  ---  ---
        0.0         1.0  1.0  1.0
        1.0         1.0  1.0  1.0
        2.0         1.0  1.0  1.0
        3.0         1.0  1.0  1.0
        4.0         1.0  1.0  1.0
        Metadata
        ----------  ---  ---  ---
        l1          1    2    3
        l2          x    x    y
        l3          4    5    6
        dtype: float64, shape: (5, 3)

        To drop a single metadata row:

        >>> tsdframe.drop_info("l1")
        >>> tsdframe
        Time (s)    0    1    2
        ----------  ---  ---  ---
        0.0         1.0  1.0  1.0
        1.0         1.0  1.0  1.0
        2.0         1.0  1.0  1.0
        3.0         1.0  1.0  1.0
        4.0         1.0  1.0  1.0
        Metadata
        ----------  ---  ---  ---
        l2          x    x    y
        l3          4    5    6
        dtype: float64, shape: (5, 3)

        To drop multiple metadata rows:

        >>> tsdframe.drop_info(["l2", "l3"])
        >>> tsdframe
          Time (s)    0    1    2
        ----------  ---  ---  ---
                 0    1    1    1
                 1    1    1    1
                 2    1    1    1
                 3    1    1    1
                 4    1    1    1
        dtype: float64, shape: (5, 3)
        """
        return _MetadataMixin.drop_info(self, key)

    @add_or_convert_metadata
    @add_meta_docstring("groupby")
    def groupby(self, by, get_group=None):
        """
        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> metadata = {"l1": [1, 2, 2], "l2": ["x", "x", "y"]}
        >>> tsdframe = nap.TsdFrame(t=np.arange(5), d=np.ones((5, 3)), metadata=metadata)
        >>> print(tsdframe)
        Time (s)    0         1         2
        ----------  --------  --------  --------
        0.0         1.0       1.0       1.0
        1.0         1.0       1.0       1.0
        2.0         1.0       1.0       1.0
        3.0         1.0       1.0       1.0
        4.0         1.0       1.0       1.0
        Metadata
        --------    --------  --------  --------
        l1          1         2         2
        l2          x         x         y
        <BLANKLINE>
        dtype: float64, shape: (5, 3)

        Grouping by a single row:

        >>> tsdframe.groupby("l2")
        {'x': [0, 1], 'y': [2]}

        Grouping by multiple rows:

        >>> tsdframe.groupby(["l1","l2"])
        {(1, 'x'): [0], (2, 'x'): [1], (2, 'y'): [2]}

        Filtering to a specific group using the output dictionary:

        >>> groups = tsdframe.groupby("l2")
        >>> tsdframe[:,groups["x"]]
        Time (s)    0         1
        ----------  --------  --------
        0.0         1.0       1.0
        1.0         1.0       1.0
        2.0         1.0       1.0
        3.0         1.0       1.0
        4.0         1.0       1.0
        Metadata
        --------    --------  --------
        l1          1         2
        l2          x         x
        <BLANKLINE>
        dtype: float64, shape: (5, 2)

        Filtering to a specific group using the get_group argument:

        >>> tsdframe.groupby("l2", get_group="x")
        Time (s)    0         1
        ----------  --------  --------
        0.0         1.0       1.0
        1.0         1.0       1.0
        2.0         1.0       1.0
        3.0         1.0       1.0
        4.0         1.0       1.0
        Metadata
        --------    --------  --------
        l1          1         2
        l2          x         x
        <BLANKLINE>
        dtype: float64, shape: (5, 2)
        """
        return _MetadataMixin.groupby(self, by, get_group)

    @add_meta_docstring("groupby_apply")
    def groupby_apply(self, by, func, input_key=None, **func_kwargs):
        """
        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> metadata = {"l1": [1, 2, 2], "l2": ["x", "x", "y"]}
        >>> tsdframe = nap.TsdFrame(t=np.arange(5), d=np.ones((5, 3)), metadata=metadata)
        >>> print(tsdframe)
        Time (s)    0         1         2
        ----------  --------  --------  --------
        0.0         1.0       1.0       1.0
        1.0         1.0       1.0       1.0
        2.0         1.0       1.0       1.0
        3.0         1.0       1.0       1.0
        4.0         1.0       1.0       1.0
        Metadata
        --------    --------  --------  --------
        l1          1         2         2
        l2          x         x         y
        <BLANKLINE>
        dtype: float64, shape: (5, 3)

        Apply a numpy function:

        >>> tsdframe.groupby_apply("l1", np.sum)
        {1: 5.0, 2: 10.0}

        Apply a custom function:

        >>> tsdframe.groupby_apply("l1", lambda x: x.shape)
        {1: (5, 1), 2: (5, 2)}

        Apply a function with additional arguments:

        >>> tsdframe.groupby_apply("l1", np.sum, axis=0)
        {1: array([5.]), 2: array([5., 5.])}
        """
        return _MetadataMixin.groupby_apply(self, by, func, input_key, **func_kwargs)


class Tsd(_BaseTsd):
    """
    1-dimensional container for neurophysiological time series.

    Tsd provides standardized time representation, plus various functions for manipulating times series.

    Attributes
    ----------
    rate : float
        Frequency of the time series (Hz) computed over the time support
    time_support : IntervalSet
        The time support of the time series

    Examples
    --------
    Initialize a Tsd:

    >>> import pynapple as nap
    >>> import numpy as np
    >>> t = np.arange(100)
    >>> d = np.ones(100)
    >>> tsd = nap.Tsd(t=t, d=d)
    >>> tsd
    Time (s)
    ----------  --
    0.0          1
    1.0          1
    2.0          1
    3.0          1
    4.0          1
    5.0          1
    6.0          1
    ...
    93.0         1
    94.0         1
    95.0         1
    96.0         1
    97.0         1
    98.0         1
    99.0         1
    dtype: float64, shape: (100,)

    Initialize a Tsd with `time_support`:

    >>> t = np.arange(100)
    >>> d = np.ones(100)
    >>> time_support = nap.IntervalSet(start=0.5, end=8)
    >>> tsd = nap.Tsd(t=t, d=d, time_support=time_support)
    >>> tsd
    Time (s)
    ----------  --
    1            1
    2            1
    3            1
    4            1
    5            1
    6            1
    7            1
    8            1
    dtype: float64, shape: (8,)

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
            Default is True. Does not apply if `d` is already a numpy array or a numpy memory map.
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

    def __setitem__(self, key, value):
        if isinstance(key, Tsd):
            try:
                assert np.issubdtype(key.dtype, np.bool_)
            except AssertionError:
                raise ValueError(
                    "When indexing with a Tsd, it must contain boolean values"
                )
            key = key.d

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
        if isinstance(key, Tsd):
            try:
                assert np.issubdtype(key.dtype, np.bool_)
            except AssertionError:
                raise ValueError(
                    "When indexing with a Tsd, it must contain boolean values"
                )
            key = key.d

        output = self.values.__getitem__(key)

        if isinstance(key, tuple):
            index = self.index.__getitem__(key[0])
        elif isinstance(key, Number):
            index = np.array([key])
        else:
            index = self.index.__getitem__(key)
        return _initialize_tsd_output(self, output, time_index=index, kwargs=kwargs)

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

    @add_base_docstring("time_diff")
    def time_diff(self, align="center", epochs=None):
        """
        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tsd = nap.Tsd(t=[1, 3, 5, 6, 8, 12], d=[2, 2, 2, 3, 4, 5])
        >>> epochs = nap.IntervalSet(start=2, end=9, time_units='s')
        >>> tsd_time_diffs = tsd.time_diff(align="center", epochs=epochs)
        >>> tsd_time_diffs
        Time (s)
        ----------  --
        4            2
        5.5          1
        7            2
        dtype: float64, shape: (3,)
        """
        return _Base.time_diff(self, align, epochs)

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


class Ts(_Base):
    """
    Timestamps only object for a time series with only time index.

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

    def _define_instance(self, time_index, time_support, values=None, **kwargs):
        """
        Define a new class instance.

        Optional parameters for initialization are either passed to the function or are grabbed from self.
        """
        if values is None:
            return self.__class__(t=time_index, time_support=time_support)
        else:
            return _initialize_tsd_output(
                self,
                values,
                time_index=time_index,
                time_support=time_support,
                kwargs=kwargs,
            )

    def __repr__(self):
        upper = "Time (s)"
        rows = _get_terminal_size()[1]
        max_rows = np.maximum(rows - 10, 2)

        if len(self) > max_rows:
            n_rows = max_rows // 2
            _str_ = "\n".join(
                [str(i) for i in self.index[0:n_rows]]
                + ["..."]
                + [str(i) for i in self.index[-n_rows:]]
            )
        else:
            _str_ = "\n".join([str(i) for i in self.index])

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

    def trial_count(
        self, ep, bin_size, align="start", padding_value=np.nan, time_unit="s"
    ):
        """
        Return trial-based count tensor from an IntervalSet object. The shape of the tensor array is
        (number of trials, number of time bins).

        The `bin_size` parameter determines the number of time bins.

        The `align` parameter controls how the time series are aligned. If `align="start"`, the time
        series are aligned to the start of each trial. If `align="end"`, the time series are aligned
        to the end of each trial.

        If trials have uneven durations, the returned array is padded. The parameter `padding_value`
        determines which value is used to pad the array. Default is NaN.

        Parameters
        ----------
        ep : IntervalSet
            Epochs holding the trials. Each interval can be of unequal size.
        bin_size : Number
            The size of the time bins.
        align: str, optional
            How to align the time series ('start' [default], 'end')
        padding_value: Number, optional
            How to pad the array if unequal intervals. Default is np.nan.
        time_unit : str, optional
            Time units of the bin_size parameter ('s' [default], 'ms', 'us').

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        RuntimeError
            If `time_unit` not in ["s", "ms", "us"]
        """
        if not isinstance(ep, IntervalSet):
            raise RuntimeError("Argument ep should be of type IntervalSet")
        if time_unit not in ["s", "ms", "us"]:
            raise RuntimeError("time_unit should be 's', 'ms' or 'us'")
        if align not in ["start", "end"]:
            raise RuntimeError("align should be 'start' or 'end'")
        if not isinstance(bin_size, Number):
            raise RuntimeError("bin_size should be of type int or float")

        # Determine size of tensor
        bin_size = float(TsIndex.format_timestamps(np.array([bin_size]), time_unit)[0])
        n_t = int(np.max(np.ceil((ep.end + bin_size - ep.start) / bin_size)))
        count = self.count(bin_size=bin_size, ep=ep)

        output = np.ones(shape=(len(ep), n_t)) * padding_value
        n_ep = np.zeros(len(ep), dtype="int")  # To trim to the minimum length

        if align == "start":
            for i in range(len(ep)):
                tmp = count.get(ep.start[i], ep.end[i]).values
                n_ep[i] = tmp.shape[0]
                output[i, 0 : tmp.shape[0]] = np.transpose(tmp)
            output = output[:, 0 : np.max(n_ep)]

        if align == "end":
            for i in range(len(ep)):
                tmp = count.get(ep.start[i], ep.end[i]).values
                n_ep[i] = tmp.shape[0]
                output[i, -tmp.shape[0] :] = np.transpose(tmp)
            output = output[:, -np.max(n_ep) :]

        return output

    @add_base_docstring("time_diff")
    def time_diff(self, align="center", epochs=None):
        """
        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> ts = nap.Ts(t=[1, 3, 5, 6, 8, 12])
        >>> epochs = nap.IntervalSet(start=2, end=9, time_units='s')
        >>> tsd_time_diffs = ts.time_diff(align="center", epochs=epochs)
        >>> tsd_time_diffs
        Time (s)
        ----------  --
        4            2
        5.5          1
        7            2
        dtype: float64, shape: (3,)
        """
        return _Base.time_diff(self, align, epochs)
