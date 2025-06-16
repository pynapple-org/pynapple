---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Introduction to pynapple

The goal of this tutorial is to quickly learn enough about pynapple to get started with data analysis. This tutorial assumes familiarity with the basics functionalities of numpy. 

You can check how to install pynapple [here](../installing.md).

:::{important}
By default, pynapple will assume a time units in seconds when passing timestamps array or time parameters such as bin size (unless specified with the `time_units` argument)
:::

***
Importing pynapple
------------------

The convention is to import pynapple with a namespace:

```{code-cell} ipython3
import pynapple as nap
```


```{code-cell} ipython3
:tags: [hide-cell]

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
```

***
Instantiating pynapple objects
------------------------------

### [`nap.Tsd`](pynapple.Tsd): 1-dimensional time series

If you have a 1-dimensional time series, you use the [`nap.Tsd`](pynapple.Tsd) object. The arguments `t` and `d` are the arguments for timestamps and data.


```{code-cell} ipython3
tsd = nap.Tsd(t=np.arange(100), d=np.random.rand(100))

print(tsd)
```


### [`nap.TsdFrame`](pynapple.TsdFrame): 2-dimensional time series

If you have a 2-dimensional time series, you use the [`nap.TsdFrame`](pynapple.TsdFrame) object. The arguments `t` and `d` are the arguments for timestamps and data. You can add the argument `columns` to label each columns.


```{code-cell} ipython3
tsdframe = nap.TsdFrame(
    t=np.arange(100), d=np.random.rand(100, 3), columns=["a", "b", "c"]
)

print(tsdframe)
```

### [`nap.TsdTensor`](pynapple.TsdTensor): n-dimensional time series

If you have larger than 2 dimensions time series (typically movies), you use the [`nap.TsdTensor`](pynapple.TsdTensor) object . The arguments `t` and `d` are the arguments for timestamps and data.


```{code-cell} ipython3
tsdtensor = nap.TsdTensor(
    t=np.arange(100), d=np.random.rand(100, 3, 4)
)

print(tsdtensor)
```

### [`nap.IntervalSet`](pynapple.IntervalSet): intervals

The [`IntervalSet`](pynapple.IntervalSet) object stores multiple epochs with a common time unit in a table format. The epochs are strictly _non-overlapping_. Both `start` and `end` arguments are necessary.


```{code-cell} ipython3
epochs = nap.IntervalSet(start=[0, 10], end=[5, 15])

print(epochs)

```

### [`nap.Ts`](pynapple.Ts): timestamps

The [`Ts`](pynapple.Ts) object stores timestamps data (typically spike times or reward times). The argument `t` for timestamps is necessary.


```{code-cell} ipython3
ts = nap.Ts(t=np.sort(np.random.uniform(0, 100, 10)))

print(ts)
```

### [`nap.TsGroup`](pynapple.TsGroup): group of timestamps


[`TsGroup`](pynapple.TsGroup) is a dictionnary that stores multiple time series with different time stamps (.i.e. a group of neurons with different spike times from one session). The first argument `data` can be a dictionnary of `Ts`, `Tsd` or numpy 1d array.


```{code-cell} ipython3
data = {
    0: nap.Ts(t=np.sort(np.random.uniform(0, 100, 1000))),
    1: nap.Ts(t=np.sort(np.random.uniform(0, 100, 2000))),
    2: nap.Ts(t=np.sort(np.random.uniform(0, 100, 3000))),
}

tsgroup = nap.TsGroup(data)

print(tsgroup, "\n")
```


***
Interaction between pynapple objects
------------------------------------

### Time support : attribute of time series


A key feature of how pynapple manipulates time series is an inherent time support object defined for `Ts`, `Tsd`, `TsdFrame` and `TsGroup` objects. The time support object is defined as an `IntervalSet` that provides the time serie with a context. For example, the restrict operation will automatically update the time support object for the new time series. Ideally, the time support object should be defined for all time series when instantiating them. If no time series is given, the time support is inferred from the start and end of the time series.

In this example, a `Tsd` is instantiated with and without a time support of intervals 0 to 5 seconds. Notice how the shape of the `Tsd` varies.


```{code-cell} ipython3
time_support = nap.IntervalSet(start=0, end=2)

print(time_support)
```

Without time support :

```{code-cell} ipython3

print(nap.Tsd(t=[0, 1, 2, 3, 4], d=[0, 1, 2, 3, 4]))

```
With time support :

```{code-cell} ipython3

print(
    nap.Tsd(
        t=[0, 1, 2, 3, 4], d=[0, 1, 2, 3, 4], 
        time_support = time_support
        )
    )

```

The time support object has become an attribute of the time series. Depending on the operation applied to the time series, it will be updated. 

```{code-cell} ipython3

tsd = nap.Tsd(
    t=np.arange(10), d=np.random.randn(10), 
    time_support = time_support
    )

print(tsd.time_support)
```



### Restricting time series to epochs

The central function of pynapple is the [`restrict`](pynapple.Tsd.restrict) method of `Ts`, `Tsd`, `TsdFrame` and `TsGroup`. The argument is an `IntervalSet` object. Only time points within the intervals of the `IntervalSet` object are returned. The time support of the time series is updated accordingly.


```{code-cell} ipython3
tsd = nap.Tsd(t=np.arange(10), d=np.random.randn(10))

ep = nap.IntervalSet(start=[0, 7], end=[3.5, 12.4])

print(ep)
```

From :

```{code-cell} ipython3
print(tsd)
```

to :

```{code-cell} ipython3
new_tsd = tsd.restrict(ep)

print(new_tsd)

```


***
Numpy & pynapple
----------------

Pynapple relies on numpy to store the data. Pynapple objects behave very similarly to numpy and numpy functions can be applied directly

```{code-cell} ipython3
tsdtensor = nap.TsdTensor(t=np.arange(100), d=np.random.rand(100, 3, 4))
```

If a numpy function preserves the time axis, a pynapple object is returned. 

In this example, averaging a `TsdTensor` along the second dimension returns a `TsdFrame`:

```{code-cell} ipython3
print(
    np.mean(tsdtensor, 1)
    )
```

Averaging along the time axis will return a numpy array object:

```{code-cell} ipython3
print(
    np.mean(tsdtensor, 0)
    )
```

***
Slicing objects
---------------

### Slicing time series and intervals

#### Like numpy array

`Ts`, `Tsd`, `TsdFrame`, `TsdTensor` and `IntervalSet` can be sliced similar to numpy array:

```{code-cell} ipython3
tsdframe = nap.TsdFrame(t=np.arange(10)/10, d=np.random.randn(10,4))
print(tsdframe)
```

```{code-cell} ipython3
print(tsdframe[4:7])
```

```{code-cell} ipython3
print(tsdframe[:,0])
```

```{code-cell} ipython3
ep = nap.IntervalSet(start=[0, 10, 20], end=[4, 15, 25])
print(ep)
```

```{code-cell} ipython3
print(ep[0:2])
```

```{code-cell} ipython3
print(ep[1])
```

#### Like pandas DataFrame

:::{important}
This [page](03_core_methods.md#special-slicing-tsdframe) references all the way to slice `TsdFrame`
:::


`TsdFrame` can be sliced like pandas DataFrame when the columns have been labelled with strings :

```{code-cell} ipython3
tsdframe = nap.TsdFrame(t=np.arange(10), d=np.random.randn(10,3), columns=['a', 'b', 'c'])
print(tsdframe['a'])
```
but integer-indexing only works like numpy if a list of integers is used to label columns :

```{code-cell} ipython3
tsdframe = nap.TsdFrame(t=np.arange(4), d=np.random.randn(4,3), columns=[3, 2, 1])
print(tsdframe, "\n")
print(tsdframe[3])
```

The [`loc`](pynapple.TsdFrame.loc) method can be used to slice column-based only:

```
print(tsdframe.loc[3])
```

### Slicing TsGroup

`TsGroup` object can be indexed to return directly the timestamp object or sliced to return a new `TsGroup`. 

Indexing:

```{code-cell} ipython3
print(tsgroup[0], "\n")
```

Slicing:


```{code-cell} ipython3
print(tsgroup[[0, 2]])
```

***
Core functions
--------------

Objects have methods that can help transform and refine time series. This is a non exhaustive list.

### Binning: counting events

Time series objects have the [`count`](pynapple.Tsd.count) method that count the number of timestamps. This is typically used when counting number of spikes within a particular bin over multiple intervals. The returned object is a `Tsd` or `TsdFrame` with the timestamps being the center of the bins.


```{code-cell} ipython3
count = tsgroup.count(1)

print(count)
```

### Thresholding

Some time series have specific methods. The [`threshold`](pynapple.Tsd.threshold) method of `Tsd` returns a new `Tsd` with all the data above or below a given value.

```{code-cell} ipython3
tsd = nap.Tsd(t=np.arange(10), d=np.random.rand(10))

print(tsd)

print(tsd.threshold(0.5))
```

An important aspect of the tresholding is that the time support get updated based on the time points remaining. To get the epochs above/below a certain threshold, you can access the time support of the returned object.

```{code-cell} ipython3
print(tsd.time_support)

print(tsd.threshold(0.5, "below").time_support)

```


### Time-bin averaging of data

Many analyses requires to bring time series to the same rates and same dimensions. A quick way to downsample a time series to match in size for example a count array is to bin average. The [`bin_average`](pynapple.TsdFrame.bin_average) method takes a bin size in unit of time.

```{code-cell} ipython3
tsdframe = nap.TsdFrame(t=np.arange(0, 100)/10, d=np.random.randn(100,3))

print(tsdframe)

```

Here we go from a timepoint every 100ms to a timepoint every second.


```{code-cell} ipython3
print(tsdframe.bin_average(1))

```

***
Loading data
------------

See [here](02_input_output.md) for more details about loading data.

### Loading NWB

Pynapple supports by default the [NWB standard](https://pynwb.readthedocs.io/en/latest/). 

NWB files can be loaded with :

```
nwb = nap.load_file("path/to/my.nwb")
```

or directly with the `NWBFile` class:

```
nwb = nap.NWBFile("path/to/my.nwb")

print(nwb)
```
```
my.nwb
┍━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━┑
│ Keys            │ Type        │
┝━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━┥
│ units           │ TsGroup     │
┕━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━┙
```

The returned object behaves like a dictionnary. The first column indicates the keys. The second column indicate the object type.

```
print(nwb['units'])
```
```
  Index    rate  location      group
-------  ------  ----------  -------
      0    1.0  brain        0
      1    1.0  brain        0
      2    1.0  brain        0
```


***
Overview of advanced analysis
-----------------------------

The `process` module of pynapple contains submodules that group methods that can be applied for high level analysis. All of the method are directly available from the `nap` namespace.

:::{important}
Some functions have been doubled given the nature of the data. For instance, computing a 1d tuning curves from spiking activity requires the [`nap.compute_1d_tuning_curves`](pynapple.process.tuning_curves.compute_1d_tuning_curves). The same function for calcium imaging data which is a continuous time series is available with [`nap.compute_1d_tuning_curves_continuous`](pynapple.process.tuning_curves.compute_1d_tuning_curves_continuous). 
:::

**[Discrete correlograms & ISI](05_correlograms_isi)**

This module analyses discrete events, specifically correlograms (for example by computing the cross-correlograms of a population of neurons) and interspike interval (ISI) distributions.

**[Bayesian decoding](07_decoding)**

The decoding module perfoms bayesian decoding given a set of tuning curves and a `TsGroup`.

**[Filtering](12_filtering)**

Bandpass, lowpass, highpass or bandstop filtering can be done to any time series using either Butterworth filter or windowed-sinc convolution.

**[Perievent time histogram](08_perievent)**

The perievent module has a set of functions to center time series and timestamps data around a particular events.

**[Randomizing](09_randomization)**

The randomize module holds multiple technique to shuffle timestamps in order to create surrogate datasets.

**[Spectrum](10_power_spectral_density)**

The spectrum module contains the methods to return the (mean) power spectral density of a time series.

**[Tuning curves](06_tuning_curves)**

Tuning curves of neurons based on spiking or calcium activity can be computed.

**[Wavelets](11_wavelets)**

The wavelets module performs Morlet wavelets decomposition of a time series. 

**[Warping](13_warping)**

This module provides methods for building trial-based tensors and time-warped trial-based tensors.
