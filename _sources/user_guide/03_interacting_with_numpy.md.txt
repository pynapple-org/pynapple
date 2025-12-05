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

Numpy tutorial
=======================

This tutorial shows how pynapple interact with numpy.

```{code-cell} ipython3
:tags: [hide-cell]
import numpy as np
import pynapple as nap
import pandas as pd
```

Multiple time series object are avaible depending on the shape of the data.

- `TsdTensor` : for data with of more than 2 dimensions, typically movies.
- `TsdFrame` : for column-based data. It can be easily converted to a pandas.DataFrame. Columns can be labelled and selected similar to pandas.
- `Tsd` : one-dimensional time series. It can be converted to a pandas.Series.
- `Ts` : For timestamps data only.


+++

Initialization
--------------


```{code-cell} ipython3
tsdtensor = nap.TsdTensor(t=np.arange(100), d=np.random.rand(100, 5, 5), time_units="s")
tsdframe = nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 3), columns = ['a', 'b', 'c'])
tsd = nap.Tsd(t=np.arange(100), d=np.random.rand(100))
ts = nap.Ts(t=np.arange(100))

print(tsdtensor)
```

`Tsd` and `Ts` can be converted to a `pandas.Series`.


```{code-cell} ipython3
print(tsd.as_series())
```

`TsdFrame` to a `pandas.DataFrame`.


```{code-cell} ipython3
print(tsdframe.as_dataframe())
```

Attributes
----------
The numpy array is accesible with the attributes `.values`, `.d` and functions [`as_array()`](pynapple.Tsd.as_array), [`to_numpy()`](pynapple.Tsd.to_numpy).
The time index array is a `TsIndex` object accessible with `.index` or `.t`.
`.shape` and `.ndim` are also accessible.


```{code-cell} ipython3
print(tsdtensor.ndim)
print(tsdframe.shape)
print(len(tsd))
```

Slicing
-------
Slicing is very similar to numpy array. The first dimension is always time and time support is always passed on if a pynapple object is returned.

First 10 elements. Return a `TsdTensor`


```{code-cell} ipython3
print(tsdtensor[0:10])
```

First column. Return a `Tsd`


```{code-cell} ipython3
print(tsdframe[:,0])
```

First element. Return a numpy ndarray


```{code-cell} ipython3
print(tsdtensor[0])
```

The time support is never changing when slicing time down.


```{code-cell} ipython3
print(tsd.time_support)
print(tsd[0:20].time_support)
```

`TsdFrame` offers special slicing similar to `pandas.DataFrame`.

Only `TsdFrame` can have columns labelling and indexing.


```{code-cell} ipython3
print(tsdframe.loc['a'])
print(tsdframe.loc[['a', 'c']])
```

Arithmetic
----------
Arithmetical operations works similar to numpy


```{code-cell} ipython3
tsd = nap.Tsd(t=np.arange(5), d=np.ones(5))
print(tsd + 1)
```

It is possible to do array operations on the time series provided that the dimensions matches.
The output will still be a time series object.


```{code-cell} ipython3
print(tsd - np.ones(5))
```

Nevertheless operations like this are not permitted :


```{code-cell} ipython3
try:
	tsd + tsd
except Exception as error:
	print(error)
```

Array operations
----------------
The most common numpy functions will return a time series if the output first dimension matches the shape of the time index.

Here the `TsdTensor` is averaged along the time axis. The output is a numpy array.


```{code-cell} ipython3
print(np.mean(tsdtensor, 0))
```

Here averaging across the second dimension returns a `TsdFrame`.


```{code-cell} ipython3
print(np.mean(tsdtensor, 1))
```

This is not true for FFT functions though.


```{code-cell} ipython3
try:
	np.fft.fft(tsd)
except Exception as error:
	print(error)
```

Concatenating
-------------
It is possible to concatenate time series providing than they don't overlap meaning time indexe should be already sorted through all time series to concatenate


```{code-cell} ipython3
tsd1 = nap.Tsd(t=np.arange(5), d=np.ones(5))
tsd2 = nap.Tsd(t=np.arange(5)+10, d=np.ones(5)*2)
tsd3 = nap.Tsd(t=np.arange(5)+20, d=np.ones(5)*3)

print(np.concatenate((tsd1, tsd2, tsd3)))
```

It's also possible to concatenate vertically if time indexes matches up to pynapple float precision


```{code-cell} ipython3
tsdframe = nap.TsdFrame(t=np.arange(5), d=np.random.randn(5, 3))

print(np.concatenate((tsdframe, tsdframe), 1))
```

Spliting
--------
Array split functions are also implemented


```{code-cell} ipython3
print(np.array_split(tsdtensor[0:10], 2))
```

Modifying
---------
It is possible to modify a time series element wise


```{code-cell} ipython3
print(tsd1)

tsd1[0] = np.pi

print(tsd1)
```

It is also possible to modify a time series with logical operations


```{code-cell} ipython3
tsd[tsd.values>0.5] = 0.0

print(tsd)
```

Sorting
---------
It is not possible to sort along the first dimension as it would break the sorting of the time index


```{code-cell} ipython3
tsd = nap.Tsd(t=np.arange(100), d=np.random.rand(100))

try:
	np.sort(tsd)
except Exception as error:
	print(error)
```
