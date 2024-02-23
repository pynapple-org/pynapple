# coding: utf-8
"""
Core Tutorial
============

This script will introduce the basics of handling time series data with pynapple.

"""
# %%
# !!! warning
#     This tutorial uses seaborn and matplotlib for displaying the figure.
#
#     You can install both with `pip install matplotlib seaborn`

import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
import pandas as pd
import seaborn as sns

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)

# %%
# ***
# Time series object
# ------------------
#
# Let's create a Tsd object with artificial data. In this example, every time point is 1 second apart.


tsd = nap.Tsd(t=np.arange(100), d=np.random.rand(100), time_units="s")

print(tsd)

# %%
# It is possible to toggle between seconds, milliseconds and microseconds. Note that when using *as_units*, the returned object is a simple pandas series.

print(tsd.as_units("ms"), "\n")
print(tsd.as_units("us"))

# %%
# Pynapple is able to handle data that only contains timestamps, such as an object containing only spike times. To do so, we construct a Ts object which holds only times. In this case, we generate 10 random spike times between 0 and 100 ms.

ts = nap.Ts(t=np.sort(np.random.uniform(0, 100, 10)), time_units="ms")

print(ts)

# %%
# If the time series contains multiple columns, we use a TsdFrame.

tsdframe = nap.TsdFrame(
    t=np.arange(100), d=np.random.rand(100, 3), time_units="s", columns=["a", "b", "c"]
)

print(tsdframe)

# %%
# And if the number of dimension is even larger, we can use the TsdTensor (typically movies).

tsdframe = nap.TsdTensor(
    t=np.arange(100), d=np.random.rand(100, 3, 4)
)

print(tsdframe)


# %%
# ***
# Interval Sets object
# --------------------
#
# The [IntervalSet](https://peyrachelab.github.io/pynapple/core.interval_set/) object stores multiple epochs with a common time unit. It can then be used to restrict time series to this particular set of epochs.


epochs = nap.IntervalSet(start=[0, 10], end=[5, 15], time_units="s")

new_tsd = tsd.restrict(epochs)

print(epochs)
print("\n")
print(new_tsd)

# %%
# Multiple operations are available for IntervalSet. For example, IntervalSet can be merged. See the full documentation of the class [here](https://peyrachelab.github.io/pynapple/core.interval_set/#pynapple.core.interval_set.IntervalSet.intersect) for a list of all the functions that can be used to manipulate IntervalSets.


epoch1 = nap.IntervalSet(start=0, end=10)  # no time units passed. Default is us.
epoch2 = nap.IntervalSet(start=[5, 30], end=[20, 45])

epoch = epoch1.union(epoch2)
print(epoch1, "\n")
print(epoch2, "\n")
print(epoch)

# %%
# ***
# TsGroup object
# --------------
#
# Multiple time series with different time stamps (.i.e. a group of neurons with different spike times from one session) can be grouped with the TsGroup object. The TsGroup behaves like a dictionary but it is also possible to slice with a list of indexes

my_ts = {
    0: nap.Ts(
        t=np.sort(np.random.uniform(0, 100, 1000)), time_units="s"
    ),  # here a simple dictionary
    1: nap.Ts(t=np.sort(np.random.uniform(0, 100, 2000)), time_units="s"),
    2: nap.Ts(t=np.sort(np.random.uniform(0, 100, 3000)), time_units="s"),
}

tsgroup = nap.TsGroup(my_ts)

print(tsgroup, "\n")
print(tsgroup[0], "\n")  # dictionary like indexing returns directly the Ts object
print(tsgroup[[0, 2]])  # list like indexing

# %%
# Operations such as restrict can thus be directly applied to the TsGroup as well as other operations.

newtsgroup = tsgroup.restrict(epochs)

count = tsgroup.count(
    1, epochs, time_units="s"
)  # Here counting the elements within bins of 1 seconds

print(count)

# %%
# One advantage of grouping time series is that metainformation can be appended directly on an element-wise basis. In this case, we add labels to each Ts object when instantiating the group and after. We can then use this label to split the group. See the [TsGroup](https://peyrachelab.github.io/pynapple/core.ts_group/) documentation for a complete methodology for splitting TsGroup objects.


label1 = pd.Series(index=list(my_ts.keys()), data=[0, 1, 0])

tsgroup = nap.TsGroup(my_ts, time_units="s", label1=label1)
tsgroup.set_info(label2=np.array(["a", "a", "b"]))

print(tsgroup, "\n")

newtsgroup = tsgroup.getby_category("label1")
print(newtsgroup[0], "\n")
print(newtsgroup[1])


# %%
# ***
# Time support
# ------------
#
# A key feature of how pynapple manipulates time series is an inherent time support object defined for Ts, Tsd, TsdFrame and TsGroup objects. The time support object is defined as an IntervalSet that provides the time serie with a context. For example, the restrict operation will automatically update the time support object for the new time series. Ideally, the time support object should be defined for all time series when instantiating them. If no time series is given, the time support is inferred from the start and end of the time series.
#
# In this example, a TsGroup is instantiated with and without a time support. Notice how the frequency of each Ts element is changed when the time support is defined explicitly.


time_support = nap.IntervalSet(start=0, end=200, time_units="s")

my_ts = {
    0: nap.Ts(
        t=np.sort(np.random.uniform(0, 100, 10)), time_units="s"
    ),  # here a simple dictionnary
    1: nap.Ts(t=np.sort(np.random.uniform(0, 100, 20)), time_units="s"),
    2: nap.Ts(t=np.sort(np.random.uniform(0, 100, 30)), time_units="s"),
}

tsgroup = nap.TsGroup(my_ts)

tsgroup_with_time_support = nap.TsGroup(my_ts, time_support=time_support)

print(tsgroup, "\n")

print(tsgroup_with_time_support, "\n")

print(tsgroup_with_time_support.time_support)  # acceding the time support

# %%
# We can use value_from which as it indicates assign to every timestamps the closed value in time from another time series.
# Let's define the time series we want to assign values from.

tsd_sin = nap.Tsd(t=np.arange(0, 100, 1), d=np.sin(np.arange(0, 10, 0.1)))

tsgroup_sin = tsgroup.value_from(tsd_sin)

plt.figure(figsize=(12, 6))
plt.plot(tsgroup[0].fillna(0), "|", markersize=20, mew=3)
plt.plot(tsd_sin, linewidth=2)
plt.plot(tsgroup_sin[0], "o", markersize=20)
plt.title("ts.value_from(tsd)")
plt.xlabel("Time (s)")
plt.yticks([-1, 0, 1])
plt.show()
