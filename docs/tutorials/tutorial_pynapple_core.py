# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-26 21:06:38
# @Last Modified by:   gviejo
# @Last Modified time: 2022-01-26 21:13:18

# # Core Tutorial
# 
# This script will introduce you to the basics of time series handling with pynapple.

import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap


# Time series object
#  
# Let's create a Tsd object with fake data. 
# In this case, every time point is 1 second apart. 
# A Tsd object is a wrapper of pandas series.

tsd = nap.Tsd(t = np.arange(100), d = np.random.rand(100), time_units = 's')

print(tsd)


# While the tsd object appears in second, 
# it actually holds the values in microseconds by default. 
# It is possible to switch between seconds, milliseconds and microseconds. 
# Note that when using *as_units*, the returned object is a simple pandas series.

print(tsd.as_units('ms'))
print(tsd.as_units('us'))


# If only timestamps are available, for example spike times, 
# we can construct a Ts object which holds only times. 
# In this case, we generate 10 random spike times between 0 and 100 ms.

ts = nap.Ts(t = np.sort(np.random.uniform(0, 100, 10)), time_units = 'ms')
print(ts)

# If the time series contains multiple columns, we can use a TsdFrame.

tsdframe = nap.TsdFrame(t = np.arange(100), d = np.random.rand(100,3), time_units = 's', columns = ['a', 'b', 'c'])
print(tsdframe)


## Interval Sets object
# The IntervalSet object stores multiple epochs with a common time units. 
# It can then be used to restrict time series to this particular set of epochs.

epochs = nap.IntervalSet(start = [0, 10], end = [5, 15], time_units = 's')

new_tsd = tsd.restrict(epochs)

print(epochs)
print('\n')
print(new_tsd)


# Multiple operations are available for IntervalSet. 
# For example, IntervalSet can be merged. 
# See the full documentation of the class at 
# https://pynapple-org.github.io/pynapple/core.interval_set/#pynapple.core.interval_set.IntervalSet.intersect 
# for a list of all the functions that can be used to manipulate IntervalSets.


epoch1 = nap.IntervalSet(start=[0], end=[10]) # no time units passed. Default is us.
epoch2 = nap.IntervalSet(start=[5,30],end=[20,45])

epoch = epoch1.union(epoch2)
print(epoch1, '\n')
print(epoch2, '\n')
print(epoch)


# ## TsGroup 
# Multiple time series with different time stamps 
# (.i.e. a group of neurons with different spike times from one session) 
# can be grouped with the TsGroup object. 
# The TsGroup behaves like a dictionary but it is also possible to slice with a list of indexes


my_ts = {0:nap.Ts(t = np.sort(np.random.uniform(0, 100, 1000)), time_units = 's'), # here a simple dictionary
         1:nap.Ts(t = np.sort(np.random.uniform(0, 100, 2000)), time_units = 's'),
         2:nap.Ts(t = np.sort(np.random.uniform(0, 100, 3000)), time_units = 's')}

tsgroup = nap.TsGroup(my_ts)

print(tsgroup, '\n')
print(tsgroup[0], '\n') # dictionary like indexing returns directly the Ts object
print(tsgroup[[0,2]]) # list like indexing

# Operations such as restrict can thus be directly applied to the TsGroup as well as other operations.

newtsgroup = tsgroup.restrict(epochs)
count = tsgroup.count(1, epochs, time_units='s') # Here counting the elements within bins of 1 seconds
print(count)


# One advantage of grouping time series is that metainformation can be added about each elements. 
# In this case, we add labels to each Ts object when instantiating the group and after. 
# We can then use this label to split the group. 
# See the documentation about TsGroup at 
# https://pynapple-org.github.io/pynapple/core.ts_group/ 
# for all the ways to split TsGroup.

tsgroup = nap.TsGroup(my_ts, time_units = 's', label1=[0,1,0])
tsgroup.set_info(label1=np.array(['a', 'a', 'b']))

print(tsgroup, '\n')

newtsgroup= tsgroup.getby_category('label1')
print(newtsgroup['a'], '\n')
print(newtsgroup['b'])


# ## Time support
# A key element of the manipulation of time series by pynapple is the inherent time support defined for Ts, Tsd, TsdFrame and TsGroup objects. 
# The time support is defined as an IntervalSet that provides the time series with a context. 
# For example,, the restrict operation will update automatically the time support to the new time series. 
# Ideally the time support should be defined for all time series when instantiating them. 
# If no time series is given, the time support is inferred from the start and end of the time series. 
# In this example, a TsGroup is instantiated with and without a time support. Notice how the frequency of each Ts element is changed when the time support is defined explicitly.

time_support = nap.IntervalSet(start = 0, end = 100, time_units = 's')

my_ts = {0:nap.Ts(t = np.sort(np.random.uniform(0, 100, 10)), time_units = 's'), # here a simple dictionary
         1:nap.Ts(t = np.sort(np.random.uniform(0, 100, 20)), time_units = 's'),
         2:nap.Ts(t = np.sort(np.random.uniform(0, 100, 30)), time_units = 's')}

tsgroup = nap.TsGroup(my_ts)

tsgroup_with_time_support = nap.TsGroup(my_ts, time_support = time_support)

print(tsgroup, '\n')

print(tsgroup_with_time_support, '\n')

print(tsgroup_with_time_support.time_support) # acceding the time support

