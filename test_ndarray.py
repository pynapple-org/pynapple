# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2023-08-18 17:38:35
# @Last Modified by:   gviejo
# @Last Modified time: 2023-09-16 14:04:18
import numpy as np
import pandas as pd
import pynapple as nap

tsdtensor = nap.TsdTensor(t=np.sort(np.random.uniform(0, 100, 100)), d=np.random.rand(100,10,20,5))

tsdframe = nap.TsdFrame(t = np.arange(10), d = np.random.rand(10, 3), columns=['a', 'b', 'c'])

tsd = nap.Tsd(t=np.arange(10), d=np.random.rand(10))

ts = nap.Ts(t=np.arange(10))

ep = nap.IntervalSet(start=[0,50], end=[20,60])

