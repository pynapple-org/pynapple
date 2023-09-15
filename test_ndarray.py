# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2023-08-18 17:38:35
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-09-15 18:17:09
import numpy as np
import pandas as pd
import pynapple as nap

tsdtensor = nap.TsdTensor(t=np.arange(10), d=np.random.rand(10,3,4,5))

tsdframe = nap.TsdFrame(t = np.arange(10), d = np.random.rand(10, 3), columns=['a', 'b', 'c'])

tsd = nap.Tsd(t=np.arange(10), d=np.random.rand(10))

ts = nap.Ts(t=np.arange(10))


