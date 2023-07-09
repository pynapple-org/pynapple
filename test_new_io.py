# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-15 15:37:03
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-07-08 17:51:46

import numpy as np
import pynapple as nap

path = "/Users/gviejo/Dropbox/MyProject"

project = nap.load_folder(path)

data = project['sub-A001']['ses-01']['pynapple']

# project = nap.load_project(path)

# print(project)

# print(project['sub-A001'])

# print(project['sub-A001']['ses-01'])

# print(project['sub-A001']['ses-01']['pynapple']['spikes'])


# session = project['sub-A001']['ses-01']

# epoch = nap.IntervalSet(start = np.array([0, 3]), end = np.array([1, 6]))

# session.save(epoch, "stimulus-fish", "Fish pictures to V1")

# print(session)

# print(session['stimulus-fish'])

# print(session.doc('stimulus-fish'))