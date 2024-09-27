# coding: utf-8
"""
# NWB & Lazy-loading

Pynapple currently provides loaders for two data formats :

 - `npz` with a special structure. You can check this [notebook](../tutorial_pynapple_io) for a descrition of the methods for saving/loading `npz` files.

 - [NWB format](https://pynwb.readthedocs.io/en/stable/index.html#)

This notebook focuses on the NWB format. Additionally it demonstrates the capabilities of pynapple for lazy-loading different formats.

"""
# %%
# Let's import libraries.

import numpy as np
import pynapple as nap
import os
import requests, math
import tqdm
import zipfile

# %%
# Here we download the data.

project_path = "MyProject"

if project_path not in os.listdir("."):
  r = requests.get(f"https://osf.io/a9n6r/download", stream=True)
  block_size = 1024*1024
  with open(project_path+".zip", 'wb') as f:
    for data in tqdm.tqdm(r.iter_content(block_size), unit='MB', unit_scale=True,
      total=math.ceil(int(r.headers.get('content-length', 0))//block_size)):
      f.write(data)

  with zipfile.ZipFile(project_path+".zip", 'r') as zip_ref:
    zip_ref.extractall(".")

# %%
# NWB
# --------------
# When loading a NWB file, pynapple will walk through it and test the compatibility of each data structure with a pynapple objects. If the data structure is incompatible, pynapple will ignore it. The class that deals with reading NWB file is [`nap.NWBFile`](../../../reference/io/interface_nwb/). You can pass the path to a NWB file or directly an opened NWB file. Alternatively you can use the function [`nap.load_file`](../../../reference/io/misc/#pynapple.io.misc.load_file).
#
# 
# !!! note
# 	Creating the NWB file is outside the scope of pynapple. The NWB file used here has already been created before.
# 	Multiple tools exists to create NWB file automatically. You can check [neuroconv](https://neuroconv.readthedocs.io/en/main/), [NWBGuide](https://nwb-guide.readthedocs.io/en/latest/) or even [NWBmatic](https://github.com/pynapple-org/nwbmatic).


data = nap.load_file("MyProject/sub-A2929/A2929-200711/pynapplenwb/A2929-200711.nwb")

print(data)

# %%
# Pynapple will give you a table with all the entries of the NWB file that are compatible with a pynapple object.
# When parsing the NWB file, nothing is loaded. The `NWBFile` keeps track of the position of the data within the NWB file with a key. You can see it with the attributes `key_to_id`.

print(data.key_to_id)


# %%
# Loading an entry will get pynapple to read the data.

z = data['z']

print(data['z'])

# %%
#
# Internally, the `NWBClass` has replaced the pointer to the data with the actual data.
#
# While it looks like pynapple has loaded the data, in fact it still did not. By default, calling the NWB object will return an HDF5 dataset.
# !!! warning
# 
#     New in `0.6.6`

print(type(z.values))

# %%
# Notice that the time array is always loaded.

print(type(z.index.values))

# %%
# This is very useful in the case of large dataset that do not fit in memory. You can then get a chunk of the data that will actually be loaded.

z_chunk = z.get(670, 680) # getting 10s of data.

print(z_chunk)

# %%
# Data are now loaded.

print(type(z_chunk.values))

# %%
# You can still apply any high level function of pynapple. For example here, we compute some tuning curves without preloading the dataset.

tc = nap.compute_1d_tuning_curves(data['units'], data['y'], 10)

print(tc)

# %%
#   !!! warning
#       Carefulness should still apply when calling any pynapple function on a memory map. Pynapple does not implement any batching function internally. Calling a high level function of pynapple on a dataset that do not fit in memory will likely cause a memory error.

# %%
# To change this behavior, you can pass `lazy_loading=False` when instantiating the `NWBClass`.
path = "MyProject/sub-A2929/A2929-200711/pynapplenwb/A2929-200711.nwb"
data = nap.NWBFile(path, lazy_loading=False)

z = data['z']

print(type(z.d))


# %%
# Numpy memory map
# ----------------
#
# In fact, pynapple can work with any type of memory map. Here we read a binary file with [`np.memmap`](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html).

eeg_path = "MyProject/sub-A2929/A2929-200711/A2929-200711.eeg"
frequency = 1250 # Hz
n_channels = 16
f = open(eeg_path, 'rb') 
startoffile = f.seek(0, 0)
endoffile = f.seek(0, 2)
f.close()
bytes_size = 2
n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
duration = n_samples/frequency
interval = 1/frequency

fp = np.memmap(eeg_path, np.int16, 'r', shape = (n_samples, n_channels))
timestep = np.arange(0, n_samples)/frequency

print(type(fp))

# %%
# Instantiating a pynapple `TsdFrame` will keep the data as a memory map.

eeg = nap.TsdFrame(t=timestep, d=fp)

print(eeg)

# %%
# We can check the type of `eeg.values`.

print(type(eeg.values))


# %%
# Zarr
# --------------
#
# It is also possible to use Higher level library like [zarr](https://zarr.readthedocs.io/en/stable/index.html) also not directly.

import zarr
data = zarr.zeros((10000, 5), chunks=(1000, 5), dtype='i4')
timestep = np.arange(len(data))

tsdframe = nap.TsdFrame(t=timestep, d=data)

# %%
# As the warning suggest, `data` is converted to numpy array.

print(type(tsdframe.d))

# %%
# To maintain a zarr array, you can change the argument `load_array` to False.

tsdframe = nap.TsdFrame(t=timestep, d=data, load_array=False)

print(type(tsdframe.d))

# %%
# Within pynapple, numpy memory map are recognized as numpy array while zarr array are not.

print(type(fp), "Is np.ndarray? ", isinstance(fp, np.ndarray))
print(type(data), "Is np.ndarray? ", isinstance(data, np.ndarray))


# %%
# Similar to numpy memory map, you can use pynapple functions directly.

ep = nap.IntervalSet(0, 10)
tsdframe.restrict(ep)

# %%
group = nap.TsGroup({0:nap.Ts(t=[10, 20, 30])})

sta = nap.compute_event_trigger_average(group, tsdframe, 1, (-2, 3))

print(type(tsdframe.values))
print("\n")
print(sta)
