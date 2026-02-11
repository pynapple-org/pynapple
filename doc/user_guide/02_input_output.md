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

# Input-output & lazy-loading

Pynapple provides multiple ways to load data. The two main formats are NWB and NPZ. In addition, raw LFP data can be loaded through the NEO library. 

Each pynapple objects can be saved as a [`npz`](https://numpy.org/devdocs/reference/generated/numpy.savez.html) with a special structure and loaded as a `npz`.

In addition, the `Folder` class helps you walk through a set of nested folders to load/save `npz`/`nwb` files.

```{contents}
:local:
:depth: 2
```

## NWB

When loading a NWB file, pynapple will walk through it and test the compatibility of each data structure with a 
pynapple objects. If the data structure is incompatible, pynapple will ignore it. The class that deals with reading 
NWB file is [`nap.NWBFile`](pynapple.io.interface_nwb.NWBFile). You can pass the path to a NWB file or directly an opened NWB file. Alternatively 
you can use the function [`nap.load_file`](pynapple.io.misc.load_file).

:::{note}
Creating the NWB file is outside the scope of pynapple. The NWB files used here have already been created before.
Multiple tools exists to create NWB file automatically. You can check [neuroconv](https://neuroconv.readthedocs.io/en/main/), [NWBGuide](https://nwb-guide.readthedocs.io/en/latest/) or even [NWBmatic](https://github.com/pynapple-org/nwbmatic).
:::


```{code-cell} ipython3
:tags: [hide-cell]

import numpy as np
import pynapple as nap
import os
import requests, math
import tqdm
import matplotlib.pyplot as plt
import numpy as np

nwb_path = 'A2929-200711.nwb'

if nwb_path not in os.listdir("."):
  r = requests.get(f"https://osf.io/fqht6/download", stream=True)
  block_size = 1024*1024
  with open(nwb_path, 'wb') as f:
    for data in tqdm.tqdm(r.iter_content(block_size), unit='MB', unit_scale=True,
      total=math.ceil(int(r.headers.get('content-length', 0))//block_size)):
      f.write(data)
```

```{code-cell} ipython3
data = nap.load_file(nwb_path)

print(data)
```

Pynapple will give you a table with all the entries of the NWB file that are compatible with a pynapple object.
When parsing the NWB file, nothing is loaded. The `NWBFile` class keeps track of the position of the data within the NWB file with a key. You can see it with the attributes `key_to_id`.


```{code-cell} ipython3
data.key_to_id
```

Loading an entry will get pynapple to read the data.


```{code-cell} ipython3
z = data['z']

print(data['z'])
```

Internally, the `NWBClass` has replaced the pointer to the data with the actual data.

While it looks like pynapple has loaded the data, in fact it did not. By default, calling the NWB object will return an HDF5 dataset.


```{code-cell} ipython3
print(type(z.values))
```

Notice that the time array is always loaded.


```{code-cell} ipython3
print(type(z.index.values))
```

This is very useful in the case of large dataset that do not fit in memory. You can then get a chunk of the data that will actually be loaded.


```{code-cell} ipython3
z_chunk = z.get(670, 680) # getting 10s of data.

print(z_chunk)
```

Data are now loaded.


```{code-cell} ipython3
print(type(z_chunk.values))
```

You can still apply any high level function of pynapple. For example here, we compute some tuning curves without preloading the dataset.


```{code-cell} ipython3
tc = nap.compute_tuning_curves(data['units'], data['y'], 10)

```

:::{warning}
Carefulness should still apply when calling any pynapple function on a memory map. Pynapple does not implement any batching function internally. Calling a high level function of pynapple on a dataset that do not fit in memory will likely cause a memory error.
:::


To change this behavior, you can pass `lazy_loading=False` when instantiating the `NWBClass`.


```{code-cell} ipython3
data = nap.NWBFile(nwb_path, lazy_loading=False)

z = data['z']

print(type(z.d))
```

## LFP loading & NEO compatibility

Raw LFP data can be loaded with pynapple through the NEO library.
Internally, pynapple uses the NEO raw IO classes to read the data and convert them to one of the pynapple time series object.
This is done through the class [`nap.EphysReader`](pynapple.io.interface_neo.EphysReader).

See here the [list of supported formats](https://neo.readthedocs.io/en/stable/rawiolist.html) of python-neo.

The minimal example to load a dataset is to instantiate the `EphysReader` class with the path to the data file.

```
import pynapple as nap
data = nap.EphysReader("path_to_your_file")
```

Below are shown a few examples of loading LFP and spikes.

### Neuroscope / Binary file

If you have a session recorded as binary files with Neuroscope, you can load it with the `EphysReader` class or directly 
with the function [`nap.load_eeg`](pynapple.io.misc.load_eeg).

```
📂 my_session
 ┣ 📄 my_session.dat
 ┗ 📄 my_session.xml
```


```
>>> import pynapple as nap

>>> data = nap.EphysReader("path/my_session", format="NeuroscopeIO")
    my_session
    ┍━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━┑
    │ Key                 │ Type     │
    ┝━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━┥
    │ my_session.dat      │ TsdFrame │
    ┕━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━┙

>>> data["my_session.dat"]
    Time (s)            0          1          2          3          4  ...
    ----------  ---------  ---------  ---------  ---------  ---------  -----
    0.0          0.274353   0.249023   0.3125     0.17395    0.240173  ...
    5e-05        0.262146   0.205078   0.325928   0.163574   0.235901  ...
    0.0001       0.263062   0.221863   0.325317   0.181885   0.266113  ...
    ...                                                                ...
    1199.99585  -0.181885  -0.186157  -0.283508  -0.15686   -0.209961  ...
    1199.9959   -0.098877  -0.209961  -0.291138  -0.213623  -0.209656  ...
    1199.99595  -0.109863  -0.141296  -0.257874  -0.188293  -0.163879  ...
    dtype: int16, shape: (23999920, 16)
```

A more direct way to load binary file is to use the function `nap.load_eeg` that will directly return a `TsdFrame` object.

```
>>> import pynapple as nap
>>> data = nap.load_eeg("path/my_session/my_session.dat",
    channel=None # A list of channels to return. If None, returns all channels.
    n_channels=16, # The number of channels in the binary file. Only used if channel is None.
    sampling_rate=20000, # The sampling rate of the data.
    precision='int16' # The precision of the data. Can be 'int16', 'float32' or 'float64'.
    bytes_size=2 # The byte size of the data. Can be 2, 4 or 8.
    )
>>> data
    Time (s)            0          1          2          3          4  ...
    ----------  ---------  ---------  ---------  ---------  ---------  -----
    0.0          0.274353   0.249023   0.3125     0.17395    0.240173  ...
    5e-05        0.262146   0.205078   0.325928   0.163574   0.235901  ...
    0.0001       0.263062   0.221863   0.325317   0.181885   0.266113  ...
    ...                                                                ...
    1199.99585  -0.181885  -0.186157  -0.283508  -0.15686   -0.209961  ...
    1199.9959   -0.098877  -0.209961  -0.291138  -0.213623  -0.209656  ...
    1199.99595  -0.109863  -0.141296  -0.257874  -0.188293  -0.163879  ...
    dtype: float32, shape: (23999920, n_channels)
```

### OpenEphys

A typical OpenEphys dataset has the following structure:

```
📂 Record Node 109
 ┣ 📄 settings.xml
 ┗ 📂 experiment1
    ┗ 📂 recording1
       ┣ 📄 structure.oebin
       ┣ 📄 sync_messages.txt
       ┣ 📂 continuous
       │  ┗ 📂 Neuropix-PXI-103.ProbeA
       │     ┣ 📄 continuous.dat
       │     ┣ 📄 sample_numbers.npy
       │     ┗ 📄 timestamps.npy
       ┗ 📂 events
          ┣ 📂 MessageCenter
          │  ┣ 📄 sample_numbers.npy
          │  ┣ 📄 text.npy
          │  ┗ 📄 timestamps.npy
          ┗ 📂 Neuropix-PXI-103.ProbeA
             ┗ 📂 TTL
                ┣ 📄 full_words.npy
                ┣ 📄 sample_numbers.npy
                ┣ 📄 states.npy
                ┗ 📄 timestamps.npy
```

You can load the LFP data with the `EphysReader` class with :

```
>>> import pynapple as nap

>>> data = nap.EphysReader("path/to/Record Node 109", format="OpenEphysBinaryIO")
    Record Node 109
    ┍━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━┑
    │ Key                                                     │ Type     │
    ┝━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━┥
    │ 0:Record Node 109#Neuropix-PXI-103.ProbeA               │ TsdFrame │
    │ 1:Record Node 109#Neuropix-PXI-103.ProbeASYNC           │ TsdFrame │
    │ 0:Messages                                              │ Ts       │
    │ 1:Neuropixels PXI Sync                                  │ Ts       │
    ┕━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━┙
    
>>> data["0:Record Node 109#Neuropix-PXI-103.ProbeA"]
    Time (s)             0        1         2         3         4  ...
    ------------  --------  -------  --------  --------  --------  -----
    8.044366667   -138.06   -91.65   -115.245  -187.59   -130.455  ...
    8.0444        -132.795  -82.485  -106.08   -185.445  -131.235  ...
    8.044433333   -132.015  -90.87   -105.3    -187.59   -122.85   ...
    ...                                                            ...
    68.606033333    33.54    56.55     50.31    -30.615     6.045  ...
    68.606066667    35.88    56.55     45.045   -35.88     -1.56   ...
    68.6061         32.76    61.815    51.87    -33.54      7.605  ...
    dtype: int16, shape: (1816853, 384)
    
```

### Plexon file

```{code-cell} ipython3
:tags: [hide-cell]
import urllib
distantfile = "https://web.gin.g-node.org/NeuralEnsemble/ephy_testing_data/raw/master/plexon/File_plexon_3.plx"
urllib.request.urlretrieve(distantfile, "File_plexon_3.plx")
```

```{code-cell} ipython3
:tags: [hide-output]
data = nap.EphysReader("File_plexon_3.plx", format="PlexonIO");
```
```{code-cell} ipython3
print(data)
```

```{code-cell} ipython3
lfp = data['0:V']
spikes = data['TsGroup']
```

```{code-cell} ipython3
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
ax1.plot(lfp.t, lfp.d[:])
ax1.set_xlabel("Time (s)")
colors = plt.cm.jet(np.linspace(0, 1, len(spikes)))
ax2.eventplot([spikes[i].t for i in spikes.keys()], colors=colors)
ax2.set_xlabel("Time (s)")
plt.tight_layout()
plt.show()
```

## Saving as NPZ

Pynapple objects have [`save`](pynapple.Tsd.save) methods to save them as npz files. 

```{code-cell} ipython3
tsd = nap.Tsd(t=np.arange(10), d=np.arange(10))
tsd.save("my_tsd.npz")

print(nap.load_file("my_tsd.npz"))
```

To load  a NPZ to pynapple, it must contain particular set of keys.

```{code-cell} ipython3
print(np.load("my_tsd.npz"))
```

When the pynapple object have metadata, they are added to the NPZ file.

```{code-cell} ipython3
tsgroup = nap.TsGroup({
    0:nap.Ts(t=[0,1,2]),
    1:nap.Ts(t=[0,1,2])
    }, metadata={"my_label":["a", "b"]})
tsgroup.save("group")

print(np.load("group.npz", allow_pickle=True))
```
By default, they are added within the `_metadata` key:

```{code-cell} ipython3
print(dict(np.load("group.npz", allow_pickle=True))["_metadata"])
```

## Memory map

### Numpy memory map

Pynapple can work with [`numpy.memmap`](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html).


```{code-cell} ipython3
:tags: [hide-cell]

data = np.memmap("memmap.dat", dtype='float32', mode='w+', shape = (10, 3))

data[:] = np.random.randn(10, 3).astype('float32')

timestep = np.arange(10)
```

```{code-cell}
print(type(data))
```

Instantiating a pynapple `TsdFrame` will keep the `data` as a memory map.


```{code-cell} ipython3
eeg = nap.TsdFrame(t=timestep, d=data)

print(eeg)
```

We can check the type of `eeg.values`.


```{code-cell} ipython3
print(type(eeg.values))
```

### Zarr

It is possible to use Higher level library like [zarr](https://zarr.readthedocs.io/en/stable/index.html) also not directly.


```{code-cell} ipython3
import zarr
zarr_array = zarr.zeros((10000, 5), chunks=(1000, 5), dtype='i4')
timestep = np.arange(zarr_array.shape[0])

tsdframe = nap.TsdFrame(t=timestep, d=zarr_array)
```

As the warning suggest, `zarr_array` is converted to numpy array.


```{code-cell} ipython3
print(type(tsdframe.d))
```

To maintain a zarr array, you can change the argument `load_array` to False.


```{code-cell} ipython3
tsdframe = nap.TsdFrame(t=timestep, d=zarr_array, load_array=False)

print(type(tsdframe.d))
```

Within pynapple, numpy memory map are recognized as numpy array while zarr array are not.


```{code-cell} ipython3
print(type(data), "Is np.ndarray? ", isinstance(data, np.ndarray))
print(type(zarr_array), "Is np.ndarray? ", isinstance(zarr_array, np.ndarray))
```

Similar to numpy memory map, you can use pynapple functions directly.


```{code-cell} ipython3
ep = nap.IntervalSet(0, 10)
tsdframe.restrict(ep)
```

```{code-cell} ipython3
group = nap.TsGroup({0:nap.Ts(t=[10, 20, 30])})

sta = nap.compute_event_trigger_average(group, tsdframe, 1, (-2, 3))

print(type(tsdframe.values))
print("\n")
print(sta)
```

## Navigating a dataset

```{code-cell} ipython3
:tags: [hide-cell]

import zipfile

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

```

We can load a folder containing multiple animals and sessions with the `Folders` class. The method [`nap.load_folder`](pynapple.io.misc.load_folder) provides a shortcut.


```{code-cell} ipython3
project = nap.load_folder(project_path)

print(project)
```

The pynapple IO offers a convenient way of visualizing and navigating a folder 
based dataset. To visualize the whole hierarchy of Folders, you can call the 
view property or the expand function.

```{code-cell} ipython3

project.view

```

Here it shows all the subjects (in this case only A2929), 
all the sessions and all of the derivatives folders. 
It shows as well all the NPZ files that contains a pynapple object 
and the NWB files.

The object project behaves like a nested dictionary. 
It is then easy to loop and navigate through a hierarchy of folders 
when doing analyses. In this case, we are gonna take only the 
session A2929-200711.

```{code-cell} ipython3
session = project["sub-A2929"]["A2929-200711"]
print(session)
```

The Folder view gives the path to any object. It can then be easily loaded.

```{code-cell} ipython3
print(project["sub-A2929"]["A2929-200711"]["pynapplenwb"]["A2929-200711"])
```
### JSON sidecar file

A good practice for sharing datasets is to write as many 
metainformation as possible. Following 
[BIDS](https://bids-standard.github.io/bids-starter-kit/index.html) 
specifications, any data files should be accompagned by a JSON sidecar file.

This is possible using the `Folder` class of pynapple with the argument `description`.

```{code-cell} ipython3
epoch = nap.IntervalSet(start=np.array([0, 3]), end=np.array([1, 6]))
session.save("stimulus-fish", epoch, description="Fish pictures to V1")
```

It is then possible to read the description with the `doc` method of the `Folder` object.

```{code-cell} ipython3
session.doc("stimulus-fish")
```
