# Releases

### 0.9.2 (2025-06-16)

- Implement `time_diff` method for time series objects
- Implement `nap.compute_isi_distribution`, which uses `time_diff` to compute the distribution of inter spike intervals
- Fix IntervalSet and TsGroup `__repr__`
- Fix backward compatibility for loading old npz files. 

### 0.9.1 (2025-06-04)

- Fix TsdFrame `__repr__`

### 0.9.0 (2025-05-13)

- New private class: `_MetadataMixin` and `_Metadata(UserDict)` (core/metadata_class.py). Can be inherited by:
    - IntervalSet
    - TsdFrame
    - TsGroup 
  This class assumes that whatever is inheriting it has the private property `self._initialized`.
  `metadata`: public read-only view of metadata
- Add a decimate method to _BaseTsd.
- Adds support for a new derivative method which wraps `np.gradient` with support for epochs and time index

### 0.8.5 (2025-03-24)

- Implements `nap.build_tensor` and `nap.warp_tensor` for trial-based data.
- Fix horizontal slicing for TsdFrame (Issue )
- Fix empty TsGroup. The rate attribute was not added to the metadata dataframe.
- New example notebook : Trial-aligned choice decoding in International Brain Lab data
- Set pynapple version dynamically by reading the github tag.

### 0.8.4 (2025-02-07)

- Fix value printing of IntervalSet when rows are collapsed 
- Backward compatibility fix for loading npz files with TsGroup
- Fix indexing of IntervalSet to be able to use -1
- Add column names for compute_wavelet_transform 

### 0.8.3 (2025-01-24)

- `compute_mean_power_spectral_density` computes the mean periodogram.

### 0.8.2 (2025-01-22)

- `compute_power_spectral_density` now computes the periodogram, where previously it was only computing the FFT
- `compute_fft` has been added that contains the old functionality of `compute_power_spectral_density`.

### 0.8.1 (2025-01-17)

- Bugfix : time support was not updated for `bin_average` and `interpolate` with new `_initialize_tsd_output` method 

### 0.8.0 (2025-01-15)

- New private class: `_MetadataMixin` (core/metadata_class.py). Can be inherited by `IntervalSet`, `TsdFrame` and `TsGroup`.
- `decode_1d` and `decode_2d` now accepts `TsdFrame` as input. 

### 0.7.1 (2024-09-24)

- Fixing nan issue when computing 1d tuning curve (See issue #334).
- Refactor tuning curves and correlogram tests.
- Adding validators decorators for tuning curves and correlogram modules.

### 0.7.0 (2024-09-16)

- Morlet wavelets spectrogram with utility for plotting the wavelets.
- (Mean) Power spectral density. Returns a Pandas DataFrame.
- Convolve function works for any dimension of time series and any dimensions of kernel.
- `dtype` in count function
- `get_slice`: public method with a simplified API, argument start, end, time_units. returns a slice that matches behavior of Base.get.
- `_get_slice`: private method, adds the argument "mode" this can be: "after_t", "before_t", "closest_t", "restrict".
- `split` method for IntervalSet. Argument is `interval_size` in time unit.
- Changed os import to pathlib.
- Fixed pickling issue. TsGroup can now be saved as pickle.
- TsGroup can be created from an iterable of Ts/Tsd objects.
- IntervalSet can be created from (start, end) pairs


### 0.6.6 (2024-05-28)

- Full lazy-loading for NWB file.
- Parameter `load_array` for time series can prevent loading zarr array
- Function to merge a list of `TsGroup`


### 0.6.5 (2024-05-14)

- Full `pynajax` backend compatibility
- Fixed `TsdFrame` column slicing


### 0.6.4 (2024-04-18)

- Fixing IntervalSet `__repr__`. Tabulate conflict with numpy 1.26.


### 0.6.3 (2024-04-17)

- Improving `__repr__` for all objects.
- TsGroup `__getattr__` and `__setattr__` added to access metadata columns directly
- TsGroup `__setitem__` now allows changes directly to metadata
- TsGroup `__getitem__` returns column of metadata if passed as string


### 0.6.2 (2024-04-04)

- `smooth` now takes standard deviation in time units
- Fixed `TsGroup` saving method.
- `__getattr__` of `BaseTsd` allow numpy functions to be attached as attributes of Tsd objects
- Added `get` method for `TsGroup`
- Tsds can be concatenate vertically if time indexes matches.


### 0.6.1 (2024-03-03)

- Fixed pynapple `loc` method for new `IntervalSet`


### 0.6.0 (2024-03-02)

- Refactoring `IntervalSet` to pure numpy ndarray.
- Implementing new chain of inheritance for time series with abstract base class. `base_class.Base` holds the temporal methods for all time series and `Ts`. `time_series.BaseTsd` inherit `Base` and implements the common methods for `Tsd`, `TsdFrame` and `Tsd`.
- Automatic conversion to numpy ndarray for all objects that are numpy-like (typically jax).


### 0.5.1 (2024-01-29)

- Implementing `event_trigger_average` for all dimensions.
- Hiding jitted functions from users.


### 0.5.0 (2023-12-12)

- Removing GUI stack from pynapple. To create a NWB file, users need to install nwbmatic (https://github.com/pynapple-org/nwbmatic)
- Implementing `compute_perievent_continuous`
- Implementing `convolve` for Tsd, TsdFrame and TsdTensor
- Implementing `smooth` for fast gaussian smoothing of time series


### 0.4.1 (2023-10-30)

- Implementing `get` method that return both an interval or the closest timepoint


### 0.4.0 (2023-10-11)

- Implementing the numpy array container approach within pynapple
- TsdTensor for objects larger than 2 dimensions is now available


### 0.3.6 (2023-09-11)

- Fix issue in NWB reader class with units
- Implement a linear interpolation function.


### 0.3.5 (2023-08-08)

- NWB reader class
- NPZ reader class
- Folder class for navigating a dataset.
- Cross-correlograms function can take tuple
- New doc with mkdocs-gallery


### 0.3.4 (2023-06-29)

- 	`TsGroup.to_tsd` and `Tsd.to_tsgroup` transformations
- 	`count` can take IntervalSet
-	Saving to npz functions for all objects.
- 	`tsd.value_from` can take TsdFrame
- 	Warning message for deprecating current IO. 


### 0.3.3 (2023-04-17)

- 	Fixed minor bug with tkinter


### 0.3.2 (2023-04-12)

- 	PyQt removed from the list of dependencies


### 0.3.1 (2022-12-08)

- 	Core functions rewritten with Numba


### 0.2.4 (2022-05-02)


### 0.2.3 (2022-04-05)

-   Fixed minor bug when saving DLC in NWB.


### 0.2.3 (2022-04-05)

-   Alpha release


### 0.2.2 (2022-04-05)

-   Beta testing version for public


### 0.2.1 (2022-02-07)

-   Beta testing version for Peyrache Lab.


### 0.2.0 (2022-01-10)

-   First version for pynapple with main features in core, process and IO.


### 0.2.0 Pre-release (2022-01-06)

-   Pre-release version for pynapple with main features in core and process.


### 0.1.1 (2021-10-25)

-   First release on PyPI.
- 	Firt minimal version