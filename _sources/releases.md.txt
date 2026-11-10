# Releases

### Unreleased


### 0.11.4 (2026-08-19)

- Major speed-up of `restrict` and object reconstruction: internal operations that provably return a sorted, in-support index skip revalidation, and `restrict` uses `searchsorted`. At 10M samples, `restrict` 33 ms -> 2.8 ms, `get` 15 ms -> 12 us.
- `compute_tuning_curves` accepts bin edges directly for a single feature, and errors when the number of bin specifications does not match the number of features.
- New [Learning material](learning_material) page collecting workshop and summer school material.
- `convolve` and `smooth` now warn when the kernel is longer than an epoch, where the output is entirely determined by the zero-padding.
- NWB loader attaches `rois` metadata to `TsdFrame` from a `RoiResponseSeries`, skipping per-row array columns (`image_mask`, `pixel_mask`).
- `IntervalSet` drops rows with `NaN` start or end times at initialization, with a warning.
- `NeuroSuiteIO` XML parsing no longer fails on a missing `units` section or empty `nSamples`, `nFeatures`, `peakSampleIndex` fields.
- Fixed tuning curves carrying the pandas index name (e.g. `id`) into the xarray coordinate instead of `unit`.
- Added a `codespell` tox environment and CI workflow, plus typo fixes throughout.
- Added Python 3.12 to the tox test matrix.
- Added [pynapple-org/claude-skills](https://github.com/pynapple-org/claude-skills), a Claude Code skill teaching pynapple idioms to AI coding assistants. See [External Projects](external).


### 0.11.3 (2026-05-26)

- Fixed `NeuroSuiteIO` channel indexing: `skip` and `groups` are now keyed by channel ID instead of sequential count, correcting channel ordering when IDs are non-contiguous.
- Fixed `TsIndex` to preserve full floating-point precision for timestamps already in seconds by removing unnecessary `np.around` rounding, resolving precision loss for UTC-based timestamps.
- Fixed doctests across core modules (`interval_set`, `time_series`, `ts_group`) and process modules (`signal`, `decoding`, `randomize`). Added a GitHub Actions CI workflow to run doctests on every pull request.


### 0.11.2 (2026-05-13)

- Fixed out-of-bounds memory access in `jitrestrict`, `jitrestrict_with_count`, `jitin_interval`, `jitremove_nan`, `jitthreshold`, and `jitunion_isets` when called with empty time arrays or empty epoch sets. With Numba JIT enabled these manifested as random crashes or segfaults; with JIT disabled they raised `IndexError`.
- Fixed missing `k < m` bounds guard in `jitrestrict_with_count` and `jitin_interval`, which could cause an out-of-bounds read when all epoch ends precede the first timestamp.
- Fixed `jitthreshold` producing an out-of-bounds access when the input contains a single time point.
- Fixed undefined `nan_cond` variable in `jitvaluefrom` when an epoch contains exactly one target timestamp and mode is `before` or `closest`, which could silently return wrong results instead of `NaN`.
- Fixed `compute_perievent` crashing on `TsGroup` inputs when any event has no spikes within the requested window.


### 0.11.1 (2026-05-06)

- New `signal` module exposing `apply_hilbert_transform`, `compute_hilbert_envelope`, and `compute_hilbert_phase` for computing the Hilbert transform, signal envelope, and instantaneous phase of time series.
- `detect_oscillatory_events` updated to use the Hilbert transform internally and to validate input types.
- New `find_peaks` method for `Tsd` and `TsdFrame`, wrapping `scipy.signal.find_peaks` with support for epoch-restricted peak detection and optional return of peak properties.
- New tutorial `tutorial_ripple_detection.md` demonstrating oscillatory event detection on hippocampal LFP data.
- `NeuroSuiteIO` now reads epoch files and returns them as an `IntervalSet`.
- Added `copy` method to `TsGroup`.
- NWB loader now reads `ElectrodesTable` metadata when loading an `ElectricalSeries`.
- Fixed time support propagation in `compute_perievent`.


### 0.11.0 (2026-03-29)

- New `EphysReader` class and `NeoSignalInterface` for reading electrophysiology files through the [Neo](https://neo.readthedocs.io) library. Supports a wide range of formats (Plexon, Open Ephys, Neuralynx, …) with lazy loading of analog signals.
- New `NeuroSuiteIO` class for reading Neuroscope/NeuroSuite formatted data (binary `.dat`/`.eeg`/`.lfp` files and `.clu`/`.res` spike sorting results).
- New `load_binary_file` function for loading raw binary electrophysiology recordings.
- Refactoring of the perievent module with a unified `compute_perievent` function that automatically handles both discrete (Ts, TsGroup) and continuous (Tsd, TsdFrame, TsdTensor) data.
- `compute_event_trigger_average` renamed to `compute_event_triggered_average`. `compute_perievent_continuous` replaced by `compute_spike_triggered_average` (an alias of `compute_event_triggered_average`).
- Scipy and xarray are now imported lazily at runtime rather than at package import, reducing startup time.
- Fixed scrambled electrode metadata when loading NWB files whose units table contains a `DynamicTableRegion` column.
- Fixed `warp_tensor` to correctly handle empty intervals.


### 0.10.3 (2026-02-06)

- Added a `subsample` method to TsGroup, enabling random subsampling of timestamps per element with reproducibility options and support for both Ts and Tsd.
- Improved handling of empty or zero-length IntervalSet objects to ensure correct initialization of time_support and rate.
- Fixed `nap.randomize.shift_timestamps` with a mode argument to either wrap or drop timestamps that exceed the interval boundaries after shifting.


### 0.10.2 (2025-12-05)

- New tutorial, `tutorial_null_distributions.md`, demonstrating how to use randomization methods to generate null distributions for testing spatial firing.
- `compute_tuning_curves` include firing rates (rates) in the returned attributes
- Fixed a key mismatch in `ts_group.py` for loading data from NPZ files, correcting `"data"` to `"d"`

### 0.10.1 (2025-10-30)

- Fixing smoothing for `nap.decode_bayes`.
- Fixing `np.einsum`.

### 0.10.0 (2025-10-27)

- Generalizing `nap.compute_tuning_curves`. It can take any time series object (Tsd, TsdFrame, TsGroup, TsdTensor) as input and 
  work for any dimension of data.
- `nap.compute_1d_tuning_curve`, `nap.compute_2d_tuning_curve`, `nap.compute_1d_tuning_curve_continuous`, `nap.compute_2d_tuning_curve_continuous`
  are being deprecated in favor of the general `nap.compute_tuning_curves`.
- Generalization of `nap.decode_1d` and `nap.decode_2d` to `nap.decode_bayes` for bayesian decoding of any dimension of data.
- New function `nap.decode_template` for template matching decoding of any dimension of data.
- Metadata can be restricted with `restrict_info`.
- New function `detect_oscillatory_events` to detect oscillatory events in a Tsd object.
- Fix TsdFrame `__repr__` for boolean data type.
- Refactoring of `nap.compute_mutual_information` to take as input xarray tuning curves object.
- `in_interval` method for IntervalSet to check if time points are within intervals.
- Refactoring `nap.compute_discrete_tuning_curves` to `compute_response_per_epoch`.
- Tuning curves function can return spike counts and occupancy separately.

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
- 	First minimal version