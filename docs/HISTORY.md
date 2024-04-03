History
=======

This package somehow started about 20 years ago in Bruce McNaughton's lab. Dave Redish started the *TSToolbox* package in Matlab. 
Another postdoc in the lab, Francesco Battaglia, then made major contributions to the package. Francesco passed it on to Adrien Peyrache and other trainees in Paris and The Netherlands.
Around 2016-2017, Luke Sjulson started *TSToolbox2*, still in Matlab and which includes some important changes.

In 2018, Francesco started neuroseries, a Python package built on Pandas. It was quickly adopted in Adrien's lab, especially by Guillaume Viejo, a postdoc in the lab. Gradually, the majority of the lab was using it and new functions were constantly added.
In 2021, Guillaume and other trainees in Adrien's lab decided to fork from neuroseries and started *pynapple*. The core of pynapple is largely built upon neuroseries. Some of the original changes to TSToolbox made by Luke were included in this package, especially the *time_support* property of all ts/tsd objects.


0.6.2 (Soon)
------------------

- `smooth` now takes standard deviation in time units
- Fixed `TsGroup` saving method.
- `__getattr__` of `BaseTsd` allow numpy functions to be attached as attributes of Tsd objects
- Added `get` method for `TsGroup`


0.6.1 (2024-03-03)
------------------

- Fixed pynapple `loc` method for new `IntervalSet`


0.6.0 (2024-03-02)
------------------

- Refactoring `IntervalSet` to pure numpy ndarray.
- Implementing new chain of inheritance for time series with abstract base class. `base_class.Base` holds the temporal methods for all time series and `Ts`. `time_series.BaseTsd` inherit `Base` and implements the common methods for `Tsd`, `TsdFrame` and `Tsd`.
- Automatic conversion to numpy ndarray for all objects that are numpy-like (typically jax).


0.5.1 (2024-01-29)
------------------

- Implementing `event_trigger_average` for all dimensions.
- Hiding jitted functions from users.


0.5.0 (2023-12-12)
------------------

- Removing GUI stack from pynapple. To create a NWB file, users need to install nwbmatic (https://github.com/pynapple-org/nwbmatic)
- Implementing `compute_perievent_continuous`
- Implementing `convolve` for Tsd, TsdFrame and TsdTensor
- Implementing `smooth` for fast gaussian smoothing of time series


0.4.1 (2023-10-30)
------------------

- Implementing `get` method that return both an interval or the closest timepoint


0.4.0 (2023-10-11)
------------------

- Implementing the numpy array container approach within pynapple
- TsdTensor for objects larger than 2 dimensions is now available


0.3.6 (2023-09-11)
------------------

- Fix issue in NWB reader class with units
- Implement a linear interpolation function.


0.3.5 (2023-08-08)
------------------

- NWB reader class
- NPZ reader class
- Folder class for navigating a dataset.
- Cross-correlograms function can take tuple
- New doc with mkdocs-gallery


0.3.4 (2023-06-29)
------------------

- 	`TsGroup.to_tsd` and `Tsd.to_tsgroup` transformations
- 	`count` can take IntervalSet
-	Saving to npz functions for all objects.
- 	`tsd.value_from` can take TsdFrame
- 	Warning message for deprecating current IO. 


0.3.3 (2023-04-17)
------------------

- 	Fixed minor bug with tkinter


0.3.2 (2023-04-12)
------------------

- 	PyQt removed from the list of dependencies


0.3.1 (2022-12-08)
------------------

- 	Core functions rewritten with Numba


0.2.4 (2022-05-02)
------------------


0.2.3 (2022-04-05)
------------------

-   Fixed minor bug when saving DLC in NWB.

0.2.3 (2022-04-05)
------------------

-   Alpha release


0.2.2 (2022-04-05)
------------------

-   Beta testing version for public


0.2.1 (2022-02-07)
------------------

-   Beta testing version for Peyrache Lab.


0.2.0 (2022-01-10)
------------------

-   First version for pynapple with main features in core, process and IO.


0.2.0 Pre-release (2022-01-06)
------------------------------

-   Pre-release version for pynapple with main features in core and process.


0.1.1 (2021-10-25)
------------------

-   First release on PyPI.
- 	Firt minimal version