"""
Pynapple interface for the movement package.
Converts movement xarray.Dataset objects to pynapple time series.
"""

from collections import UserDict

import numpy as np
from tabulate import tabulate

from .. import core as nap


def _check_xarray():
    try:
        import xarray  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'xarray' package is required to use from_movement. "
            "Install it with: pip install xarray"
        )


def _dim(ds, *candidates):
    """Return the first candidate that exists as a coordinate in ds."""
    for name in candidates:
        if name in ds.coords:
            return name
    raise KeyError(
        f"None of {candidates} found in dataset coordinates: {list(ds.coords)}"
    )


def _ind_values(ds):
    """Return the list of individual names, handling scalar (dropped-dim) case."""
    ind_dim = _dim(ds, "individual", "individuals")
    val = ds.coords[ind_dim].values.tolist()
    # After ds.sel(individuals="name") the coord becomes a 0-d scalar string
    if isinstance(val, str):
        return [val]
    return val


def _validate_movement_dataset(ds):
    """Validate that ds is a supported movement xarray.Dataset."""
    _check_xarray()
    import xarray as xr

    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"Expected an xarray.Dataset, got {type(ds).__name__}.")

    ds_type = ds.attrs.get("ds_type")
    if ds_type not in ("poses", "bboxes"):
        raise ValueError(
            f"Dataset 'ds_type' attribute must be 'poses' or 'bboxes', got {ds_type!r}. "
            "Make sure this dataset was created by the movement package."
        )

    time_unit = ds.attrs.get("time_unit", "seconds")
    if time_unit != "seconds":
        raise ValueError(
            f"Dataset time_unit is {time_unit!r}. Pynapple requires time in seconds. "
            "Reload the movement dataset with fps provided so that time is in seconds."
        )

    required = {
        "poses": ["position", "confidence"],
        "bboxes": ["position", "shape", "confidence"],
    }
    for var in required[ds_type]:
        if var not in ds:
            raise ValueError(
                f"Expected variable '{var}' not found in the dataset."
            )

    # Ensure individual/individuals and (for poses) keypoint/keypoints are present
    _dim(ds, "individual", "individuals")
    if ds_type == "poses":
        _dim(ds, "keypoint", "keypoints")


def _col_preview(cols, n):
    """Short column preview: 'a, b, c, ...  (n)'."""
    preview = ", ".join(str(c) for c in cols[:3])
    if n > 3:
        preview += ", ..."
    return f"{preview}  ({n})"


def _build_view_rows(ds, individuals):
    """Build tabulate rows from xarray metadata — no pynapple objects created."""
    ds_type = ds.attrs.get("ds_type")
    n_t = len(ds.coords["time"])
    rows = []

    for ind in individuals:
        if ds_type == "poses":
            kp_dim = _dim(ds, "keypoint", "keypoints")
            keypoints = ds.coords[kp_dim].values.tolist()
            spaces = ds.coords["space"].values.tolist()
            pos_cols = [f"{kp}_{sp}" for kp in keypoints for sp in spaces]
            rows.append([ind, "position", "TsdFrame", _col_preview(pos_cols, len(pos_cols))])
            rows.append([ind, "confidence", "TsdFrame", _col_preview(keypoints, len(keypoints))])
        else:  # bboxes
            rows.append([ind, "position", "TsdFrame", "x, y  (2)"])
            rows.append([ind, "shape", "TsdFrame", "width, height  (2)"])
            rows.append([ind, "confidence", "Tsd", f"({n_t},)"])

    return rows


def _sel_individual(da, ind_dim, ind_name):
    """Select one individual from a DataArray, handling scalar (dropped-dim) case."""
    if ind_dim in da.dims:
        return da.sel({ind_dim: ind_name})
    # Dimension was already dropped by an upstream .sel() call — data is already
    # for this individual, just return as-is
    return da


def _poses_individual_to_pynapple(ds, ind_name):
    """Convert a single individual's poses data to pynapple objects."""
    t = ds.coords["time"].values
    kp_dim = _dim(ds, "keypoint", "keypoints")
    ind_dim = _dim(ds, "individual", "individuals")
    keypoints = ds.coords[kp_dim].values.tolist()
    spaces = ds.coords["space"].values.tolist()

    # position: (time, space, keypoint) — select individual if dimension still present
    pos = _sel_individual(ds["position"], ind_dim, ind_name).values
    n_t, n_space, n_kp = pos.shape

    # Flatten to (time, n_kp * n_space) with columns: nose_x, nose_y, ear_x, ear_y, ...
    cols = [f"{kp}_{sp}" for kp in keypoints for sp in spaces]
    d_pos = pos.transpose(0, 2, 1).reshape(n_t, n_kp * n_space)
    position = nap.TsdFrame(t=t, d=d_pos, columns=cols)

    # confidence: (time, keypoint)
    conf = _sel_individual(ds["confidence"], ind_dim, ind_name).values
    confidence = nap.TsdFrame(t=t, d=conf, columns=keypoints)

    return {"position": position, "confidence": confidence}


def _bboxes_individual_to_pynapple(ds, ind_name):
    """Convert a single individual's bboxes data to pynapple objects."""
    t = ds.coords["time"].values
    ind_dim = _dim(ds, "individual", "individuals")

    # position: (time, space)
    pos = _sel_individual(ds["position"], ind_dim, ind_name).values
    position = nap.TsdFrame(t=t, d=pos, columns=["x", "y"])

    # shape: (time, space)
    shp = _sel_individual(ds["shape"], ind_dim, ind_name).values
    shape = nap.TsdFrame(t=t, d=shp, columns=["width", "height"])

    # confidence: (time,)
    conf = _sel_individual(ds["confidence"], ind_dim, ind_name).values.ravel()
    confidence = nap.Tsd(t=t, d=conf)

    return {"position": position, "shape": shape, "confidence": confidence}


def _individual_to_pynapple(ds, ind_name):
    ds_type = ds.attrs.get("ds_type")
    if ds_type == "poses":
        return _poses_individual_to_pynapple(ds, ind_name)
    return _bboxes_individual_to_pynapple(ds, ind_name)


def _variable_keys(ds):
    ds_type = ds.attrs.get("ds_type")
    if ds_type == "poses":
        return ["position", "confidence"]
    return ["position", "shape", "confidence"]


class MovementDataset(UserDict):
    """
    Dict-like container returned by from_movement().

    Prints a preview table on repr (same style as NWBFile).
    Pynapple objects are built lazily on first access.

    For a single individual (or when individual= is passed to from_movement),
    keys are variable names: 'position', 'confidence' (and 'shape' for bboxes).
    For multiple individuals, keys are individual names — each resolving to
    a per-individual MovementDataset.

    Examples
    --------
    >>> import pynapple as nap
    >>> data = nap.from_movement(ds)
    >>> data
    test.h5  (poses | 100 frames | 10.0 fps)
    ╭──────────────┬────────────────┬──────────┬──────────────────────────────────────╮
    │ Individual   │ Variable       │ Type     │ Columns / Shape                      │
    ├──────────────┼────────────────┼──────────┼──────────────────────────────────────┤
    │ mouse1       │ position       │ TsdFrame │ nose_x, nose_y, left_ear_x, ...  (6) │
    │ mouse1       │ confidence     │ TsdFrame │ nose, left_ear, right_ear  (3)       │
    ╰──────────────┴────────────────┴──────────┴──────────────────────────────────────╯
    >>> data['position']
    Time (s)    nose_x    nose_y    ...
    """

    def __init__(self, ds, individuals, name=""):
        self._ds = ds
        self._individuals = list(individuals)
        self._name = name
        self._single = len(self._individuals) == 1
        self._view = _build_view_rows(ds, self._individuals)

        # Pre-populate data dict with None sentinels so keys(), __contains__, __iter__ work
        keys = _variable_keys(ds) if self._single else self._individuals
        UserDict.__init__(self, {k: None for k in keys})

    def __repr__(self):
        ds_type = self._ds.attrs.get("ds_type", "?")
        n_frames = len(self._ds.coords["time"])
        fps = self._ds.attrs.get("fps", "?")
        title = f"{self._name}  ({ds_type} | {n_frames} frames | {fps} fps)\n"
        headers = ["Individual", "Variable", "Type", "Columns / Shape"]
        return title + tabulate(self._view, headers=headers, tablefmt="mixed_outline")

    def __getitem__(self, key):
        if key not in self.data:
            raise KeyError(key)
        if self.data[key] is None:
            self._load(key)
        return self.data[key]

    def _load(self, key):
        if self._single:
            # Load all variables for this individual at once and cache them
            ind = self._individuals[0]
            objects = _individual_to_pynapple(self._ds, ind)
            for k, v in objects.items():
                if k in self.data:
                    self.data[k] = v
        else:
            # Multi-individual: build a sub-MovementDataset for this individual
            self.data[key] = MovementDataset(self._ds, [key], name=key)


def from_movement(ds, individual=None):
    """
    Convert a movement xarray.Dataset to a MovementDataset.

    Parameters
    ----------
    ds : xarray.Dataset
        A movement poses or bboxes dataset (ds.attrs['ds_type'] must be 'poses'
        or 'bboxes', and ds.attrs['time_unit'] must be 'seconds').
    individual : str, optional
        If provided, expose only this individual at the top level. The returned
        MovementDataset will have variable names as keys ('position', 'confidence',
        etc.) rather than individual names.

    Returns
    -------
    MovementDataset
        Dict-like object. Prints a preview table on repr.

        - Single individual (or individual= given): keys are 'position',
          'confidence' (and 'shape' for bboxes datasets).
        - Multiple individuals: keys are individual names, each resolving to
          a per-individual MovementDataset with variable-name keys.

    Raises
    ------
    TypeError
        If ds is not an xarray.Dataset.
    ValueError
        If ds_type is not 'poses' or 'bboxes', if time_unit is not 'seconds',
        or if required variables are missing.
    ImportError
        If xarray is not installed.

    Examples
    --------
    Single individual:

    >>> import pynapple as nap
    >>> data = nap.from_movement(ds)
    >>> position = data['position']   # TsdFrame with columns nose_x, nose_y, ...
    >>> confidence = data['confidence']

    Multiple individuals:

    >>> data = nap.from_movement(ds_multi)
    >>> mouse1_pos = data['mouse1']['position']

    Select one individual from a multi-individual dataset:

    >>> data = nap.from_movement(ds_multi, individual='mouse1')
    >>> position = data['position']
    """
    _validate_movement_dataset(ds)

    individuals = _ind_values(ds)
    name = ds.attrs.get("source_file", "movement dataset")
    if name:
        name = str(name).split("/")[-1]  # basename only

    if individual is not None:
        if individual not in individuals:
            raise ValueError(
                f"Individual {individual!r} not found. "
                f"Available individuals: {individuals}"
            )
        individuals = [individual]

    return MovementDataset(ds, individuals, name=name)
