"""Tests for pynapple.io.movement (from_movement / MovementDataset)."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

import pynapple as nap
from pynapple.io.movement import MovementDataset, from_movement


# ---------------------------------------------------------------------------
# Fixtures: synthetic movement-style xarray.Dataset objects
# ---------------------------------------------------------------------------


def _make_poses_ds(n_individuals=1, n_keypoints=3, n_frames=100, fps=10.0):
    n_sp = 2
    ind_names = [f"mouse{i+1}" for i in range(n_individuals)]
    kp_names = ["nose", "left_ear", "right_ear"][:n_keypoints]
    t = np.arange(n_frames) / fps

    return xr.Dataset(
        {
            "position": xr.DataArray(
                np.random.rand(n_frames, n_sp, n_keypoints, n_individuals),
                dims=["time", "space", "keypoint", "individual"],
            ),
            "confidence": xr.DataArray(
                np.random.rand(n_frames, n_keypoints, n_individuals),
                dims=["time", "keypoint", "individual"],
            ),
        },
        coords={
            "time": t,
            "space": ["x", "y"],
            "keypoint": kp_names,
            "individual": ind_names,
        },
        attrs={
            "ds_type": "poses",
            "fps": fps,
            "time_unit": "seconds",
            "source_file": "test.h5",
        },
    )


def _make_bboxes_ds(n_individuals=1, n_frames=50, fps=25.0):
    n_sp = 2
    ind_names = [f"id_{i}" for i in range(n_individuals)]
    t = np.arange(n_frames) / fps

    return xr.Dataset(
        {
            "position": xr.DataArray(
                np.random.rand(n_frames, n_sp, n_individuals),
                dims=["time", "space", "individual"],
            ),
            "shape": xr.DataArray(
                np.random.rand(n_frames, n_sp, n_individuals),
                dims=["time", "space", "individual"],
            ),
            "confidence": xr.DataArray(
                np.random.rand(n_frames, n_individuals),
                dims=["time", "individual"],
            ),
        },
        coords={
            "time": t,
            "space": ["x", "y"],
            "individual": ind_names,
        },
        attrs={
            "ds_type": "bboxes",
            "fps": fps,
            "time_unit": "seconds",
            "source_file": "test_bbox.h5",
        },
    )


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="xarray.Dataset"):
            from_movement({"not": "a dataset"})

    def test_missing_ds_type_raises(self):
        ds = _make_poses_ds()
        del ds.attrs["ds_type"]
        with pytest.raises(ValueError, match="ds_type"):
            from_movement(ds)

    def test_invalid_ds_type_raises(self):
        ds = _make_poses_ds()
        ds.attrs["ds_type"] = "tracks"
        with pytest.raises(ValueError, match="ds_type"):
            from_movement(ds)

    def test_wrong_time_unit_raises(self):
        ds = _make_poses_ds()
        ds.attrs["time_unit"] = "frames"
        with pytest.raises(ValueError, match="time_unit"):
            from_movement(ds)

    def test_missing_variable_raises(self):
        ds = _make_poses_ds()
        ds = ds.drop_vars("confidence")
        with pytest.raises(ValueError, match="confidence"):
            from_movement(ds)

    def test_invalid_individual_raises(self):
        ds = _make_poses_ds(n_individuals=2)
        with pytest.raises(ValueError, match="'ghost'"):
            from_movement(ds, individual="ghost")


# ---------------------------------------------------------------------------
# Poses — single individual
# ---------------------------------------------------------------------------


class TestPosesSingleIndividual:
    @pytest.fixture
    def data(self):
        return from_movement(_make_poses_ds(n_individuals=1, n_keypoints=3))

    def test_returns_movement_dataset(self, data):
        assert isinstance(data, MovementDataset)

    def test_keys(self, data):
        assert set(data.keys()) == {"position", "confidence"}

    def test_position_type_and_shape(self, data):
        pos = data["position"]
        assert isinstance(pos, nap.TsdFrame)
        assert pos.shape == (100, 6)  # n_frames x (n_kp * n_space)

    def test_position_columns(self, data):
        expected = ["nose_x", "nose_y", "left_ear_x", "left_ear_y", "right_ear_x", "right_ear_y"]
        assert list(data["position"].columns) == expected

    def test_confidence_type_and_shape(self, data):
        conf = data["confidence"]
        assert isinstance(conf, nap.TsdFrame)
        assert conf.shape == (100, 3)

    def test_confidence_columns(self, data):
        assert list(data["confidence"].columns) == ["nose", "left_ear", "right_ear"]

    def test_time_index(self, data):
        expected_t = np.arange(100) / 10.0
        np.testing.assert_allclose(data["position"].index.values, expected_t)

    def test_repr_contains_ds_type(self, data):
        assert "poses" in repr(data)

    def test_repr_contains_filename(self, data):
        assert "test.h5" in repr(data)

    def test_lazy_loading(self):
        ds = _make_poses_ds()
        data = from_movement(ds)
        # Before access, all values are None sentinels
        assert all(v is None for v in data.data.values())
        _ = data["position"]
        # After access, all variables for the individual are populated
        assert data.data["position"] is not None
        assert data.data["confidence"] is not None


# ---------------------------------------------------------------------------
# Poses — multiple individuals
# ---------------------------------------------------------------------------


class TestPosesMultipleIndividuals:
    @pytest.fixture
    def data(self):
        return from_movement(_make_poses_ds(n_individuals=2, n_keypoints=3))

    def test_keys_are_individual_names(self, data):
        assert set(data.keys()) == {"mouse1", "mouse2"}

    def test_subitem_is_movement_dataset(self, data):
        assert isinstance(data["mouse1"], MovementDataset)

    def test_subitem_keys(self, data):
        assert set(data["mouse1"].keys()) == {"position", "confidence"}

    def test_position_shape_per_individual(self, data):
        pos = data["mouse1"]["position"]
        assert isinstance(pos, nap.TsdFrame)
        assert pos.shape == (100, 6)

    def test_individuals_have_same_columns(self, data):
        assert list(data["mouse1"]["position"].columns) == list(
            data["mouse2"]["position"].columns
        )

    def test_repr_lists_all_individuals(self, data):
        r = repr(data)
        assert "mouse1" in r
        assert "mouse2" in r


# ---------------------------------------------------------------------------
# Poses — individual= filter argument
# ---------------------------------------------------------------------------


class TestPosesIndividualFilter:
    def test_returns_flat_dataset(self):
        ds = _make_poses_ds(n_individuals=2)
        data = from_movement(ds, individual="mouse1")
        assert set(data.keys()) == {"position", "confidence"}

    def test_position_values_match(self):
        ds = _make_poses_ds(n_individuals=2)
        data_filtered = from_movement(ds, individual="mouse1")
        data_full = from_movement(ds)
        np.testing.assert_allclose(
            data_filtered["position"].values,
            data_full["mouse1"]["position"].values,
        )


# ---------------------------------------------------------------------------
# Bboxes — single individual
# ---------------------------------------------------------------------------


class TestBboxesSingleIndividual:
    @pytest.fixture
    def data(self):
        return from_movement(_make_bboxes_ds(n_individuals=1))

    def test_keys(self, data):
        assert set(data.keys()) == {"position", "shape", "confidence"}

    def test_position_type_and_columns(self, data):
        pos = data["position"]
        assert isinstance(pos, nap.TsdFrame)
        assert list(pos.columns) == ["x", "y"]

    def test_shape_type_and_columns(self, data):
        shp = data["shape"]
        assert isinstance(shp, nap.TsdFrame)
        assert list(shp.columns) == ["width", "height"]

    def test_confidence_is_tsd(self, data):
        conf = data["confidence"]
        assert isinstance(conf, nap.Tsd)
        assert conf.shape == (50,)

    def test_time_index(self, data):
        expected_t = np.arange(50) / 25.0
        np.testing.assert_allclose(data["position"].index.values, expected_t)


# ---------------------------------------------------------------------------
# Bboxes — multiple individuals
# ---------------------------------------------------------------------------


class TestBboxesMultipleIndividuals:
    @pytest.fixture
    def data(self):
        return from_movement(_make_bboxes_ds(n_individuals=3))

    def test_keys_are_individual_names(self, data):
        assert set(data.keys()) == {"id_0", "id_1", "id_2"}

    def test_subitem_keys(self, data):
        assert set(data["id_0"].keys()) == {"position", "shape", "confidence"}

    def test_confidence_is_tsd_per_individual(self, data):
        assert isinstance(data["id_1"]["confidence"], nap.Tsd)


# ---------------------------------------------------------------------------
# Preview table
# ---------------------------------------------------------------------------


class TestPreviewTable:
    def test_view_rows_poses(self):
        data = from_movement(_make_poses_ds(n_individuals=1, n_keypoints=3))
        # 2 rows: position, confidence
        assert len(data._view) == 2
        assert data._view[0][1] == "position"
        assert data._view[1][1] == "confidence"

    def test_view_rows_bboxes(self):
        data = from_movement(_make_bboxes_ds(n_individuals=1))
        # 3 rows: position, shape, confidence
        assert len(data._view) == 3
        assert {r[1] for r in data._view} == {"position", "shape", "confidence"}

    def test_view_rows_multi_individual(self):
        data = from_movement(_make_poses_ds(n_individuals=2, n_keypoints=3))
        # 4 rows: 2 individuals × 2 variables
        assert len(data._view) == 4

    def test_repr_is_string(self):
        data = from_movement(_make_poses_ds())
        assert isinstance(repr(data), str)

    def test_repr_contains_fps(self):
        data = from_movement(_make_poses_ds(fps=30.0))
        assert "30.0" in repr(data)
