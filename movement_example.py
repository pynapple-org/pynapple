"""
Movement → Pynapple minimal example
Based on: https://movement.neuroinformatics.dev/latest/examples/scale.html
"""

import warnings
from movement import sample_data
from movement.filtering import filter_by_confidence, interpolate_over_time, median_filter
from movement.transforms import scale
import pynapple as nap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Load and pre-process
# ---------------------------------------------------------------------------

ds = sample_data.fetch_dataset("DLC_single-mouse_DBTravelator_2D.predictions.h5")

kp_dim  = "keypoint"   if "keypoint"   in ds.coords else "keypoints"
ind_dim = "individual" if "individual" in ds.coords else "individuals"

landmark_keypoints = [
    "Door", "StartPlatL", "StartPlatR", "StepL", "StepR", "TransitionL", "TransitionR",
]

ds_mouse = ds.sel(
    {
        kp_dim:  ~ds[kp_dim].isin(landmark_keypoints),
        ind_dim: ds[ind_dim].values[0],
    }
)

ds_mouse["position"] = filter_by_confidence(ds_mouse.position, ds_mouse.confidence, threshold=0.9)
ds_mouse["position"] = interpolate_over_time(ds_mouse.position, max_gap=40)
ds_mouse["position"] = median_filter(ds_mouse.position, window=6, min_periods=2)

# Scale pixels → cm (1 cm grid ≈ 29 px) and flip y-axis
ds_mouse["position"] = scale(ds_mouse["position"], factor=1.0 / 29.0, space_unit="cm")
y = ds_mouse["position"].sel(space="y")
ds_mouse["position"].loc[dict(space="y")] = y.max() - y

# ---------------------------------------------------------------------------
# Convert to pynapple
# ---------------------------------------------------------------------------

data = nap.from_movement(ds_mouse)
print(data)

position   = data["position"]    # TsdFrame — columns: Nose_x, Nose_y, EarL_x, ...
confidence = data["confidence"]  # TsdFrame — columns: Nose, EarL, ...

print(position)
