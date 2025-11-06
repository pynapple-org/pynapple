# ====================
# imports
# ====================
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import seaborn as sns
import xarray as xr
from dandi.dandiapi import DandiAPIClient
import fsspec
from fsspec.implementations.cached import CachingFileSystem
from pynwb import NWBHDF5IO
import h5py

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
xr.set_options(display_expand_attrs=False)


# ====================
# data loading
# ====================
dandiset_id, filepath = (
    "000053",
    "sub-npI5/sub-npI5_ses-20190414_behavior+ecephys.nwb",
)
with DandiAPIClient() as client:
    asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(filepath)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
fs = CachingFileSystem(
    fs=fsspec.filesystem("http"),
    cache_storage=str(Path("~/.caches/nwb-cache").expanduser()),
)
io = NWBHDF5IO(file=h5py.File(fs.open(s3_url, "rb")), load_namespaces=True)
nwb = nap.NWBFile(io.read(), lazy_loading=True)

# ====================
# data visualisation
# ====================

example_epoch = nap.IntervalSet(0, 350)
fig, (ax_position, ax_speed) = plt.subplots(2, 1, sharex=True)
ax_position.plot(nwb["position"].restrict(example_epoch))
ax_position.set_ylabel("position (cm)")
ax_speed.plot(nwb["body_speed"].restrict(example_epoch))
ax_speed.set_ylabel("speed (cm/s)")
plt.xlabel("time (s)")
plt.show()

# ====================
# tuning curves
# ====================
print(nwb["position"].min())
print(nwb["position"].max())
moving = nwb["body_speed"].threshold(5, method="above").time_support
tuning_curves = nap.compute_tuning_curves(
    data=nwb["units"],
    features=nwb["position"],
    epochs=moving.intersect(nwb["trials"]),
    feature_names=["position"],
    bins=100,
    range=[(0, 400)],
)
tuning_curves.values = (
    tuning_curves.values - np.mean(tuning_curves.values, axis=1, keepdims=True)
) / np.std(tuning_curves.values, axis=1, keepdims=True)
tuning_curves.plot()
plt.show()
