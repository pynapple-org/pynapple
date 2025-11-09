# ====================
# imports
# ====================
from pathlib import Path
import matplotlib.pyplot as plt
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
    "sub-npJ1/sub-npJ1_ses-20190521_behavior+ecephys.nwb",
    # "sub-npI5/sub-npI5_ses-20190414_behavior+ecephys.nwb",
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

# for trial in nwb["trials"].intersect(example_epoch):
#    ax_position.axvspan(trial.start[0], trial.end[0], color="gray", alpha=0.2)
plt.xlabel("time (s)")
plt.show()

# ====================
# tuning curves
# ====================
moving = nwb["body_speed"].threshold(3, method="above").time_support
epochs = moving.intersect(nwb["trials"])
print(nwb["position"].min())
print(nwb["position"].max())
tuning_curves = nap.compute_tuning_curves(
    data=nwb["units"].count(0.02),
    features=nwb["position"].clip(0, 400),
    feature_names=["position"],
    epochs=epochs,
    bins=400,
)
# tuning_curves.values = (
#    tuning_curves.values - np.mean(tuning_curves.values, axis=1, keepdims=True)
# ) / np.std(tuning_curves.values, axis=1, keepdims=True)
tuning_curves[0].plot()
plt.show()

# ====================
# mutual information
# ====================
mutual_information = nap.compute_mutual_information(tuning_curves)
print(mutual_information)

# ====================
# null distribution
# ====================
null_distribution = [
    nap.shift_timestamps(nwb["units"], min_shift=20.0) for _ in range(10)
]
