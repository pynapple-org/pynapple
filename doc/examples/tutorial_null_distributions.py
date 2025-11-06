# ====================
# imports
# ====================
from pathlib import Path
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
    "000582",
    "sub-10073/sub-10073_ses-17010302_behavior+ecephys.nwb",
)
with DandiAPIClient() as client:
    asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(filepath)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
fs = CachingFileSystem(
    fs=fsspec.filesystem("http"),
    cache_storage=str(Path("~/.caches/nwb-cache").expanduser()),
)
io = NWBHDF5IO(file=h5py.File(fs.open(s3_url, "rb")), load_namespaces=True)
nwb = nap.NWBFile(io.read())
print(nwb)

# ====================
# data visualisation
# ====================
