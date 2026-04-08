import math
import os

from scipy.signal import hilbert

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
import pynapple as nap

# some configuration, you can ignore this
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
path = "Achilles_10252013_EEG.nwb"
if path not in os.listdir("."):
    r = requests.get(f"https://osf.io/2dfvp/download", stream=True)
    block_size = 1024 * 1024
    with open(path, "wb") as f:
        for data in r.iter_content(block_size):
            f.write(data)
data = nap.load_file(path)
data

lfp = data["eeg"]
print(lfp)
print(lfp.rate)

filtered_lfp = nap.apply_bandpass_filter(lfp, cutoff=(150, 250), fs=lfp.rate)

segment = nap.IntervalSet(18356.0, 18357.5)
# plt.plot(lfp.restrict(segment), label="LFP")
# plt.plot(filtered_lfp.restrict(segment), label="filtered LFP (150-250Hz)")
# plt.xlabel("time (s)")
# plt.ylabel("amplitude (a.u.)")
# plt.legend()
# plt.show()

envelope = hilbert(filtered_lfp.values, axis=0)
envelope = nap.TsdFrame(t=lfp.times(), d=np.abs(envelope), columns=lfp.columns)
# plt.plot(filtered_lfp.restrict(segment), label="filtered LFP (150-250Hz)")
# plt.plot(envelope.restrict(segment), label="envelope")
# plt.xlabel("time (s)")
# plt.ylabel("amplitude (a.u.)")
# plt.legend()
# plt.show()

filter = np.ones(7) / 7
smoothed = envelope.convolve(filter)
zscored = (smoothed - smoothed.mean()) / smoothed.std()
print(zscored)

fig, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
axs[0].plot(lfp.restrict(segment), alpha=0.5)
axs[0].set_title("LFP")
axs[1].plot(filtered_lfp.restrict(segment), alpha=0.5)
axs[1].plot(envelope.restrict(segment))
axs[1].set_title("filtered LFP (150-250Hz)")
axs[2].plot(zscored.restrict(segment))
axs[2].set_title("smoothed & z-scored")
axs[2].set_xlabel("time (s)")
plt.show()
