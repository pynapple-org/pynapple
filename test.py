import pynapple as nap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)

from scipy.ndimage import gaussian_filter1d

# Fake Tuning curves
N = 6  # Number of neurons
bins = np.linspace(0, 2 * np.pi, 61)
x = np.linspace(-np.pi, np.pi, len(bins) - 1)
tmp = np.roll(np.exp(-((1.5 * x) ** 2)), (len(bins) - 1) // 2)
tc = np.array([np.roll(tmp, i * (len(bins) - 1) // N) for i in range(N)]).T

tc_1d = pd.DataFrame(index=bins[0:-1], data=tc)

# Feature
T = 10000
dt = 0.01
timestep = np.arange(0, T) * dt
feature = nap.Tsd(
    t=timestep,
    d=gaussian_filter1d(np.cumsum(np.random.randn(T) * 0.5), 20) % (2 * np.pi),
)
index = np.digitize(feature, bins) - 1

# Spiking activity

count = np.random.poisson(tc[index]) > 0
tsgroup = nap.TsGroup({i: nap.Ts(timestep[count[:, i]]) for i in range(N)})
epochs = nap.IntervalSet(5, 10)


tuning_curves_1d = nap.compute_tuning_curves(
    tsgroup, feature, bins=61, range=(0, 2 * np.pi), feature_names=["feature"]
)

decoded, proba_feature = nap.decode_bayes(
    tuning_curves=tuning_curves_1d,
    group=tsgroup,
    epochs=epochs,
    bin_size=0.06,
)

fig, (ax1, ax2) = plt.subplots(figsize=(8, 5), nrows=2, ncols=1, sharex=True)
ax1.plot(
    np.linspace(0, len(decoded), len(feature.restrict(epochs))),
    feature.restrict(epochs),
    label="True",
)
ax1.scatter(
    np.linspace(0, len(decoded), len(decoded)),
    decoded,
    label="Decoded",
    c="orange",
)
ax1.legend(
    frameon=False,
    bbox_to_anchor=(1.0, 1.0),
)
ax1.set_xlim(epochs[0, 0], epochs[0, 1])
im = ax2.imshow(proba_feature.values.T, aspect="auto", origin="lower", cmap="viridis")
cbar_ax = fig.add_axes([0.93, 0.1, 0.015, 0.36])
fig.colorbar(im, cax=cbar_ax, label="Probability")
ax2.set_xticks([0, len(decoded)], epochs.values[0])
ax2.set_yticks([])
ax2.set_xlabel("Time (s)", labelpad=-20)
plt.savefig("decode_bayes_1d.pdf", dpi=300, bbox_inches="tight")
plt.close()

dt = 0.1
epochs = nap.IntervalSet(start=0, end=1000, time_units="s")
features = np.vstack((np.cos(np.arange(0, 1000, dt)), np.sin(np.arange(0, 1000, dt)))).T
features = nap.TsdFrame(
    t=np.arange(0, 1000, dt),
    d=features,
    time_units="s",
    time_support=epochs,
    columns=["a", "b"],
)

times = features.as_units("us").index.values
ft = features.values
alpha = np.arctan2(ft[:, 1], ft[:, 0])
bin_centers = np.linspace(-np.pi, np.pi, 12)
kappa = 4.0
ts_group = {}
for i, mu in enumerate(bin_centers):
    weights = np.exp(kappa * np.cos(alpha - mu))  # wrapped Gaussian
    weights /= np.max(weights)  # normalize to 0–1
    mask = weights > 0.5
    ts = times[mask]
    ts_group[i] = nap.Ts(ts, time_units="us")
ts_group = nap.TsGroup(ts_group)

tuning_curves_2d = nap.compute_tuning_curves(
    group=ts_group,
    features=features,  # containing 2 features
    bins=10,
    epochs=epochs,
    range=[(-1.0, 1.0), (-1.0, 1.0)],  # range can be specified for each feature
)

decoded, proba_feature = nap.decode_bayes(
    tuning_curves=tuning_curves_2d,
    group=ts_group,
    epochs=ts_group.time_support,
    bin_size=0.2,
)

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(8, 3), nrows=1, ncols=3, sharey=True)
ax1.plot(features["a"].get(0, 20), label="True")
ax1.scatter(
    decoded["a"].get(0, 20).times(),
    decoded["a"].get(0, 20),
    label="Decoded",
    c="orange",
)
ax1.set_title("Feature a")
ax1.set_xlabel("Time (s)")

ax2.plot(features["b"].get(0, 20), label="True")
ax2.scatter(
    decoded["b"].get(0, 20).times(),
    decoded["b"].get(0, 20),
    label="Decoded",
    c="orange",
)
ax2.set_xlabel("Time (s)")
ax2.set_title("Feature b")

ax3.plot(
    features["a"].get(0, 20),
    features["b"].get(0, 20),
    label="True",
)
ax3.scatter(
    decoded["a"].get(0, 20),
    decoded["b"].get(0, 20),
    label="Decoded",
    c="orange",
)
ax3.set_title("Combined")
plt.savefig("decode_template_2d.pdf", dpi=300, bbox_inches="tight")
