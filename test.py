import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from scipy.ndimage import gaussian_filter1d
import pandas as pd

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
xr.set_options(display_expand_attrs=False)
group = {
    0: nap.Ts(t=np.sort(np.random.uniform(0, 100, 10))),
    1: nap.Ts(t=np.sort(np.random.uniform(0, 100, 20))),
    2: nap.Ts(t=np.sort(np.random.uniform(0, 100, 30))),
}
tsgroup = nap.TsGroup(group)
dt = 0.01
T = 10
epoch = nap.IntervalSet(start=0, end=T, time_units="s")
features = np.vstack((np.cos(np.arange(0, T, dt)), np.sin(np.arange(0, T, dt)))).T
features = nap.TsdFrame(
    t=np.arange(0, T, dt),
    d=features,
    time_units="s",
    time_support=epoch,
    columns=["x", "y"],
)


# Calcium activity
ft = features.values
alpha = np.arctan2(ft[:, 1], ft[:, 0])
bin_centers = np.linspace(-np.pi, np.pi, 6)
kappa = 4.0
units = []
for i, mu in enumerate(bin_centers):
    units.append(np.exp(kappa * np.cos(alpha - mu)))  # wrapped Gaussian
units = np.stack(units, axis=1)
tsdframe = nap.TsdFrame(t=features.times(), d=units)
tuning_curves_2d = nap.compute_tuning_curves(
    data=tsdframe, features=features, bins=9, feature_names=["x", "y"]
)
tuning_curves_2d
tuning_curves_2d.name = "ΔF/F"
tuning_curves_2d.attrs["unit"] = "a.u."
g = tuning_curves_2d.plot(
    col="unit",
    col_wrap=3,
    figsize=(8, 5),
    add_colorbar=False,  # IMPORTANT: don't add colorbar yet
)
g.fig.set_constrained_layout(True)
mappable = g.axs.flat[0].collections[0]  # likely for pcolormesh or contourf

# Add shared colorbar
cbar = g.fig.colorbar(mappable, ax=g.axs.ravel().tolist(), location="right", shrink=0.8)
cbar.set_label("ΔF/F [a.u.]")
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.savefig("tuning_curves_2d.pdf", dpi=300)
plt.close()


dt = 0.1
epochs = nap.IntervalSet(start=0, end=1000, time_units="s")
features = np.vstack((np.cos(np.arange(0, 1000, dt)), np.sin(np.arange(0, 1000, dt)))).T
features = nap.TsdFrame(
    t=np.arange(0, 1000, dt),
    d=features,
    time_units="s",
    time_support=epochs,
    columns=["x", "y"],
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
    data=ts_group,
    features=features,  # containing 2 features
    bins=9,
    epochs=epochs,
    range=[(-1.0, 1.0), (-1.0, 1.0)],  # range can be specified for each feature
)
decoded, proba_feature = nap.decode_bayes(
    tuning_curves=tuning_curves_2d,
    data=ts_group,
    epochs=epochs,
    bin_size=0.2,
)
fig, (ax1, ax2, ax3) = plt.subplots(
    figsize=(8, 3.5), nrows=1, ncols=3, sharey=True, layout="constrained"
)
ax1.plot(features["x"].get(0, 20), label="True")
ax1.scatter(
    decoded["x"].get(0, 20).times(),
    decoded["x"].get(0, 20),
    label="Decoded",
    c="orange",
)
ax1.set_title("x")
ax1.set_xlabel("Time (s)")

ax2.plot(features["y"].get(0, 20), label="True")
ax2.scatter(
    decoded["y"].get(0, 20).times(),
    decoded["y"].get(0, 20),
    label="Decoded",
    c="orange",
)
ax2.set_xlabel("Time (s)")
ax2.set_title("y")

ax3.plot(
    features["x"].get(0, 20),
    features["y"].get(0, 20),
    label="True",
)
ax3.scatter(
    decoded["x"].get(0, 20),
    decoded["y"].get(0, 20),
    label="Decoded",
    c="orange",
)
ax3.set_title("Combined")
plt.savefig("decode_template_2d.pdf", dpi=300)
plt.close()


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
epochs = nap.IntervalSet(0, 10)
tuning_curves_1d = nap.compute_tuning_curves(
    tsgroup, feature, bins=61, range=(0, 2 * np.pi), feature_names=["Circular feature"]
)

fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
tuning_curves_1d.name = "Firing rate"
tuning_curves_1d.attrs["unit"] = "Hz"
tuning_curves_1d.coords["Circular feature"].attrs["unit"] = "rad"
tuning_curves_1d.plot.line(
    ax=ax,
    x="Circular feature",
    add_legend=False,
)
plt.xticks([0, 2 * np.pi], ["0", "2π"])
plt.xlabel("Circular feature [rad]", labelpad=-16)
plt.savefig("tuning_curves_1d.pdf", dpi=300)

decoded, proba_feature = nap.decode_bayes(
    tuning_curves=tuning_curves_1d,
    data=tsgroup,
    epochs=epochs,
    bin_size=0.06,
)
fig, (ax1, ax2) = plt.subplots(
    figsize=(8, 4), nrows=2, ncols=1, sharex=True, layout="constrained"
)
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
cbar_ax = fig.add_axes([0.8, 0.1, 0.015, 0.41])
fig.colorbar(im, cax=cbar_ax, label="Probability")
ax2.set_xticks([0, len(decoded)], epochs.values[0])
ax2.set_yticks([0, proba_feature.shape[1] - 1], ["0", "2π"])
ax1.set_yticks([0, 2 * np.pi], ["0", "2π"])
ax1.set_ylabel("Circular\nfeature [rad]")
ax2.set_xlabel("Time (s)", labelpad=-20)
ax2.set_ylabel("Circular\nfeature [rad]")
plt.savefig("decode_bayes_1d.pdf", dpi=300)
