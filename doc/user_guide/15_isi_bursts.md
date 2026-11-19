---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Inter-spike interval (ISI) & burst detection

```{contents}
:local:
:depth: 3
```

```{code-cell} ipython3
:tags: [hide-cell]
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
```

Let's generate some data. Here we have two neurons recorded together. Both fire bursts of 3 to 8
spikes on top of a background of single spikes. In the first neuron the bursts are tight, with
spikes about 6 ms apart, and stand out clearly from the background. In the second they are
looser and more variable, with spikes about 30 ms apart, and the background is denser, so
intervals within and between bursts overlap. We can group them in a TsGroup.

```{code-cell} ipython3
np.random.seed(0)
duration = 600

def bursty_unit(background_rate, burst_rate, burst_isi, isi_spread):
    """Poisson background, plus bursts of 3 to 8 spikes with log-normal ISIs."""
    background = np.random.uniform(0, duration, int(background_rate * duration))
    starts = np.random.uniform(0, duration, int(burst_rate * duration))
    bursts = [
        start
        + np.cumsum(
            np.random.lognormal(np.log(burst_isi), isi_spread, np.random.randint(3, 9))
        )
        for start in starts
    ]
    return nap.Ts(t=np.sort(np.concatenate([background] + bursts)))

ts1 = bursty_unit(background_rate=2, burst_rate=0.3, burst_isi=0.006, isi_spread=0.3)
ts2 = bursty_unit(background_rate=4, burst_rate=0.3, burst_isi=0.03, isi_spread=0.8)

epoch = nap.IntervalSet(start=0, end=duration)
ts_group = nap.TsGroup({0: ts1, 1: ts2}, time_support=epoch)
ts_group
```

## ISI distribution

The interspike interval distribution, computed with [`compute_isi_distribution`](pynapple.process.isi.compute_isi_distribution), shows how the time differences between subsequent spikes (events) are distributed.
The input can be any object with timestamps. Passing `epochs` restricts the computation to the given epochs.
The output will be a dataframe with the bin centres as index and containing the corresponding ISI counts per unit.

```{code-cell} ipython3
isi_distribution = nap.compute_isi_distribution(
    data=ts_group, bins=10, epochs=epoch
    )
isi_distribution
```

```{code-cell} ipython3
:tags: [hide-input]
for col in isi_distribution.columns:
    plt.bar(
        isi_distribution.index,
        isi_distribution[col].values,
        width=np.diff(isi_distribution.index).mean(),
        alpha=0.5,
        label=col,
        align='center',
        edgecolor='none'
    )
plt.xlabel("ISI (s)")
plt.ylabel("Count")
plt.legend(title="Unit")
plt.show()
```

The `bins` argument allows for choosing either the number of bins as an integer or the bin edges as an array directly:
```{code-cell} ipython3
isi_distribution = nap.compute_isi_distribution(
    data=ts_group, bins=np.linspace(0, 3, 10), epochs=epoch
    )
isi_distribution
```

```{code-cell} ipython3
:tags: [hide-input]
for col in isi_distribution.columns:
    plt.bar(
        isi_distribution.index,
        isi_distribution[col].values,
        width=np.diff(isi_distribution.index).mean(),
        alpha=0.5,
        label=col,
        align='center',
        edgecolor='none'
    )
plt.xlabel("ISI (s)")
plt.ylabel("Count")
plt.legend(title="Unit")
plt.show()
```

The `log_scale` argument allows for applying the log-transform to the ISIs:
```{code-cell} ipython3
isi_distribution = nap.compute_isi_distribution(
    data=ts_group, bins=10, log_scale=True, epochs=epoch
    )
isi_distribution
```

```{code-cell} ipython3
:tags: [hide-input]
# finer bins for plotting (with log_scale=True, bin edges are given in log units)
isi_distribution = nap.compute_isi_distribution(
    data=ts_group, bins=np.linspace(-10, 2.5, 54), log_scale=True, epochs=epoch
    )
plt.figure(figsize=(10, 3))
for col in isi_distribution.columns:
    plt.bar(
        isi_distribution.index,
        isi_distribution[col].values,
        width=np.diff(isi_distribution.index).mean(),
        alpha=0.5,
        label=col,
        align='center',
        edgecolor='none'
    )
plt.xlim(-10, 2.5)
plt.xlabel("log ISI (s)")
plt.ylabel("Count")
plt.legend(title="Unit")
plt.tight_layout()
plt.show()
```

## Burst detection

A burst is a group of spikes much closer to each other than to the spikes around them.
[`detect_spike_bursts`](pynapple.process.isi.detect_spike_bursts) finds them and returns an
`IntervalSet` from the first to the last spike of each burst, with the number of spikes as
metadata.

There is no single agreed definition of a burst, and many detection methods exist. Cotterill &
Eglen (2018) [^1] compared eight of them and chose MaxInterval, logISI, Poisson surprise and CMA
as the best performing. Two of these are available in pynapple:

- `poisson_surprise` (default) compares each group of spikes with the mean firing rate of its
  own spike train, so it needs little tuning per neuron or class of neurons. The review
  recommends it when intra- and inter-burst intervals are not clearly separated.
- `max_interval` is the review's first choice. Its five thresholds are easy to interpret, but
  they have to be tuned by hand to the data.

A third method, `instantaneous_rate`, is not part of the review. It is a single threshold on the
instantaneous firing rate, the burst definition used in the birdsong field [^3].

logISI is not included: it was developed for neuronal cultures, and according to the review it
is most effective when intra- and inter-burst intervals are clearly distinct.

With every method, a `TsGroup` is processed unit by unit and returns a dictionary of `IntervalSet`
keyed by unit. Bursts are detected within each epoch of the time support, so a burst never spans
a gap in it.

### Poisson surprise

The `poisson_surprise` method (Legendy & Salcman, 1985 [^2]) scores a group of spikes by how
unlikely it would be if the spike train were a Poisson process at its mean firing rate
$\lambda$. For $N$ spikes spanning a duration $T$, the surprise is

$$
S = -\log P, \qquad P = \sum_{n=N-1}^{\infty} e^{-\lambda T} \frac{(\lambda T)^n}{n!}
$$

where $P$ is the probability that a Poisson process with rate $\lambda$ fires at least $N - 1$
spikes within a time $T$ of a given spike.

A surprise maximisation algorithm then finds the bursts with the highest surprise along the
spike train [^1]:

1. Initial bursts are identified as any sequence of three consecutive spikes separated by ISIs
   which are less than half of the mean ISI of the spike train.
2. Spikes are added to the end of each initial burst while this increases the surprise. Up to 10
   spikes ahead are tried, but none beyond an ISI longer than `max_isi_burst`.
3. Spikes are removed from the beginning of the burst while this increases the surprise, which
   leaves the sequence of spikes with the maximum surprise.
4. Bursts with a surprise below `min_surprise` are discarded.

For details, see [^1] and [^2].

| Argument | Default | Meaning |
|---|---|---|
| `min_surprise` | 5 | Bursts with a lower surprise are discarded. |
| `max_isi_burst` | `None` | ISI, in seconds, beyond which no more spikes are looked at when extending a burst. `None` uses twice the mean ISI. |

The same call, with the default parameters, finds bursts in both units:

```{code-cell} ipython3
bursts_ps = nap.detect_spike_bursts(ts_group)
bursts_ps[1]
```

The method assumes Poisson firing, which real spike trains only approximate. The review notes
that it tends to overestimate bursting in spike trains with sparse or no bursting, and advises
screening the detected bursts for outliers in duration. Here too, some bursts run on through the
background spikes and last for seconds:

```{code-cell} ipython3
{unit: float(np.max(bursts.end - bursts.start)) for unit, bursts in bursts_ps.items()}
```

A lower `max_isi_burst` keeps bursts from being extended that far:

```{code-cell} ipython3
bursts_ps_capped = nap.detect_spike_bursts(ts_group, max_isi_burst=0.1)
{unit: float(np.max(bursts.end - bursts.start)) for unit, bursts in bursts_ps_capped.items()}
```

### Choosing thresholds from the ISI distribution

The other two methods use fixed thresholds on the ISI. The log-scaled ISI distribution is the
place to start: bursts appear as a separate peak at short ISIs, and the valley after that peak
is a natural threshold. The first unit has such a valley, at about 20 ms:

```{code-cell} ipython3
threshold = 0.02
```

```{code-cell} ipython3
:tags: [hide-input]
isi_distribution = nap.compute_isi_distribution(
    ts_group, bins=np.linspace(-10, 2.5, 54), log_scale=True
)
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
for ax, unit in zip(axes, isi_distribution.columns):
    ax.bar(
        isi_distribution.index,
        isi_distribution[unit].values,
        width=np.diff(isi_distribution.index).mean(),
        color=f"C{unit}",
        edgecolor="none",
    )
    ax.axvline(np.log(threshold), color="red", label="threshold")
    ax.set_ylabel("Count")
    ax.set_title(f"Unit {unit}")
axes[0].legend()
axes[1].set_xlim(-10, 2.5)
axes[1].set_xlabel("log ISI (s)")
plt.tight_layout();
```

The second unit has no valley: the intervals within and between its bursts overlap into a single
broad peak, so its histogram does not say where to put a threshold. The threshold of the first
unit is applied to both units below, and cuts through the bursts of the second.

### MaxInterval

The `max_interval` method [^1] uses five fixed thresholds, illustrated below.

```{figure} ../_static/burst_max_interval.png
:name: fig-max-interval
:width: 80%

Parameters of the MaxInterval method. From Cotterill & Eglen (2018) [^1], figure 3.
```

| In the figure | Argument | Default (s) | Meaning |
|---|---|---|---|
| max ISI at start of burst | `max_isi_start` | 0.17 | A burst starts at the first of two spikes closer than this. |
| max ISI in burst | `max_isi_burst` | 0.3 | A burst continues while consecutive spikes are no further apart than this. |
| min inter-burst interval | `min_inter_burst_interval` | 0.2 | Bursts closer than this, last spike to first spike, are merged. |
| min burst duration | `min_burst_duration` | 0.01 | Shorter bursts are discarded. |
| min number of spikes in burst | `min_spikes_in_burst` | 3 | Bursts with fewer spikes are discarded. |

The defaults are those of [^1] for cultured neurons. For units with short
bursts they are usually too loose and need to be tuned, here with the threshold read above:

```{code-cell} ipython3
bursts_mi = nap.detect_spike_bursts(
    ts_group, method="max_interval", max_isi_start=threshold, max_isi_burst=threshold
)
{unit: len(bursts) for unit, bursts in bursts_mi.items()}
```

### Instantaneous firing rate

The `instantaneous_rate` method defines a burst as a run of spikes whose instantaneous firing
rate (1 / ISI) stays above `rate_threshold`. This is the burst definition used in the
birdsong field, with 125 Hz as the default (see Leonardo & Fee, 2005 [^3]). Runs with fewer than
`min_spikes_in_burst` spikes are discarded:

```{code-cell} ipython3
bursts_ir = nap.detect_spike_bursts(
    ts_group, method="instantaneous_rate", rate_threshold=1 / threshold, min_spikes_in_burst=3
)
{unit: len(bursts) for unit, bursts in bursts_ir.items()}
```

### Comparing the methods

One second of the first unit and two seconds of the second, with the bursts each method detects
shaded. In the first unit, the two threshold methods separate the bursts cleanly, while Poisson
surprise with its default parameters joins them through a background spike; the lower
`max_isi_burst` separates them. In the second unit, the threshold of the first unit only catches
the tighter burst, while Poisson surprise finds both.

```{code-cell} ipython3
:tags: [hide-input]
windows = {0: (237, 238), 1: (204.5, 206.5)}
detected = {
    "poisson_surprise": bursts_ps,
    "poisson_surprise, max_isi_burst=0.1": bursts_ps_capped,
    "max_interval": bursts_mi,
    "instantaneous_rate": bursts_ir,
}
fig, axes = plt.subplots(4, 2, figsize=(12, 8), sharex="col")
for row, (label, bursts) in zip(axes, detected.items()):
    for ax, unit in zip(row, ts_group.keys()):
        t_start, t_end = windows[unit]
        ax.vlines(ts_group[unit].get(t_start, t_end).t, 0, 1, color="k", linewidth=1)
        # bursts are drawn whole, the axis limits cut the view
        for start, end in bursts[unit].values:
            ax.axvspan(start, end, color="orange", alpha=0.4)
        ax.set_xlim(t_start, t_end)
        ax.set_yticks([])
        ax.set_title(f"Unit {unit}: {label}", fontsize=13, loc="left")
for ax in axes[-1]:
    ax.set_xlabel("time (s)")
plt.tight_layout();
```

:::{card}
Authors
^^^
[Akseli Ilmanen](https://github.com/Akseli-Ilmanen)

Wolf De Wulf

:::

## References

[^1]: Cotterill, E., & Eglen, S. J. (2018). Burst detection methods. arXiv:1802.01287.
    <https://arxiv.org/abs/1802.01287>
[^2]: Legendy, C. R., & Salcman, M. (1985). Bursts and recurrences of bursts in the spike trains of
    spontaneously active striate cortex neurons. *Journal of Neurophysiology*, 53(4), 926-939.
    <https://doi.org/10.1152/jn.1985.53.4.926>
[^3]: Leonardo, A., & Fee, M. S. (2005). Ensemble coding of vocal control in birdsong. *Journal of
    Neuroscience*, 25(3), 652-661. <https://doi.org/10.1523/JNEUROSCI.3036-04.2005>
