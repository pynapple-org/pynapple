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

# ISI & burst detection

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

Let's generate some data. Here we have two neurons recorded together. We can group them in a TsGroup.

```{code-cell} ipython3
ts1 = nap.Ts(t=np.sort(np.random.uniform(0, 1000, 2000)), time_units="s")
ts2 = nap.Ts(t=np.sort(np.random.uniform(0, 1000, 1000)), time_units="s")
epoch = nap.IntervalSet(start=0, end=1000, time_units="s")
ts_group = nap.TsGroup({0: ts1, 1: ts2}, time_support=epoch)
print(ts_group)
```

## Interspike interval (ISI) distribution

The interspike interval distribution, computed with [`compute_isi_distribution`](pynapple.process.isi.compute_isi_distribution), shows how the time differences between subsequent spikes (events) are distributed.
The input can be any object with timestamps. Passing `epochs` restricts the computation to the given epochs.
The output will be a dataframe with the bin centres as index and containing the corresponding ISI counts per unit.

```{code-cell} ipython3
isi_distribution = nap.compute_isi_distribution(
    data=ts_group, bins=10, epochs=epoch
    )
print(isi_distribution)
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
print(isi_distribution)
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
plt.xlabel("log ISI (s)")
plt.ylabel("Count")
plt.legend(title="Unit")
plt.show()
```

The `log_scale` argument allows for applying the log-transform to the ISIs:
```{code-cell} ipython3
isi_distribution = nap.compute_isi_distribution(
    data=ts_group, bins=10, log_scale=True, epochs=epoch
    )
print(isi_distribution)
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
plt.xlabel("log ISI (s)")
plt.ylabel("Count")
plt.legend(title="Unit")
plt.show()
```

## Burst detection

A burst is a group of spikes much closer to each other than to the spikes around them.
[`detect_spike_bursts`](pynapple.process.isi.detect_spike_bursts) finds them with one of three
methods and returns an `IntervalSet` from the first to the last spike of each burst, with the
number of spikes as metadata.

Let's simulate a neuron firing at a low background rate with occasional bursts of 5 spikes about 10 ms apart:

```{code-cell} ipython3
np.random.seed(0)
duration = 600
background = np.random.uniform(0, duration, 1200)
burst_starts = np.random.uniform(0, duration, 300)
bursts = [s + np.cumsum(np.random.uniform(0.005, 0.015, 5)) for s in burst_starts]
spikes = nap.Ts(t=np.sort(np.concatenate([background] + bursts)))
spikes
```

### logISI

The default method, `log_isi` (Pasquale et al., 2010 [1]), needs no tuning. It reads the threshold
between intra- and inter-burst intervals off the histogram of the interspike intervals on a
logarithmic axis. The threshold is the minimum ($C_{\mathrm{min}_1}$) between the intra-burst peak ($C_{\mathrm{IBP}}$)
and the first well separated inter-burst peak ($C_{p_1}$). For details on the algorithm,
see references [1] and [2]. 

When no clear minimum is found, the published method falls back to
`max_cutoff` as the threshold. It was developed for cultured neurons, whose histograms are usually
bimodal; on in vivo units this fallback produces many spurious bursts, so pynapple returns no
bursts instead and warns. Pass `fallback=True` to reproduce the published behaviour.

```{figure} ../_static/burst_log_isi.png
:name: fig-log-isi
:width: 60%

Log-ISI histogram of a spike train. The dashed vertical line (`maxISI`) is the threshold at
$C_{\mathrm{min}_1}$, separating intra-burst from inter-burst intervals. It is the value returned by
`compute_log_isi_threshold`. From Cotterill & Eglen (2018) [2], figure 4.
```

| Argument | Default | Meaning |
|---|---|---|
| `max_cutoff` | 0.1 s | Largest ISI at which the intra-burst peak may lie. Also the threshold used when no clear minimum between the peaks is found. |
| `min_spikes_in_burst` | 3 | Bursts with fewer spikes are discarded. |
| `fallback` | `False` | Whether to use `max_cutoff` as the threshold when no clear minimum is found, as the published method does. By default such a train returns no bursts, with a warning. |

The threshold can be inspected with
[`compute_log_isi_threshold`](pynapple.process.isi.compute_log_isi_threshold). It is the lower
edge of the lowest bin between the two peaks:

```{code-cell} ipython3
threshold = nap.compute_log_isi_threshold(spikes)
print(threshold)
```

```{code-cell} ipython3
:tags: [hide-input]
# same bin edges as the method: 10 per decade, from 1 ms to 10 s
edges = np.logspace(0, 4, 40) / 1000
isi_distribution = nap.compute_isi_distribution(spikes, bins=edges)
plt.figure(figsize=(10, 3))
plt.bar(edges[:-1], isi_distribution[0], width=np.diff(edges), align="edge", edgecolor="none")
plt.axvline(threshold, color="red", label="threshold")
plt.xscale("log")
plt.xlabel("ISI (s)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout();
```

```{code-cell} ipython3
bursts = nap.detect_spike_bursts(spikes)
bursts
```

```{code-cell} ipython3
:tags: [hide-input]
segment = nap.IntervalSet(101, 105)
plt.figure(figsize=(10, 2))
plt.vlines(spikes.restrict(segment).t, 0, 1, color="k", linewidth=0.5)
for s, e in bursts.intersect(segment).values:
    plt.axvspan(s, e, color="orange", alpha=0.4)
plt.yticks([])
plt.xlabel("time (s)")
plt.title("nap.detect_spike_bursts(spikes)")
plt.tight_layout();
```

### MaxInterval

The `max_interval` method [2] uses five fixed thresholds, illustrated below.

```{figure} ../_static/burst_max_interval.png
:name: fig-max-interval
:width: 80%

Parameters of the MaxInterval method. From Cotterill & Eglen (2018) [2], figure 3.
```

| In the figure | Argument | Default (s) | Meaning |
|---|---|---|---|
| max ISI at start of burst | `max_isi_start` | 0.17 | A burst starts at the first of two spikes closer than this. |
| max ISI in burst | `max_isi_burst` | 0.3 | A burst continues while consecutive spikes are no further apart than this. |
| min inter-burst interval | `min_inter_burst_interval` | 0.2 | Bursts closer than this, last spike to first spike, are merged. |
| min burst duration | `min_burst_duration` | 0.01 | Shorter bursts are discarded. |
| min number of spikes in burst | `min_spikes_in_burst` | 3 | Bursts with fewer spikes are discarded. |

The defaults are those of [2] for cultured neurons. For units with short
bursts they are usually too loose and need to be tuned:

```{code-cell} ipython3
bursts_mi = nap.detect_spike_bursts(
    spikes, method="max_interval", max_isi_start=0.02, max_isi_burst=0.02
)
print(len(bursts), len(bursts_mi))
```

### Instantaneous firing rate

The `instantaneous_rate` method defines a burst as a run of spikes whose instantaneous firing
rate (1 / ISI) stays above `rate_threshold`. This is the burst definition commonly used in the
birdsong field, with 125 Hz as the default (see Leonardo & Fee, 2005 [3]). Runs with fewer than
`min_spikes_in_burst` spikes are discarded:

```{code-cell} ipython3
bursts_ir = nap.detect_spike_bursts(
    spikes, method="instantaneous_rate", rate_threshold=50, min_spikes_in_burst=3
)
print(len(bursts_ir))
```

With every method, a `TsGroup` is processed unit by unit and returns a dictionary of `IntervalSet`
keyed by unit. Bursts are detected within each epoch of the time support, so a burst never spans
a gap in it.

## References

1. Pasquale, V., Martinoia, S., & Chiappalone, M. (2010). A self-adapting approach for the
   detection of bursts and network bursts in neuronal cultures. *Journal of Computational
   Neuroscience*, 29(1-2), 213-229. <https://doi.org/10.1007/s10827-009-0175-1>
2. Cotterill, E., & Eglen, S. J. (2018). Burst detection methods. arXiv:1802.01287.
   <https://arxiv.org/abs/1802.01287>
3. Leonardo, A., & Fee, M. S. (2005). Ensemble coding of vocal control in birdsong. *Journal of
   Neuroscience*, 25(3), 652-661. <https://doi.org/10.1523/JNEUROSCI.3036-04.2005>

:::{card}
Authors
^^^
[Akseli Ilmanen](https://github.com/Akseli-Ilmanen)

Guillaume Viejo

:::
