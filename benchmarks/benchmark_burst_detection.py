"""Benchmark `detect_spike_bursts` (issue #668: python loops against jitted loops).

Every burst detection method walks the spikes of each unit in a loop. This
script times the three methods on a single `Ts`, a medium `TsGroup` and a big
`TsGroup`, using only the public API so it runs unchanged on any version.

Run before and after a change, saving one and comparing the other:

    python benchmarks/benchmark_burst_detection.py --save before.json
    python benchmarks/benchmark_burst_detection.py --compare before.json

The first section times the very first call of each method in the process,
before any warm-up of the burst detection: with jitted loops it includes the
time numba takes to compile them, or to load them from its cache. To time the
compilation itself, point numba to an empty cache:

    NUMBA_CACHE_DIR=$(mktemp -d) python benchmarks/benchmark_burst_detection.py

The other numbers are the median of repeated runs after that first call. Units
fire a 5 Hz Poisson background plus one burst per second of 3 to 8 spikes, 2 to
8 ms apart, over 20 minutes (about 12,600 spikes per unit).
"""

import argparse
import json
import timeit
import warnings
from pathlib import Path

import numpy as np

import pynapple as nap

DURATION = 1200.0
BACKGROUND_RATE = 5.0
BURST_RATE = 1.0
UNIT_COUNTS = [1, 50, 1000]
METHODS = ["poisson_surprise", "max_interval", "instantaneous_rate"]


def median_ms(fn, repeat=5, min_time=0.02, slow=0.5):
    """Median milliseconds per call, over `repeat` runs.

    The function times one call, then sets the number of calls in each run so
    that a run lasts about `min_time` seconds. A call slower than `min_time`
    runs once in each run. A call slower than `slow` seconds runs 3 times in
    all, and the first call counts as a run: this keeps the slow rows short.
    """
    once = timeit.timeit(fn, number=1)
    if once >= slow:
        runs = [once] + [timeit.timeit(fn, number=1) for _ in range(2)]
        return 1e3 * np.median(runs)
    n = max(1, int(np.ceil(min_time / max(once, 1e-9))))
    return 1e3 * np.median([timeit.timeit(fn, number=n) / n for _ in range(repeat)])


def make_spike_times(rng):
    """Spike times of one bursty unit: a Poisson background plus bursts."""
    background = rng.random(rng.poisson(BACKGROUND_RATE * DURATION)) * DURATION
    burst_starts = rng.random(rng.poisson(BURST_RATE * DURATION)) * DURATION
    n_spikes = rng.integers(3, 9, len(burst_starts))
    offsets = np.cumsum(rng.uniform(0.002, 0.008, (len(burst_starts), 8)), axis=1)
    in_burst = np.arange(8) < n_spikes[:, None]
    bursts = (burst_starts[:, None] + offsets)[in_burst]
    times = np.sort(np.concatenate([background, bursts]))
    return times[times < DURATION]


def make_spikes(n_units, seed=0):
    """A `Ts` for one unit, a `TsGroup` for more."""
    rng = np.random.default_rng(seed)
    support = nap.IntervalSet(0, DURATION)
    if n_units == 1:
        return nap.Ts(make_spike_times(rng), time_support=support)
    return nap.TsGroup(
        {i: nap.Ts(make_spike_times(rng)) for i in range(n_units)},
        time_support=support,
    )


def fragmented_ep(n_intervals):
    """`n_intervals` equal intervals separated by equal gaps -- a stand-in for a
    very fragmented recording (many short trials / artifact-free windows)."""
    edges = np.linspace(0, DURATION, 2 * n_intervals + 1)
    return nap.IntervalSet(start=edges[0:-1:2], end=edges[1::2])


def bench_first_call():
    """The first call of each method in this process, on one unit. Must run
    before anything else calls `detect_spike_bursts`."""
    spikes = make_spikes(1)
    results = {}
    for method in METHODS:
        once = timeit.timeit(
            lambda: nap.detect_spike_bursts(spikes, method=method), number=1
        )
        results[method] = {1: 1e3 * once}
    return "first call in the process (one unit)", "n_units", results


def bench_vs_units():
    """Every method against the number of units."""
    results = {}
    for n_units in UNIT_COUNTS:
        spikes = make_spikes(n_units)
        for method in METHODS:
            results.setdefault(method, {})[n_units] = median_ms(
                lambda: nap.detect_spike_bursts(spikes, method=method)
            )
    return "vs number of units (~12,600 spikes/unit)", "n_units", results


def bench_fragmentation():
    """Every method against the number of epochs: bursts are detected within
    each epoch of the time support."""
    spikes = make_spikes(50)
    results = {}
    for n_intervals in [1, 100, 1000]:
        restricted = spikes.restrict(fragmented_ep(n_intervals))
        for method in METHODS:
            results.setdefault(method, {})[n_intervals] = median_ms(
                lambda: nap.detect_spike_bursts(restricted, method=method)
            )
    return "vs number of epochs (50 units)", "n_epochs", results


def print_section(title, column, results, reference=None):
    cols = list(next(iter(results.values())).keys())
    width = 24 if reference else 14
    print(f"\n## {title}\n")
    header = f"{'method':<20}" + "".join(f"{f'{column}={c}':>{width}}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, times in results.items():
        cells = []
        for c, v in times.items():
            ref = (reference or {}).get(name, {}).get(str(c))
            if ref is None:
                cells.append(f"{v:>{width - 3}.3f} ms")
            else:
                cells.append(f"{v:>11.3f} ms ({ref / v:>5.1f}x)".rjust(width))
        print(f"{name:<20}" + "".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save", help="write the results to this JSON file")
    parser.add_argument(
        "--compare",
        help="JSON file from a previous --save run: print the speedup against it "
        "(>1x means faster now)",
    )
    args = parser.parse_args()

    print(f"pynapple {nap.__version__} | numpy {np.__version__}")
    warnings.simplefilter("ignore")

    # warm up the numba functions of the core that burst detection goes
    # through, so that the first-call section only times the burst detection
    warm = make_spikes(3)
    warm.restrict(fragmented_ep(5)).time_diff()
    nap.compute_isi_distribution(warm, bins=10)
    warm[0].restrict(fragmented_ep(5)).time_diff()
    nap.compute_isi_distribution(warm[0], bins=10)

    reference = None
    if args.compare:
        reference = json.loads(Path(args.compare).read_text())

    saved = {}
    for section in [bench_first_call, bench_vs_units, bench_fragmentation]:
        title, column, results = section()
        print_section(title, column, results, (reference or {}).get(title))
        saved[title] = {
            name: {str(c): v for c, v in times.items()}
            for name, times in results.items()
        }

    if args.save:
        Path(args.save).write_text(json.dumps(saved, indent=1))
        print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
