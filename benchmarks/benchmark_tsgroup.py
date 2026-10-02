"""Benchmark `TsGroup` operations (issue #662: sorted-array representation).

`TsGroup` stores every unit's timestamps in one sorted array instead of one
`Ts` per unit, so whole-group operations run once instead of once per unit,
while building a group (a sort) and selecting a few units out of many (a scan
of every spike) get slower. This script times both sides, using only the public
API so it runs unchanged on any version.

Run on `main` and on the PR branch, saving one and comparing the other:

    python benchmarks/benchmark_tsgroup.py --save main.json     # on main
    python benchmarks/benchmark_tsgroup.py --compare main.json  # on the branch

Each number is the median of repeated runs after numba warm-up. Groups have
2,000 uniformly random spikes per unit over 100 s. NWB loading is timed only
when pynwb is installed (`--no-nwb` skips it).
"""

import argparse
import json
import tempfile
import timeit
import warnings
from pathlib import Path

import numpy as np

import pynapple as nap

DURATION = 100.0
N_SPIKES = 2_000
UNIT_COUNTS = [10, 100, 1000]


def median_ms(fn, repeat=7):
    """Median milliseconds per call, with an adaptive loop count."""
    n = 1
    while timeit.timeit(fn, number=n) < 0.05:
        n *= 5
    return 1e3 * np.median([timeit.timeit(fn, number=n) / n for _ in range(repeat)])


def make_members(n_units, seed=0):
    rng = np.random.default_rng(seed)
    return {i: nap.Ts(np.sort(rng.random(N_SPIKES) * DURATION)) for i in range(n_units)}


def make_tsgroup(n_units, seed=0):
    return nap.TsGroup(
        make_members(n_units, seed), time_support=nap.IntervalSet(0, DURATION)
    )


def fragmented_ep(n_intervals):
    """`n_intervals` equal intervals separated by equal gaps -- a stand-in for a
    very fragmented recording (many short trials / artifact-free windows)."""
    edges = np.linspace(0, DURATION, 2 * n_intervals + 1)
    return nap.IntervalSet(start=edges[0:-1:2], end=edges[1::2])


def bench_vs_units():
    """Every operation against the number of units."""
    ep = fragmented_ep(5000)
    target = nap.Tsd(
        t=np.linspace(0, DURATION, 20_000),
        d=np.random.default_rng(1).random(20_000),
    )
    results = {}
    for n_units in UNIT_COUNTS:
        members = make_members(n_units)
        support = nap.IntervalSet(0, DURATION)
        tsg = nap.TsGroup(members, time_support=support)
        benches = {
            "TsGroup(dict, support)": lambda: nap.TsGroup(
                members, time_support=support
            ),
            "count(0.01)": lambda: tsg.count(0.01),
            "count(ep)": lambda: tsg.count(ep=ep),
            "time_diff()": lambda: tsg.time_diff(),
            "restrict (5000 epochs)": lambda: tsg.restrict(ep),
            "value_from (20k target)": lambda: tsg.value_from(target),
            "to_tsd()": lambda: tsg.to_tsd(),
            "get(20, 60)": lambda: tsg.get(20.0, 60.0),
            "subsample(0.5)": lambda: tsg.subsample(0.5, seed=0),
            "tsg[k]": lambda: tsg[n_units // 2],
            "[tsg[k] for k in tsg]": lambda: [tsg[k] for k in tsg],
            "tsg[5 units]": lambda: tsg[list(range(5))],
        }
        for name, fn in benches.items():
            results.setdefault(name, {})[n_units] = median_ms(fn)
    return "vs number of units (2,000 spikes/unit)", "n_units", results


def bench_count_bin_size():
    tsg = make_tsgroup(100)
    results = {"count(bin_size)": {}}
    for bin_size in [1.0, 0.1, 0.01, 0.001]:
        results["count(bin_size)"][bin_size] = median_ms(lambda: tsg.count(bin_size))
    return "count vs bin size (100 units)", "bin_size", results


def bench_restrict_fragmentation():
    tsg = make_tsgroup(100)
    results = {"restrict(ep)": {}}
    for n_intervals in [1, 100, 1000, 5000]:
        ep = fragmented_ep(n_intervals)
        results["restrict(ep)"][n_intervals] = median_ms(lambda: tsg.restrict(ep))
    return "restrict vs number of epochs (100 units)", "n_epochs", results


def bench_slicing():
    """Selecting units: the one case expected to regress, since a time-sorted
    array has to be scanned in full whatever the selection size."""
    results = {"tsg[5 units] vs pool": {}, "tsg[k units] (pool 1000)": {}}
    for pool in [100, 1000, 5000]:
        tsg = make_tsgroup(pool)
        keys = list(range(5))
        results["tsg[5 units] vs pool"][pool] = median_ms(lambda: tsg[keys])
    tsg = make_tsgroup(1000)
    for k in [5, 100, 500]:
        keys = list(range(k))
        results["tsg[k units] (pool 1000)"][k] = median_ms(lambda: tsg[keys])
    return "slicing", "size", results


def bench_nwb():
    """Loading a units table and the first operation on it (which, with lazy
    units, includes reading the spikes)."""
    pynwb = __import__("pynwb")
    from pynwb.testing.mock.file import mock_NWBFile

    results = {"nwb['units']": {}, "first count(0.1)": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for n_units in UNIT_COUNTS:
            path = Path(tmp) / f"units_{n_units}.nwb"
            nwbfile = mock_NWBFile()
            for ts in make_members(n_units).values():
                nwbfile.add_unit(spike_times=ts.t)
            with pynwb.NWBHDF5IO(path, "w") as io:
                io.write(nwbfile)

            load, first = [], []
            for _ in range(5):
                nwb = nap.load_file(path)
                t0 = timeit.default_timer()
                units = nwb["units"]
                t1 = timeit.default_timer()
                units.count(0.1)
                t2 = timeit.default_timer()
                nwb.close()
                load.append(t1 - t0)
                first.append(t2 - t1)
            results["nwb['units']"][n_units] = 1e3 * np.median(load)
            results["first count(0.1)"][n_units] = 1e3 * np.median(first)
    return "NWB units (2,000 spikes/unit)", "n_units", results


def print_section(title, column, results, reference=None):
    cols = list(next(iter(results.values())).keys())
    width = 22 if reference else 12
    print(f"\n## {title}\n")
    header = f"{'operation':<26}" + "".join(f"{f'{column}={c}':>{width}}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, times in results.items():
        cells = []
        for c, v in times.items():
            ref = (reference or {}).get(name, {}).get(str(c))
            if ref is None:
                cells.append(f"{v:>{width - 3}.3f} ms")
            else:
                cells.append(f"{v:>9.3f} ms ({ref / v:>5.1f}x)".rjust(width))
        print(f"{name:<26}" + "".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save", help="write the results to this JSON file")
    parser.add_argument(
        "--compare",
        help="JSON file from a previous --save run: print the speedup against it "
        "(>1x means faster now)",
    )
    parser.add_argument("--no-nwb", action="store_true", help="skip NWB loading")
    args = parser.parse_args()

    print(f"pynapple {nap.__version__} | numpy {np.__version__}")
    warnings.simplefilter("ignore")

    # warm up numba (compile the paths we time)
    warm = make_tsgroup(5)
    warm.count(0.1)
    warm.count(ep=fragmented_ep(5))
    warm.time_diff()
    warm.restrict(fragmented_ep(5))
    warm.value_from(nap.Tsd(t=np.linspace(0, DURATION, 100), d=np.zeros(100)))
    warm.to_tsd()
    warm[[0, 1]]

    sections = [
        bench_vs_units,
        bench_count_bin_size,
        bench_restrict_fragmentation,
        bench_slicing,
    ]
    if not args.no_nwb:
        try:
            __import__("pynwb")
            sections.append(bench_nwb)
        except ImportError:
            print("pynwb not installed: skipping NWB loading")

    reference = None
    if args.compare:
        reference = json.loads(Path(args.compare).read_text())

    saved = {}
    for section in sections:
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
