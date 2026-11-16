"""Benchmark core pynapple operations touched by the speedups branch.

Run on the PR branch and on `dev`/`main` to compare:

    python benchmarks/benchmark_core_ops.py

Times object construction, `restrict` (one and many epochs), `get`, and
`count`, at a few data sizes. Numba is warmed up first so steady-state is
measured, and each number is the median of repeated runs.
"""

import timeit

import numpy as np

import pynapple as nap

SIZES = [100_000, 1_000_000, 10_000_000]


def median_ms(fn):
    """Median milliseconds per call, with an adaptive loop count."""
    n = 1
    while timeit.timeit(fn, number=n) < 0.05:
        n *= 5
    return 1e3 * np.median([timeit.timeit(fn, number=n) / n for _ in range(7)])


def make_data(n):
    rng = np.random.default_rng(0)
    t = np.sort(rng.random(n) * n).astype(np.float64)
    return t, rng.random(n)


def many_epochs(t, m):
    edges = np.linspace(t[0], t[-1], 2 * m + 1)
    return nap.IntervalSet(start=edges[0:-1:2], end=edges[1::2])


def main():
    print(f"pynapple {nap.__version__} | numpy {np.__version__}\n")

    # warm up numba (compile the paths we time)
    t, d = make_data(1000)
    ep = nap.IntervalSet(t[0], t[-1])
    nap.Tsd(t=t, d=d, time_support=ep).restrict(ep).count(1.0)

    results = {}
    for n in SIZES:
        t, d = make_data(n)
        # An explicit support covering all timestamps: the common "coverage"
        # case, and the one that exercises the construction fast path (with
        # time_support=None the clip is skipped on every version, hiding it).
        support = nap.IntervalSet(t[0], t[-1])
        tsd = nap.Tsd(t=t, d=d, time_support=support)
        a, b = t[n // 4], t[3 * n // 4]
        one_epoch = nap.IntervalSet(a, b)
        hundred_epochs = many_epochs(t, 100)
        bin_size = (t[-1] - t[0]) / 1000

        benches = {
            "Tsd(t, d, support)": lambda: nap.Tsd(t=t, d=d, time_support=support),
            "restrict (1 epoch)": lambda: tsd.restrict(one_epoch),
            "restrict (100 epochs)": lambda: tsd.restrict(hundred_epochs),
            "get(a, b)": lambda: tsd.get(a, b),
            "count(bin_size)": lambda: tsd.count(bin_size),
        }
        for name, fn in benches.items():
            results.setdefault(name, []).append(median_ms(fn))

    header = f"{'operation':<24}" + "".join(f"{n:>14,}" for n in SIZES)
    print(header)
    print("-" * len(header))
    for name, times in results.items():
        print(f"{name:<24}" + "".join(f"{v:>11.3f} ms" for v in times))


if __name__ == "__main__":
    main()
