# Contributor statement

## AI Assistance Disclosure
<!-- Select exactly ONE of the following: -->
- [ ] **No AI tools were used** in preparing this PR.
- [x] **If AI tools were used**, I have disclosed which ones. I have fully reviewed, understood and verified their output.
<!-- If AI tools were used, provide which tools were used here. -->

Claude (Claude Code) wrote most of the code. I asked it to port the burst detection
routines of two R packages, [burstanalysis](https://github.com/ellesec/burstanalysis)
and [sjemea](https://github.com/sje30/sjemea), to pynapple. I inspected the output
manually and compared it against the R implementations on synthetic spike trains and
on my own extracellular recordings; both reproduce the same bursts, except for the
one deliberate deviation noted below.

## Motivation

Closes #668. pynapple had no burst detection for spike trains. This adds the standard
methods from the burst detection literature (Cotterill & Eglen, 2018) on top of the
existing ISI machinery.

## PR description

- New `nap.detect_spike_bursts(spikes, method=..., **params)` for `Ts` and `TsGroup`,
  returning an `IntervalSet` (or dict of them) with an `n_spikes` metadata column.
  Methods: `log_isi` (Pasquale et al., 2010, default), `max_interval`
  (Cotterill & Eglen, 2018) and `instantaneous_rate` (Leonardo & Fee, 2005).
- New `nap.compute_log_isi_threshold` to inspect the threshold the logISI method reads
  off the log-ISI histogram.
- Bursts are detected within each epoch of the time support, so none spans a gap.
- Docs: new user guide page `doc/user_guide/15_isi_bursts.md` with figures.
  The two figures are reproduced from the Cotterill & Eglen bioRxiv paper; I asked
  the authors and was given permission (https://github.com/sje30/sjemea/issues/6).
- Tests in `tests/test_isi.py`.

One intentional deviation from the R code: when no clear minimum is found between the
intra- and inter-burst peaks, the published logISI method falls back to `max_cutoff` as
the threshold. It was developed for cultured neurons with bimodal histograms; on in vivo
units the fallback yields many spurious bursts. pynapple therefore returns no bursts and
warns by default. `fallback=True` reproduces the published behaviour.

(The R script's `lowess(f = 0.05)` over the log-ISI histogram is not ported: at fewer
than 80 bins, i.e. any realistic spike train, R's lowess returns the input unchanged.)

## Outstanding questions / particular feedback

None.

## Checklist

Affirm that you have done the following:

- [x] I have described the changes in this PR, following the template above.
- [x] I have added any necessary tests.
- [x] I have added any necessary documentation. This includes docstrings, updates to existing files found in `doc/`, or (for large changes) adding new files to the `doc/` folder.

# AI description of the PR (optional)

The module `pynapple/process/isi.py` gains three burst detectors behind one entry
point, `detect_spike_bursts`. All share the same run-finding helpers: find runs of
spikes closer than a start threshold that continue while ISIs stay under a burst
threshold, merge runs closer than a minimum inter-burst interval, and drop runs that
are too short or have too few spikes. MaxInterval exposes all five thresholds. The
instantaneous-rate method is the same machinery with a single ISI limit of
1 / `rate_threshold`. logISI derives its threshold per unit from the log-ISI histogram
(built with the existing `compute_isi_distribution`): the valley between the
intra-burst peak below `max_cutoff` and the first later peak with void parameter `>= 0.7`. When that threshold exceeds `max_cutoff`, cores are found with `max_cutoff` and extended to neighbouring spikes within the threshold, as in the published method.
Each unit of a `TsGroup` is processed independently, and each epoch of the time
support separately.
