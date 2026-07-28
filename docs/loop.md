---
title: Optimization loop
description: Iterate on a candidate kernel with crash-isolated evals and a keep/revert log
---

The loop harness turns first-party kernel work into a fixed experiment cycle — edit one
candidate module, run one evaluation, keep or revert, repeat — designed so an autonomous
agent can grind on an op for hours without weakening anything. Grading is the same
comparison the grid harness uses (fp64-triangulated, budget-relative tolerances, forward
and backward), and every case runs in its own subprocess, so a candidate that crashes or
poisons the CUDA context never takes the session down with it.

The loop is tooling, not evidence: it never writes report rows, dispatch never sees its
results, and promotion still goes through the full grid.

## Protocol

```bash
python -m popcorn.bench.loop targets              # 1. pick a target op
git checkout -b loop/<op>                         # 2. branch
git commit -am "exp 1: <hypothesis>"              # 3. commit, then evaluate
python -m popcorn.bench.loop try <op> --tag exp1 --note "<what changed>"
git reset --hard HEAD~1                           # 4. only when the verdict is REVERT
```

The candidate lives in `src/popcorn/impls/<op>_tl.py` (or `_cu.py`): a callable named
exactly `<op>` with the reference signature, autograd handled inside — the same contract
as [first-party kernels](https://github.com/tilde-research/popcorn/blob/main/CONTRIBUTING.md).
`try` imports it directly, so no registration is needed while iterating.

Make one focused change per experiment, commit before every `try`, and let the exit code
drive git: `0` means keep the commit, `1` means reset it. When `status` shows a long
revert streak, move to the next target.

## try

```bash
python -m popcorn.bench.loop try rms_norm --cases 4 --reps 5 --vs liger --tag exp7
```

- **Fixed sample.** `--cases N` draws a deterministic sample of the op's test grid — the
  same cases every run — so totals are comparable across experiments. Timings only
  compete against kept rows with the same sample, device, gradient mode, and reps;
  changing the op's contract changes the sample and resets the baseline.
- **Isolation.** Each case runs in a fresh worker process with a `--timeout`. A crash,
  hang, or poisoned context is recorded for that case and the loop continues.
- **Grading.** Identical to the grid harness: forward and backward against fp64 truth
  across seeded draws, with the reference's own error as the budget. Statuses are
  `pass`, `fail`, `crash`, `oom`, `error`, and `hang`.
- **Verdict.** `KEEP` (exit 0) requires every case to pass and the summed time to beat
  the best previously kept row by at least 1%; the first passing run sets the baseline.
  Anything else is `REVERT` (exit 1) with the reason.
- **Incumbent context.** `--vs <impl>` also times a registered implementation on the same
  inputs, as an advisory column — the number to beat before promotion is worth it.
- `--forward-only` skips backward grading and timing for inference-oriented kernels.

Every run appends one JSON line — timestamp, tag, note, per-case results, totals,
verdict, and a fingerprint of the candidate code — to
`${POPCORN_CACHE_DIR:-~/.cache/popcorn}/v<schema>/loop/<op>.jsonl` (override with `--log`),
where `<schema>` is the current report schema version.
The log is an untracked scratchpad; never commit it.

## targets

```bash
python -m popcorn.bench.loop targets --hardware H100 --top 15
```

Ranks ops by recorded headroom, worst-served first, from the bundled and user report
rows:

| column | meaning |
| --- | --- |
| `recorded impls` | non-torch implementations with any recorded row |
| `tested` | distinct (case, gradient mode) pairs with records |
| `pass%` | share of tested pairs some implementation passes |
| `grad%` | same, restricted to gradient-mode pairs |
| `x best` | median over cases of the best implementation's speedup vs the reference |
| `reg` | registered non-torch implementations |

Ops with no rows, speedups near 1x, or low `grad%` (forward-only incumbents) are the
prime targets. `ISSUES.md` adds context on why an implementation is gated.

## status

```bash
python -m popcorn.bench.loop status rms_norm --last 8
```

Per op: experiments run, keeps, the revert streak since the last keep (the plateau
signal), the best kept total, and the most recent entries.

## Guardrails

- The loop edits exactly one module: the candidate under `src/popcorn/impls/`. The
  reference, the harness, tolerances, and test inputs are read-only — a faster kernel
  that needed a weaker harness is not faster.
- A `KEEP` is a local verdict on a small sample, never a validation claim. Promotion is
  unchanged: register the kernel under the `popcorn` backend, run the full grid to zero
  `fail`/`crash`/`error`, and commit the reports
  ([CONTRIBUTING.md](https://github.com/tilde-research/popcorn/blob/main/CONTRIBUTING.md)).
- Loop workers never touch the report stores, and report rows are fingerprint-stamped,
  so nothing recorded for an intermediate candidate can leak into dispatch.
