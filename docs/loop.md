---
title: Optimization loop (Experimental)
description: Experimental workflow for isolated candidate-kernel evaluation
---

> [!WARNING]
> This workflow is experimental. A passing loop result is not validation evidence.

`popcorn.bench.loop` runs repeatable local experiments on first-party kernels. It evaluates a
fixed sample, isolates each case in a subprocess, and records a `KEEP` or `REVERT` verdict.
It does not write benchmark reports or affect dispatch. Promotion still requires the full
validation grid.

## Workflow

```bash
python -m popcorn.bench.loop targets
git checkout -b loop/<op>
# Edit and commit one candidate change.
python -m popcorn.bench.loop try <op> --tag exp1 --note "<what changed>"
python -m popcorn.bench.loop status <op>
```

The candidate is `src/popcorn/impls/<op>_tl.py` or `<op>_cu.py`. It must export a callable
named `<op>`, match the reference signature, and implement autograd where required. See the
[first-party kernel contract](https://github.com/tilde-research/popcorn/blob/main/CONTRIBUTING.md#3b-first-party-kernels).
`try` imports the candidate directly, so registration is not required during development.

Keep each experiment to one committed change. Exit code `0` means keep the change; exit code
`1` means revert it.

## Evaluate

```bash
python -m popcorn.bench.loop try rms_norm --cases 4 --reps 5 --vs liger --tag exp7
```

- **Sample:** `--cases N` selects deterministic cases. Comparisons use kept rows with the
  same sample, device, gradient mode, and repetition count. Contract changes reset the
  baseline.
- **Isolation:** Each case runs in a new process with a timeout. Crashes and hangs are
  recorded without terminating the evaluation.
- **Grading:** Forward and backward results are compared with fp64 truth using the same
  tolerances as the grid harness. Statuses are `pass`, `fail`, `crash`, `oom`, `error`,
  and `hang`.
- **Verdict:** `KEEP` requires every case to pass and total time to improve by at least 1%
  over the best kept result. The first passing run establishes the baseline.
- **Comparison:** `--vs <impl>` times a registered implementation on the same inputs.
- `--forward-only` skips backward grading and timing for inference-oriented kernels.

Each run appends a JSON record with metadata, case results, totals, verdict, and candidate
fingerprint to
`${POPCORN_CACHE_DIR:-~/.cache/popcorn}/v<schema>/loop/<op>.jsonl` (override with `--log`),
where `<schema>` is the report schema version. Do not commit this log.

## Select targets

```bash
python -m popcorn.bench.loop targets --hardware H100 --top 15
```

The command ranks operations by recorded performance headroom:

| column | meaning |
| --- | --- |
| `recorded impls` | Non-Torch implementations with records |
| `tested` | Distinct case and gradient-mode pairs |
| `pass%` | Tested pairs passed by at least one implementation |
| `grad%` | Passed gradient-mode pairs |
| `x best` | Median best speedup over the reference |
| `reg` | Registered non-Torch implementations |

Prioritize operations with no records, speedups near 1x, or low gradient coverage. Check
`ISSUES.md` for known backend restrictions.

## Review history

```bash
python -m popcorn.bench.loop status rms_norm --last 8
```

The status includes experiment and keep counts, the current revert streak, the best kept
time, and recent entries.

## Promotion requirements

- Limit changes to the candidate module. Do not modify the reference, harness, tolerances,
  or test inputs.
- Treat `KEEP` as a local result, not a validation claim. Register the kernel under the
  `popcorn` backend and run the full grid with zero `fail`, `crash`, `error`, or
  `bench_error` results. Publish the rows and update the report revision as described in
  [CONTRIBUTING.md](https://github.com/tilde-research/popcorn/blob/main/CONTRIBUTING.md).
- Loop workers do not write report stores. Intermediate results cannot enter dispatch.
