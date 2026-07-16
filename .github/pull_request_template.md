<!-- The guided flow lives in CONTRIBUTING.md; every box below is explained there. -->

## What

<!-- One paragraph: what this adds or fixes, and why. For backends: op + library. -->

## Type

- [ ] New backend for an existing kernel
- [ ] New kernel (reference + registration)
- [ ] First-party implementation (`src/popcorn/impls/`)
- [ ] Benchmark reports from my hardware
- [ ] Bug fix (dispatcher, harness, or kernel)
- [ ] Docs / CI / tooling

## Hardware

<!-- GPU(s), driver, CUDA version the grid ran on. "None" for docs/CI-only changes. -->

## Results

<!-- For kernels, backends, and reports: paste the matrix row(s) printed by
     `python -m popcorn.bench run <op>` (pass/skip counts, speedups, peak memory). -->

## Checklist

- [ ] `uv run pytest tests -q` green, `scripts/format.sh` leaves no diff
- [ ] Full grid run for every op touched: zero fail, crash, error, or benchmark error
- [ ] `src/popcorn/reports/*.jsonl` rows committed, README matrix and badges regenerated (`scripts/update_readme.py`)
- [ ] Version pins in `declare_backend` and the pyproject extra match what I tested
- [ ] New library: added to the README Acknowledgement list; adapted code credits its origin in the header comment
