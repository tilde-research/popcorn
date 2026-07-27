# Agent guidance

Popcorn welcomes focused contributions made with coding agents.
[CONTRIBUTING.md](CONTRIBUTING.md) is the source of truth for setup and the complete kernel/backend workflow.
This file contains only the shortest safe path to a reviewable change.

## Start here

1. Read `CONTRIBUTING.md` and the files directly involved in the request.
2. Check `src/popcorn/core/dispatcher.py` before using registration APIs.
3. Check `ISSUES.md` before copying workarounds or treating an upstream failure as new.
4. Keep one concern per change; do not bundle drive-by refactors.

Use `uv`, not ad hoc `pip` installs. If setup or `import popcorn.kernels` fails, report the exact
failure before adding features; do not silently alter pins or bypass registration.

## Repository map

- `src/popcorn/kernels/<op>.py`: PyTorch reference and backend adapters.
- `src/popcorn/impls/`: first-party Triton and CUDA implementations.
- `src/popcorn/core/`: registration, dispatch, constraints, and backend loading.
- `src/popcorn/bench/`: correctness, benchmarking, fitting, and report tooling.
- `src/popcorn/reports/*.jsonl`: generated evidence used by dispatch.
- `tests/`: convention, CPU, and GPU-gated tests.

## Non-negotiable rules

- The PyTorch reference defines semantics. Do not weaken it, tolerances, test inputs, or the harness.
- Reuse established patterns before adding an abstraction, dimension, tag, dependency, or public option.
- Backend adapters map arguments only and load optional libraries lazily through `source=`.
- Shape validity comes from benchmark reports. Do not use removed `supports=` or `test_shapes=` arguments.
- Express dtype/value restrictions with narrowed annotations; use predicates only for runtime poison avoidance.
- Never hide a fallback, swallow an error, truncate inputs, or claim validation that was not run.
- Never hand-edit report JSONL or generated README badges; use the benchmark and update scripts.
- Adapted code requires a license-compatible MIT/BSD/Apache source and a provenance header.
- Ask before changing dependencies, public signatures, dispatch policy, CI, or release configuration.
- Do not commit or push unless explicitly requested.

## Verification

Start with the narrowest relevant test, then run the applicable handoff checks:

```bash
uv run pytest tests -q
uv run pyright
scripts/format.sh
uv run python -m popcorn.bench run <op> --backend <name>
```

Kernel/backend changes require the full relevant hardware grid with zero `fail`, `crash`, `error`,
or `bench_error` rows before performance claims. Record GPU, driver, CUDA/ROCm, Torch, Python,
backend version, dtype, shapes, gradient mode, and commands. State clearly when GPU checks were not run.

## Handoff

Use `.github/pull_request_template.md`. Explain what changed and why, list exact checks run and
checks not run, include benchmark matrix rows when applicable, and preserve unresolved uncertainty.
If blocked, leave a minimal reproduction, findings, and the smallest concrete next step.
