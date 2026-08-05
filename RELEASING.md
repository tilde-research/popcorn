# Releasing

Maintainer runbook. The proposed next version is `0.2.0`; choose the final version before editing
`pyproject.toml`.

## 1. Verify the tree

- [ ] `uv run pytest tests -q`
- [ ] `uv run pyright`
- [ ] `scripts/format.sh` leaves no diff
- [ ] `npm test`, `npm run types:check`, and `npm run build` pass in `site/` with Node 22
- [ ] Both example notebooks execute in a clean temporary environment
- [ ] `03_write_a_kernel.ipynb` matches the current registration and `backend=` APIs
- [ ] Every backend passes `tests/kernels` in its isolated environment

## 2. Verify benchmark evidence

Pull the pinned base, run only missing curve evidence, and merge the new run into the audit:

```bash
uv run python -m popcorn.bench pull
uv run python scripts/bench_sweep.py submit --nodes 6 --curves-only --watch
uv run python scripts/check_records.py --release \
  --include logs/sweeps/<run>/cache/v3/reports
```

The release gate requires current fingerprints, a pass and timing for every implementation, at
least two timed points in every eligible named curve, and no new `fail`, `crash`, `error`,
or `bench_error` rows. `oom` and `timeout` are valid measured budget frontiers that prune
dominated cases. Explicitly gated curve regions and documented implementation-wide exceptions
do not masquerade as missing evidence. Historical rows remain in the base dataset.
Run the `cudnn` phase in a CUDA 13-only image; its frontend preflight skips mixed CUDA 12/13 hosts.

Generate derived artifacts only with their owning scripts:

```bash
uv run python scripts/update_readme.py --check
uv run python scripts/update_site.py --check
```

## 3. Verify the package

- [ ] `uv build`
- [ ] The wheel contains `py.typed`, `reports/*.parquet`, `reports/REVISION`, and `impls/*.cu`
- [ ] A clean Python 3.11 environment can install the wheel and import `popcorn.kernels`
- [ ] Wheel metadata contains Apache-2.0 and bundles `LICENSE` and `NOTICE`

## 4. Publish after approval

Stop here until the reports, version, and release are explicitly approved. Then:

1. Publish reports to Hugging Face.
2. Regenerate the README badges and commit the new `reports/REVISION`.
3. Push the reviewed commit and deploy the `Update site` workflow.
4. Create and push `vX.Y.Z`, then create the matching GitHub release.

PyPI trusted publishing is configured for `tilde-research/popcorn`, workflow `pypi.yml`, and
environment `pypi`. The workflow publishes the tag's version. Fix mistakes with a new patch
release, never by replacing a published tag.
