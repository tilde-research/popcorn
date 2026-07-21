# Releasing

Runbook for extracting popcorn from the monorepo and publishing v0.1.0. Maintainers only; contributors never need this.

## 1. Extract the standalone repository

History is preserved with `git filter-repo`, keeping the contributor graph for attribution. The monorepo root `.mailmap` normalizes cluster-internal and duplicate identities — confirm the emails in it (Li Yang's in particular) before running.

```bash
pip install git-filter-repo
git clone --no-local /path/to/kitchen popcorn && cd popcorn
git filter-repo --subdirectory-filter src/popcorn --mailmap /path/to/kitchen/.mailmap
git remote add origin git@github.com:tilde-research/popcorn.git
git push -u origin main
```

The filter promotes `src/popcorn/` to the repository root, which is already a standard src layout: `pyproject.toml`, docs, `tests/`, `scripts/`, and `.github/workflows/` at the root (CI activates on the first push), the package under `src/popcorn/`. Verify with `git shortlog -sne` that no internal hostnames remain.

## 2. Pre-tag checklist

- [ ] `uv run pytest tests -q` green on CPU; GPU smoke tests green on a CUDA machine
- [ ] `scripts/format.sh` leaves no diff; `uv run pyright` clean
- [ ] Full grid on release hardware: `scripts/bench_hardware.py` (stamps `ref_hash`/`impl_hash` fingerprints into the bundled reports and regenerates the README matrix)
- [ ] `uv build` succeeds; wheel contains `py.typed`, `reports/*.jsonl`, `impls/*.cu`
- [ ] Fresh-venv floor check: `uv venv -p 3.11 && uv pip install --torch-backend=cpu 'torch==2.5.*' dist/popcorn-*.whl && python -c "import popcorn.kernels"`
- [ ] LICENSE and NOTICE present; wheel metadata shows `License-Expression: Apache-2.0` and bundles both files under `dist-info/licenses/`
- [ ] Repo public, then switch the README banner to the absolute URL `https://raw.githubusercontent.com/tilde-research/popcorn/main/images/popcorn-banner.png` (PyPI cannot resolve relative paths; raw URLs 404 while the repo is private)
- [ ] Repo public, then enable the docs site: Settings -> Pages -> Source: GitHub Actions, and re-run the `Site` workflow (it builds `site/` with data from `scripts/export_site_data.py` and deploys to `https://tilde-research.github.io/popcorn/`)

## 3. Publish

PyPI uses trusted publishing (no tokens): on PyPI, add a trusted publisher for `tilde-research/popcorn`, workflow `publish.yml`, environment `pypi`, and create the matching `pypi` environment in the GitHub repo settings.

```bash
git tag v0.1.0 && git push origin v0.1.0
```

Create a GitHub release from the tag; the publish workflow builds and uploads to PyPI. Yank-and-fix goes through a patch release, never a force-push.
