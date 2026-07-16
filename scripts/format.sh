#!/usr/bin/env bash
# Format and lint-fix the package; CI enforces both without fixing.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff format .
uv run ruff check --fix .
