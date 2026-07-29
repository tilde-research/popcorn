#!/usr/bin/env bash
# Format and lint-fix the repository.
#
# Usage:
#   scripts/format.sh
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff format .
uv run ruff check --fix .
