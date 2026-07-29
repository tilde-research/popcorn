#!/usr/bin/env bash
# Run a command once per backend in a fresh, isolated environment.
#
# Usage:
#   scripts/per_backend.sh COMMAND [ARG ...]
set -euo pipefail
cd "$(dirname "$0")/.."

BACKENDS="${BACKENDS:-fla liger quack unsloth}"
TORCH_BACKEND="${TORCH_BACKEND:-auto}"

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <command> [args...]" >&2
    exit 2
fi

failed=()
for backend in $BACKENDS; do
    echo "::group::$backend"
    echo "--- $backend: rebuilding the environment"
    uv venv --clear
    # POPCORN_SKIP_REPORTS is deliberately unset: the build hook fetches the pinned reports,
    # and without them every dispatch would resolve to the torch reference.
    uv pip install --quiet "--torch-backend=$TORCH_BACKEND" -e ".[$backend]"
    echo "--- $backend: $*"
    if uv run --no-sync "$@"; then
        echo "--- $backend: ok"
    else
        echo "--- $backend: FAILED" >&2
        failed+=("$backend")
    fi
    echo "::endgroup::"
done

if [ "${#failed[@]}" -gt 0 ]; then
    echo "failed backends: ${failed[*]}" >&2
    exit 1
fi
echo "all backends passed: $BACKENDS"
