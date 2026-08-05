#!/usr/bin/env bash
# Run a command once per backend in a fresh, isolated environment.
#
# Usage:
#   scripts/per_backend.sh COMMAND [ARG ...]
set -euo pipefail
cd "$(dirname "$0")/.."

BACKENDS="${BACKENDS:-popcorn cudnn fla liger quack transformer_engine unsloth}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_BACKEND="${TORCH_BACKEND:-auto}"

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <command> [args...]" >&2
    exit 2
fi

failed=()
for backend in $BACKENDS; do
    echo "::group::$backend"
    echo "--- $backend: rebuilding the environment"
    uv venv --clear --python "$PYTHON_VERSION"
    target=".[$backend]"
    support=()
    # First-party wall attention and NSA reuse FLA utilities.
    if [ "$backend" = "popcorn" ]; then
        target=".[fla]"
    elif [ "$backend" = "liger" ]; then
        # Liger's GRPO adapter imports transformers at call time without declaring it.
        support=("transformers>=4.52.0,<5")
    fi
    # POPCORN_SKIP_REPORTS is deliberately unset: the build hook fetches the pinned reports,
    # and without them every dispatch would resolve to the torch reference.
    uv pip install --quiet "--torch-backend=$TORCH_BACKEND" -e "$target" --group dev "${support[@]}"
    echo "--- $backend: $*"
    scratch="${TMPDIR:-/tmp}/popcorn_per_backend_${backend}_$$"
    rm -rf "$scratch"
    mkdir -p "$scratch"
    if POPCORN_TEST_BACKEND="$backend" \
        TRITON_CACHE_DIR="$scratch/triton" \
        TORCH_EXTENSIONS_DIR="$scratch/extensions" \
        TORCHINDUCTOR_CACHE_DIR="$scratch/inductor" \
        uv run --no-sync "$@"; then
        echo "--- $backend: ok"
    else
        echo "--- $backend: FAILED" >&2
        failed+=("$backend")
    fi
    rm -rf "$scratch"
    echo "::endgroup::"
done

if [ "${#failed[@]}" -gt 0 ]; then
    echo "failed backends: ${failed[*]}" >&2
    exit 1
fi
echo "all backends passed: $BACKENDS"
