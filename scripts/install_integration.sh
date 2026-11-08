#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${1:-}"
case "$BACKEND" in
    gr00t)
        UPSTREAM_DIR="$ROOT_DIR/third_party/Isaac-GR00T"
        EXPECTED_REVISION=ead52833afbbf4243f8cd5e7664f48a94de03b19
        ;;
    openpi)
        UPSTREAM_DIR="$ROOT_DIR/third_party/openpi"
        EXPECTED_REVISION=c23745b5ad24e98f66967ea795a07b2588ed6c79
        ;;
    *)
        echo "Usage: $0 {gr00t|openpi} [--check]" >&2
        exit 2
        ;;
esac
shift
if (( $# > 1 )) || [[ "${1:-}" != "" && "${1:-}" != "--check" ]]; then
    echo "Only --check is supported." >&2
    exit 2
fi

INTEGRATION_DIR="$ROOT_DIR/elegantvla/$BACKEND/integration"
if [[ ! -f "$UPSTREAM_DIR/.git" || ! -f "$UPSTREAM_DIR/pyproject.toml" ]]; then
    echo "Initialize the upstream repositories: git submodule update --init --recursive" >&2
    exit 1
fi

upstream_git() {
    git -c safe.directory="$UPSTREAM_DIR" -C "$UPSTREAM_DIR" "$@"
}

if [[ "$(upstream_git rev-parse HEAD)" != "$EXPECTED_REVISION" ]]; then
    echo "$BACKEND requires upstream revision $EXPECTED_REVISION." >&2
    exit 1
fi

# Check every destination before copying any integration files.
while IFS= read -r -d '' source; do
    relative="${source#"$INTEGRATION_DIR/"}"
    destination="$UPSTREAM_DIR/$relative"
    if [[ -L "$destination" ]]; then
        echo "Integration destination is a symlink: $destination" >&2
        exit 1
    fi
    if [[ -f "$destination" ]] && cmp -s "$source" "$destination"; then
        continue
    fi
    if upstream_git ls-files --error-unmatch -- "$relative" >/dev/null 2>&1; then
        if ! upstream_git diff --quiet HEAD -- "$relative"; then
            echo "Integration destination has local edits: $destination" >&2
            exit 1
        fi
    elif [[ -e "$destination" ]]; then
        echo "Integration destination already exists: $destination" >&2
        exit 1
    fi
done < <(find "$INTEGRATION_DIR" -type f -print0)

if [[ "${1:-}" == "--check" ]]; then
    echo "$BACKEND integration check passed."
    exit 0
fi

cp -a "$INTEGRATION_DIR/." "$UPSTREAM_DIR/"
echo "ElegantVLA $BACKEND integration installed."
echo "In this backend's Python environment, run: python -m pip install --no-deps -e $ROOT_DIR"
