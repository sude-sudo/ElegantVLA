#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
UPSTREAM_DIR="$REPO_ROOT/third_party/openpi"
PYTHON_BIN=${PYTHON_BIN:-python3}

case "${1:-}" in
    serve|evaluate|benchmark)
        entrypoint=$1
        shift
        ;;
    *)
        echo "Usage: $0 {serve|evaluate|benchmark} [arguments]" >&2
        exit 2
        ;;
esac

export PYTHONPATH="$REPO_ROOT:$UPSTREAM_DIR/src:$UPSTREAM_DIR/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m "elegantvla.openpi.entrypoints.$entrypoint" "$@"
