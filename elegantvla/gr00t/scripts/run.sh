#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(temporal_skip_resolve_repo_root "$SCRIPT_DIR")
PYTHON_BIN=${PYTHON_BIN:-python3}

case "${1:-}" in
    evaluate|benchmark|train_rl|eval_rl|train_bc)
        entrypoint=$1
        shift
        ;;
    *)
        echo "Usage: $0 {evaluate|benchmark|train_rl|eval_rl|train_bc} [arguments]" >&2
        exit 2
        ;;
esac

temporal_skip_setup_python_env "$REPO_ROOT" /tmp/mplconfig_elegantvla_gr00t
exec "$PYTHON_BIN" -m "elegantvla.gr00t.entrypoints.$entrypoint" "$@"
