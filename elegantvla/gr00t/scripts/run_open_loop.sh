#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(temporal_skip_resolve_repo_root "$SCRIPT_DIR")

PYTHON_BIN=${PYTHON_BIN:-python3}
GPU=${GPU:-5}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the GR00T checkpoint directory}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/gr00t/open_loop_error/backbone_cache_comparison}
TRAJECTORIES_PER_ENV=${TRAJECTORIES_PER_ENV:-1}
MAX_TRAJECTORY_STEPS=${MAX_TRAJECTORY_STEPS:-40}
SEED=${SEED:-0}
ENV_NAMES=${ENV_NAMES:-}

temporal_skip_setup_python_env "$REPO_ROOT" /tmp/mplconfig_elegantvla_gr00t_open_loop

set -- \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --trajectories-per-env "$TRAJECTORIES_PER_ENV" \
    --max-trajectory-steps "$MAX_TRAJECTORY_STEPS" \
    --seed "$SEED"

if [ -n "$ENV_NAMES" ]; then
    set -- "$@" --env-names $ENV_NAMES
fi
echo "gpu=$GPU"
echo "model_path=$MODEL_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "max_trajectory_steps=$MAX_TRAJECTORY_STEPS"
echo "cpu_threads=omp:$OMP_NUM_THREADS,mkl:$MKL_NUM_THREADS,openblas:$OPENBLAS_NUM_THREADS,numexpr:$NUMEXPR_NUM_THREADS"

CUDA_VISIBLE_DEVICES="$GPU" temporal_skip_exec_file \
    "$PYTHON_BIN" \
    "$REPO_ROOT/elegantvla/gr00t/analysis/open_loop_error/compare_backbone_cache.py" \
    "$@"
