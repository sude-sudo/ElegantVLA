#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(temporal_skip_resolve_repo_root "$SCRIPT_DIR")

PYTHON_BIN=${PYTHON_BIN:-python3}
GPU=${GPU:-3}
TASK=${TASK:-simpler_env_google/google_robot_pick_object}
TASK_GROUP=${TASK_GROUP:-}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the GR00T checkpoint directory}
CHECKPOINT_PATH=${CHECKPOINT_PATH:?set CHECKPOINT_PATH to the PPO checkpoint}
EPISODES=${EPISODES:-1}
SEED=${SEED:-0}
LOG_EACH_STEP=${LOG_EACH_STEP:-false}
PPO_DEVICE=${PPO_DEVICE:-cpu}
GR00T_DEVICE=${GR00T_DEVICE:-cuda}
DETERMINISTIC=${DETERMINISTIC:-1}
MAX_BACKBONE_ACTION_LEVEL=${MAX_BACKBONE_ACTION_LEVEL:-}
MAX_ACTION_HEAD_ACTION_LEVEL=${MAX_ACTION_HEAD_ACTION_LEVEL:-}
COMPILE_ACTION_HEAD=${COMPILE_ACTION_HEAD:-1}
COMPILE_MODE=${COMPILE_MODE:-max-autotune}
STATIC_INFERENCE_CACHE=${STATIC_INFERENCE_CACHE:-1}
REWARD_MODE=${REWARD_MODE:-stage1}
STAGE2_EFFICIENCY_PENALTY_MODE=${STAGE2_EFFICIENCY_PENALTY_MODE:-step}
STAGE2_BACKBONE_CONSERVATIVE_PENALTY=${STAGE2_BACKBONE_CONSERVATIVE_PENALTY:-0.05}
STAGE2_ACTION_HEAD_CONSERVATIVE_PENALTY=${STAGE2_ACTION_HEAD_CONSERVATIVE_PENALTY:-0.025}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$REPO_ROOT/outputs/gr00t/rl_eval"}
RUN_NAME=${RUN_NAME:-"$(date +%Y%m%d_%H%M%S)_$(basename "$TASK")"}
OUTPUT_DIR=${OUTPUT_DIR:-"$OUTPUT_ROOT/$RUN_NAME"}
USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-temporal-skip}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-$(basename "$OUTPUT_DIR")}
WANDB_GROUP=${WANDB_GROUP:-}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_TAGS=${WANDB_TAGS:-}
WANDB_DIR=${WANDB_DIR:-}
WANDB_EXTRA_PATHS=${WANDB_EXTRA_PATHS:-}
WANDB_LOG_STEP_HISTORY=${WANDB_LOG_STEP_HISTORY:-1}
WANDB_SAVE_TABLES=${WANDB_SAVE_TABLES:-1}

temporal_skip_setup_python_env "$REPO_ROOT" /tmp/mplconfig_elegantvla_gr00t_eval
mkdir -p "$OUTPUT_DIR"

echo "gpu=$GPU"
echo "task=$TASK"
if [ -n "$TASK_GROUP" ]; then
    echo "task_group=$TASK_GROUP"
fi
echo "model_path=$MODEL_PATH"
echo "checkpoint_path=$CHECKPOINT_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "episodes=$EPISODES"
echo "seed=$SEED"
echo "log_each_step=$LOG_EACH_STEP"
echo "cpu_threads=omp:$OMP_NUM_THREADS,mkl:$MKL_NUM_THREADS,openblas:$OPENBLAS_NUM_THREADS,numexpr:$NUMEXPR_NUM_THREADS"
echo "torchinductor_compile_threads=$TORCHINDUCTOR_COMPILE_THREADS"
echo "deterministic=$DETERMINISTIC"
echo "max_backbone_action_level=${MAX_BACKBONE_ACTION_LEVEL:-none}"
echo "max_action_head_action_level=${MAX_ACTION_HEAD_ACTION_LEVEL:-none}"
echo "compile_action_head=$COMPILE_ACTION_HEAD"
echo "compile_mode=$COMPILE_MODE"
echo "static_inference_cache=$STATIC_INFERENCE_CACHE"
echo "reward_mode=$REWARD_MODE"
if [ "$USE_WANDB" = "1" ]; then
    echo "use_wandb=1"
    echo "wandb_project=$WANDB_PROJECT"
    echo "wandb_run_name=$WANDB_RUN_NAME"
    echo "wandb_mode=$WANDB_MODE"
    if [ -n "$WANDB_EXTRA_PATHS" ]; then
        echo "wandb_extra_paths=$WANDB_EXTRA_PATHS"
    fi
fi

LOG_EACH_STEP_FLAG=
if [ "$LOG_EACH_STEP" = "true" ] || [ "$LOG_EACH_STEP" = "1" ]; then
    LOG_EACH_STEP_FLAG="--log-each-step"
fi

set -- \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --model-path "$MODEL_PATH" \
    --env-name "$TASK" \
    --episodes "$EPISODES" \
    --seed "$SEED" \
    --gr00t-device "$GR00T_DEVICE" \
    --ppo-device "$PPO_DEVICE" \
    --compile-mode "$COMPILE_MODE" \
    --reward-mode "$REWARD_MODE" \
    --stage2-efficiency-penalty-mode "$STAGE2_EFFICIENCY_PENALTY_MODE" \
    --stage2-backbone-conservative-penalty "$STAGE2_BACKBONE_CONSERVATIVE_PENALTY" \
    --stage2-action-head-conservative-penalty "$STAGE2_ACTION_HEAD_CONSERVATIVE_PENALTY" \
    --output-dir "$OUTPUT_DIR"

if [ -n "$TASK_GROUP" ]; then
    set -- "$@" --task-group "$TASK_GROUP"
fi

if [ -n "$LOG_EACH_STEP_FLAG" ]; then
    set -- "$@" "$LOG_EACH_STEP_FLAG"
fi
if [ "$DETERMINISTIC" = "0" ] || [ "$DETERMINISTIC" = "false" ]; then
    set -- "$@" --no-deterministic
fi
if [ -n "$MAX_BACKBONE_ACTION_LEVEL" ]; then
    set -- "$@" --max-backbone-action-level "$MAX_BACKBONE_ACTION_LEVEL"
fi
if [ -n "$MAX_ACTION_HEAD_ACTION_LEVEL" ]; then
    set -- "$@" --max-action-head-action-level "$MAX_ACTION_HEAD_ACTION_LEVEL"
fi
if [ "$COMPILE_ACTION_HEAD" = "0" ] || [ "$COMPILE_ACTION_HEAD" = "false" ]; then
    set -- "$@" --no-compile-action-head
fi
if [ "$STATIC_INFERENCE_CACHE" = "0" ] || [ "$STATIC_INFERENCE_CACHE" = "false" ]; then
    set -- "$@" --no-static-inference-cache
fi
if [ "$USE_WANDB" = "1" ]; then
    set -- "$@" \
        --use-wandb \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run-name "$WANDB_RUN_NAME" \
        --wandb-mode "$WANDB_MODE"
    if [ -n "$WANDB_GROUP" ]; then
        set -- "$@" --wandb-group "$WANDB_GROUP"
    fi
    if [ -n "$WANDB_ENTITY" ]; then
        set -- "$@" --wandb-entity "$WANDB_ENTITY"
    fi
    if [ -n "$WANDB_TAGS" ]; then
        set -- "$@" --wandb-tags "$WANDB_TAGS"
    fi
    if [ -n "$WANDB_DIR" ]; then
        set -- "$@" --wandb-dir "$WANDB_DIR"
    fi
    if [ -n "$WANDB_EXTRA_PATHS" ]; then
        set -- "$@" --wandb-extra-paths "$WANDB_EXTRA_PATHS"
    fi
    if [ "$WANDB_LOG_STEP_HISTORY" = "0" ]; then
        set -- "$@" --no-wandb-log-step-history
    fi
    if [ "$WANDB_SAVE_TABLES" = "0" ]; then
        set -- "$@" --no-wandb-save-tables
    fi
fi

if [ "$USE_WANDB" = "1" ] && [ -z "$WANDB_TAGS" ]; then
    env -u WANDB_TAGS CUDA_VISIBLE_DEVICES="$GPU" \
        temporal_skip_exec_file "$PYTHON_BIN" "$REPO_ROOT/elegantvla/gr00t/entrypoints/eval_rl.py" "$@"
else
    CUDA_VISIBLE_DEVICES="$GPU" \
        temporal_skip_exec_file "$PYTHON_BIN" "$REPO_ROOT/elegantvla/gr00t/entrypoints/eval_rl.py" "$@"
fi
