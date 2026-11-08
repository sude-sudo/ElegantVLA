#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"
REPO_ROOT=$(temporal_skip_resolve_repo_root "$SCRIPT_DIR")

usage() {
    cat <<'EOF'
Usage:
  sh elegantvla/gr00t/scripts/launch_rl_train.sh [--with-watcher] [--eval-gpu GPU]

Shell-only options:
  --with-watcher           Start checkpoint watcher in a background process.
  --eval-gpu GPU           GPU used by the watcher eval process.
  --eval-root DIR          Root directory for watcher eval outputs.
  --eval-task TASK         Eval task for the watcher. Default: same as TASK.
  --eval-model-path PATH   Eval model path for the watcher. Default: same as MODEL_PATH.
  --eval-episodes N        Episodes per watcher eval run. Default: 1.
  --eval-poll-seconds N    Watcher polling interval in seconds. Default: 60.
  --eval-use-wandb         Enable wandb for watcher eval runs.
  --help                   Show this message.

All other train/eval configuration continues to use environment variables.
EOF
}

PYTHON_BIN=${PYTHON_BIN:-python3}
GPU=${GPU:-3}
TASK=${TASK:-simpler_env_widowx/widowx_spoon_on_towel}
TASK_GROUP=${TASK_GROUP:-}
TRAIN_MODE=${TRAIN_MODE:-step}
REWARD_MODE=${REWARD_MODE:-stage1}
STAGE2_EFFICIENCY_PENALTY_MODE=${STAGE2_EFFICIENCY_PENALTY_MODE:-step}
STAGE2_BACKBONE_CONSERVATIVE_PENALTY=${STAGE2_BACKBONE_CONSERVATIVE_PENALTY:-0.05}
STAGE2_ACTION_HEAD_CONSERVATIVE_PENALTY=${STAGE2_ACTION_HEAD_CONSERVATIVE_PENALTY:-0.025}
MODEL_PATH=${MODEL_PATH:-}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-20000}
TOTAL_EPISODES=${TOTAL_EPISODES:-0}
TOTAL_TASK_ROUNDS=${TOTAL_TASK_ROUNDS:-200}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-1024}
EPISODES_PER_UPDATE=${EPISODES_PER_UPDATE:-0}
EPISODES_PER_TASK_PER_UPDATE=${EPISODES_PER_TASK_PER_UPDATE:-0}
PPO_DEVICE=${PPO_DEVICE:-cpu}
GR00T_DEVICE=${GR00T_DEVICE:-cuda}
SAVE_ROOT=${SAVE_ROOT:-"$REPO_ROOT/outputs/gr00t/rl_train"}
RUN_NAME=${RUN_NAME:-}
SAVE_DIR=${SAVE_DIR:-}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
RESUME_VECNORMALIZE=${RESUME_VECNORMALIZE:-}
TENSORBOARD_LOG=${TENSORBOARD_LOG:-}
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-temporal-skip}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
WANDB_GROUP=${WANDB_GROUP:-}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_TAGS=${WANDB_TAGS:-}
WANDB_DIR=${WANDB_DIR:-}
WANDB_MODEL_SAVE_FREQ=${WANDB_MODEL_SAVE_FREQ:-0}
WANDB_GRADIENT_SAVE_FREQ=${WANDB_GRADIENT_SAVE_FREQ:-0}
WANDB_LOG=${WANDB_LOG:-all}
WANDB_SYNC_TENSORBOARD=${WANDB_SYNC_TENSORBOARD:-1}
WANDB_EXTRA_PATHS=${WANDB_EXTRA_PATHS:-}
WITH_WATCHER=${WITH_WATCHER:-1}
WATCHER_GPU=${WATCHER_GPU:-}
WATCHER_EVAL_ROOT=${WATCHER_EVAL_ROOT:-}
WATCHER_TASK=${WATCHER_TASK:-}
WATCHER_MODEL_PATH=${WATCHER_MODEL_PATH:-}
WATCHER_EPISODES=${WATCHER_EPISODES:-1}
WATCHER_POLL_SECONDS=${WATCHER_POLL_SECONDS:-60}
WATCHER_USE_WANDB=${WATCHER_USE_WANDB:-$USE_WANDB}
WATCHER_WANDB_PROJECT=${WATCHER_WANDB_PROJECT:-$WANDB_PROJECT}
WATCHER_WANDB_RUN_NAME=${WATCHER_WANDB_RUN_NAME:-}
WATCHER_WANDB_GROUP=${WATCHER_WANDB_GROUP:-$WANDB_GROUP}
WATCHER_WANDB_ENTITY=${WATCHER_WANDB_ENTITY:-$WANDB_ENTITY}
WATCHER_WANDB_MODE=${WATCHER_WANDB_MODE:-$WANDB_MODE}
WATCHER_WANDB_TAGS=${WATCHER_WANDB_TAGS:-$WANDB_TAGS}
WATCHER_WANDB_DIR=${WATCHER_WANDB_DIR:-$WANDB_DIR}
WATCHER_WANDB_EXTRA_PATHS=${WATCHER_WANDB_EXTRA_PATHS:-}
WATCHER_WANDB_LOG_STEP_HISTORY=${WATCHER_WANDB_LOG_STEP_HISTORY:-1}
WATCHER_WANDB_SAVE_TABLES=${WATCHER_WANDB_SAVE_TABLES:-1}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --with-watcher)
            WITH_WATCHER=1
            ;;
        --eval-gpu)
            shift
            [ "$#" -gt 0 ] || { echo "missing value for --eval-gpu" >&2; exit 2; }
            WATCHER_GPU=$1
            ;;
        --eval-root)
            shift
            [ "$#" -gt 0 ] || { echo "missing value for --eval-root" >&2; exit 2; }
            WATCHER_EVAL_ROOT=$1
            ;;
        --eval-task)
            shift
            [ "$#" -gt 0 ] || { echo "missing value for --eval-task" >&2; exit 2; }
            WATCHER_TASK=$1
            ;;
        --eval-model-path)
            shift
            [ "$#" -gt 0 ] || { echo "missing value for --eval-model-path" >&2; exit 2; }
            WATCHER_MODEL_PATH=$1
            ;;
        --eval-episodes)
            shift
            [ "$#" -gt 0 ] || { echo "missing value for --eval-episodes" >&2; exit 2; }
            WATCHER_EPISODES=$1
            ;;
        --eval-poll-seconds)
            shift
            [ "$#" -gt 0 ] || { echo "missing value for --eval-poll-seconds" >&2; exit 2; }
            WATCHER_POLL_SECONDS=$1
            ;;
        --eval-use-wandb)
            WATCHER_USE_WANDB=1
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

temporal_skip_setup_python_env "$REPO_ROOT" /tmp/mplconfig_elegantvla_gr00t_train

if [ -z "$MODEL_PATH" ]; then
    case "${TASK_GROUP:-$TASK}" in
        google_all|simpler_env_google/*)
            MODEL_PATH=/var/tmp/isaac-gr00t-node7-server/models/GR00T-N1.6-fractal
            ;;
        *)
            MODEL_PATH=/var/tmp/isaac-gr00t-node7-server/models/GR00T-N1.6-bridge
            ;;
    esac
fi

TASK_COUNT=1
if [ -n "$TASK_GROUP" ]; then
    case "$TASK_GROUP" in
        google_all)
            TASK=simpler_env_google/google_robot_pick_object
            TASK_COUNT=6
            ;;
        widowx_all)
            TASK=simpler_env_widowx/widowx_spoon_on_towel
            TASK_COUNT=7
            ;;
        *)
            echo "unknown TASK_GROUP=$TASK_GROUP" >&2
            exit 2
            ;;
    esac
fi

case "$TRAIN_MODE" in
    step)
        ;;
    episode)
        if [ "$EPISODES_PER_UPDATE" = "0" ] && [ "$EPISODES_PER_TASK_PER_UPDATE" != "0" ]; then
            EPISODES_PER_UPDATE=$((EPISODES_PER_TASK_PER_UPDATE * TASK_COUNT))
        fi
        if [ "$TOTAL_EPISODES" = "0" ]; then
            TOTAL_EPISODES=$((TOTAL_TASK_ROUNDS * TASK_COUNT))
        fi
        if [ "$EPISODES_PER_UPDATE" = "0" ]; then
            echo "TRAIN_MODE=episode requires EPISODES_PER_UPDATE or EPISODES_PER_TASK_PER_UPDATE" >&2
            exit 2
        fi
        if [ "$TOTAL_EPISODES" = "0" ]; then
            echo "TRAIN_MODE=episode requires TOTAL_EPISODES > 0" >&2
            exit 2
        fi
        ;;
    *)
        echo "unknown TRAIN_MODE=$TRAIN_MODE (expected step or episode)" >&2
        exit 2
        ;;
esac

if [ -z "$RUN_NAME" ]; then
    if [ -n "$TASK_GROUP" ]; then
        RUN_NAME="$(date +%Y%m%d_%H%M%S)_${TASK_GROUP}"
    else
        RUN_NAME="$(date +%Y%m%d_%H%M%S)_$(basename "$TASK")"
    fi
fi
if [ -z "$SAVE_DIR" ]; then
    SAVE_DIR="$SAVE_ROOT/$RUN_NAME"
fi
mkdir -p "$SAVE_DIR"
if [ -z "$WANDB_RUN_NAME" ]; then
    WANDB_RUN_NAME=$(basename "$SAVE_DIR")
fi

if [ -z "$WATCHER_GPU" ]; then
    WATCHER_GPU=$GPU
fi
if [ -z "$WATCHER_TASK" ]; then
    WATCHER_TASK=$TASK
fi
if [ -z "$WATCHER_MODEL_PATH" ]; then
    WATCHER_MODEL_PATH=$MODEL_PATH
fi
if [ -z "$WATCHER_EVAL_ROOT" ]; then
    WATCHER_EVAL_ROOT="$SAVE_DIR/eval_watch"
fi

watcher_pid=

cleanup_watcher() {
    if [ -n "${watcher_pid:-}" ] && kill -0 "$watcher_pid" 2>/dev/null; then
        echo "stopping_watcher_pid=$watcher_pid"
        kill "$watcher_pid" 2>/dev/null || true
        wait "$watcher_pid" 2>/dev/null || true
    fi
    watcher_pid=
}

start_watcher() {
    mkdir -p "$WATCHER_EVAL_ROOT"
    if [ "$WATCHER_GPU" = "$GPU" ]; then
        echo "watcher_warning=same_gpu_as_train"
    fi
    TRAIN_DIR="$SAVE_DIR" \
    EVAL_ROOT="$WATCHER_EVAL_ROOT" \
    GPU="$WATCHER_GPU" \
    TASK="$WATCHER_TASK" \
    MODEL_PATH="$WATCHER_MODEL_PATH" \
    EPISODES="$WATCHER_EPISODES" \
    POLL_SECONDS="$WATCHER_POLL_SECONDS" \
    REWARD_MODE="$REWARD_MODE" \
    STAGE2_EFFICIENCY_PENALTY_MODE="$STAGE2_EFFICIENCY_PENALTY_MODE" \
    STAGE2_BACKBONE_CONSERVATIVE_PENALTY="$STAGE2_BACKBONE_CONSERVATIVE_PENALTY" \
    STAGE2_ACTION_HEAD_CONSERVATIVE_PENALTY="$STAGE2_ACTION_HEAD_CONSERVATIVE_PENALTY" \
    USE_WANDB="$WATCHER_USE_WANDB" \
    WANDB_PROJECT="$WATCHER_WANDB_PROJECT" \
    WANDB_RUN_NAME="$WATCHER_WANDB_RUN_NAME" \
    WANDB_GROUP="$WATCHER_WANDB_GROUP" \
    WANDB_ENTITY="$WATCHER_WANDB_ENTITY" \
    WANDB_MODE="$WATCHER_WANDB_MODE" \
    WANDB_TAGS="$WATCHER_WANDB_TAGS" \
    WANDB_DIR="$WATCHER_WANDB_DIR" \
    WANDB_EXTRA_PATHS="$WATCHER_WANDB_EXTRA_PATHS" \
    WANDB_LOG_STEP_HISTORY="$WATCHER_WANDB_LOG_STEP_HISTORY" \
    WANDB_SAVE_TABLES="$WATCHER_WANDB_SAVE_TABLES" \
    sh "$SCRIPT_DIR/tools/watch_rl_checkpoints.sh" &
    watcher_pid=$!
    echo "watcher_pid=$watcher_pid"
}

trap 'cleanup_watcher' EXIT
trap 'cleanup_watcher; exit 130' INT TERM

echo "gpu=$GPU"
echo "task=$TASK"
if [ -n "$TASK_GROUP" ]; then
    echo "task_group=$TASK_GROUP"
fi
echo "train_mode=$TRAIN_MODE"
echo "reward_mode=$REWARD_MODE"
if [ "$REWARD_MODE" = "stage2" ]; then
    echo "stage2_efficiency_penalty_mode=$STAGE2_EFFICIENCY_PENALTY_MODE"
    echo "stage2_backbone_conservative_penalty=$STAGE2_BACKBONE_CONSERVATIVE_PENALTY"
    echo "stage2_action_head_conservative_penalty=$STAGE2_ACTION_HEAD_CONSERVATIVE_PENALTY"
fi
echo "save_dir=$SAVE_DIR"
echo "total_timesteps=$TOTAL_TIMESTEPS"
if [ "$TRAIN_MODE" = "step" ]; then
    echo "checkpoint_every=$CHECKPOINT_EVERY"
fi
if [ "$TRAIN_MODE" = "episode" ]; then
    echo "total_episodes=$TOTAL_EPISODES"
    echo "episodes_per_update=$EPISODES_PER_UPDATE"
    echo "total_task_rounds=$TOTAL_TASK_ROUNDS"
    if [ "$EPISODES_PER_TASK_PER_UPDATE" != "0" ]; then
        echo "episodes_per_task_per_update=$EPISODES_PER_TASK_PER_UPDATE"
    fi
fi
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "resume_checkpoint=$RESUME_CHECKPOINT"
fi
if [ -n "$RESUME_VECNORMALIZE" ]; then
    echo "resume_vecnormalize=$RESUME_VECNORMALIZE"
fi
if [ "$USE_WANDB" = "1" ]; then
    echo "use_wandb=1"
    echo "wandb_project=$WANDB_PROJECT"
    echo "wandb_run_name=$WANDB_RUN_NAME"
    echo "wandb_mode=$WANDB_MODE"
    if [ -n "$WANDB_EXTRA_PATHS" ]; then
        echo "wandb_extra_paths=$WANDB_EXTRA_PATHS"
    fi
fi
if [ "$WITH_WATCHER" = "1" ]; then
    echo "with_watcher=1"
    echo "watcher_gpu=$WATCHER_GPU"
    echo "watcher_task=$WATCHER_TASK"
    echo "watcher_eval_root=$WATCHER_EVAL_ROOT"
    echo "watcher_episodes=$WATCHER_EPISODES"
    echo "watcher_poll_seconds=$WATCHER_POLL_SECONDS"
fi

set -- \
    --model-path "$MODEL_PATH" \
    --env-name "$TASK" \
    --schedule-mode "$TRAIN_MODE" \
    --reward-mode "$REWARD_MODE" \
    --stage2-efficiency-penalty-mode "$STAGE2_EFFICIENCY_PENALTY_MODE" \
    --stage2-backbone-conservative-penalty "$STAGE2_BACKBONE_CONSERVATIVE_PENALTY" \
    --stage2-action-head-conservative-penalty "$STAGE2_ACTION_HEAD_CONSERVATIVE_PENALTY" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --gr00t-device "$GR00T_DEVICE" \
    --ppo-device "$PPO_DEVICE" \
    --save-dir "$SAVE_DIR"

if [ -n "$TASK_GROUP" ]; then
    set -- "$@" --task-group "$TASK_GROUP"
fi
if [ "$TRAIN_MODE" = "episode" ]; then
    set -- "$@" \
        --total-episodes "$TOTAL_EPISODES" \
        --episodes-per-update "$EPISODES_PER_UPDATE"
else
    set -- "$@" --checkpoint-every "$CHECKPOINT_EVERY"
fi

if [ -n "$RESUME_CHECKPOINT" ]; then
    set -- "$@" --resume-checkpoint "$RESUME_CHECKPOINT"
fi
if [ -n "$RESUME_VECNORMALIZE" ]; then
    set -- "$@" --resume-vecnormalize "$RESUME_VECNORMALIZE"
fi
if [ -n "$TENSORBOARD_LOG" ]; then
    set -- "$@" --tensorboard-log "$TENSORBOARD_LOG"
fi
if [ "$USE_WANDB" = "1" ]; then
    set -- "$@" \
        --use-wandb \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run-name "$WANDB_RUN_NAME" \
        --wandb-mode "$WANDB_MODE" \
        --wandb-log "$WANDB_LOG" \
        --wandb-model-save-freq "$WANDB_MODEL_SAVE_FREQ" \
        --wandb-gradient-save-freq "$WANDB_GRADIENT_SAVE_FREQ"
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
    if [ "$WANDB_SYNC_TENSORBOARD" = "0" ]; then
        set -- "$@" --no-wandb-sync-tensorboard
    fi
fi

if [ "$WITH_WATCHER" = "1" ]; then
    start_watcher
fi

CUDA_VISIBLE_DEVICES="$GPU" \
    temporal_skip_exec_file "$PYTHON_BIN" "$REPO_ROOT/elegantvla/gr00t/entrypoints/train_rl.py" "$@"

if [ -n "${watcher_pid:-}" ]; then
    wait "$watcher_pid"
    watcher_pid=
fi

trap - EXIT INT TERM
