#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

TRAIN_DIR=${TRAIN_DIR:?set TRAIN_DIR}
EVAL_ROOT=${EVAL_ROOT:?set EVAL_ROOT}
GPU=${GPU:-0}
TASK=${TASK:-simpler_env_google/google_robot_pick_object}
MODEL_PATH=${MODEL_PATH:-/var/tmp/isaac-gr00t-node7-server/models/GR00T-N1.6-fractal}
EPISODES=${EPISODES:-1}
POLL_SECONDS=${POLL_SECONDS:-60}
REWARD_MODE=${REWARD_MODE:-stage1}
STAGE2_EFFICIENCY_PENALTY_MODE=${STAGE2_EFFICIENCY_PENALTY_MODE:-step}
STAGE2_BACKBONE_CONSERVATIVE_PENALTY=${STAGE2_BACKBONE_CONSERVATIVE_PENALTY:-0.05}
STAGE2_ACTION_HEAD_CONSERVATIVE_PENALTY=${STAGE2_ACTION_HEAD_CONSERVATIVE_PENALTY:-0.025}
USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-temporal-skip}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
WANDB_GROUP=${WANDB_GROUP:-}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_TAGS=${WANDB_TAGS:-}
WANDB_DIR=${WANDB_DIR:-}
WANDB_EXTRA_PATHS=${WANDB_EXTRA_PATHS:-}
WANDB_LOG_STEP_HISTORY=${WANDB_LOG_STEP_HISTORY:-1}
WANDB_SAVE_TABLES=${WANDB_SAVE_TABLES:-1}

mkdir -p "$EVAL_ROOT"

saw_training=false

run_eval() {
    checkpoint_path=$1
    eval_dir=$2
    CHECKPOINT_PATH="$checkpoint_path" \
    GPU="$GPU" \
    TASK="$TASK" \
    MODEL_PATH="$MODEL_PATH" \
    EPISODES="$EPISODES" \
    REWARD_MODE="$REWARD_MODE" \
    STAGE2_EFFICIENCY_PENALTY_MODE="$STAGE2_EFFICIENCY_PENALTY_MODE" \
    STAGE2_BACKBONE_CONSERVATIVE_PENALTY="$STAGE2_BACKBONE_CONSERVATIVE_PENALTY" \
    STAGE2_ACTION_HEAD_CONSERVATIVE_PENALTY="$STAGE2_ACTION_HEAD_CONSERVATIVE_PENALTY" \
    USE_WANDB="$USE_WANDB" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_RUN_NAME="$WANDB_RUN_NAME" \
    WANDB_GROUP="$WANDB_GROUP" \
    WANDB_ENTITY="$WANDB_ENTITY" \
    WANDB_MODE="$WANDB_MODE" \
    WANDB_TAGS="$WANDB_TAGS" \
    WANDB_DIR="$WANDB_DIR" \
    WANDB_EXTRA_PATHS="$WANDB_EXTRA_PATHS" \
    WANDB_LOG_STEP_HISTORY="$WANDB_LOG_STEP_HISTORY" \
    WANDB_SAVE_TABLES="$WANDB_SAVE_TABLES" \
    OUTPUT_DIR="$eval_dir" \
    sh "$SCRIPT_DIR/../launch_rl_eval.sh"
}


print_trace_summary() {
    eval_dir=$1
    checkpoint_name=$2
    python3 - <<PY
import json
from collections import Counter
from pathlib import Path

trace_path = Path("$eval_dir") / "decision_trace.jsonl"
if trace_path.exists():
    rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    decision_rows = [r for r in rows if sum(int(x) for x in r["backbone_action_mask"]) > 1]
    summary = {
        "checkpoint": "$checkpoint_name",
        "steps": len(rows),
        "decision_steps": len(decision_rows),
        "backbone_action_counts": Counter(r["backbone_action_name"] for r in rows),
        "action_head_action_counts": Counter(r["action_head_action_name"] for r in rows),
        "decision_backbone_action_counts": Counter(r["backbone_action_name"] for r in decision_rows),
        "decision_action_head_action_counts": Counter(r["action_head_action_name"] for r in decision_rows),
    }
    print(summary)
PY
}

while true; do
    if pgrep -f "$TRAIN_DIR" >/dev/null 2>&1; then
        saw_training=true
    fi

    found_ckpt=false
    for ckpt in "$TRAIN_DIR"/ppo_schedule_*_steps.zip "$TRAIN_DIR"/ppo_schedule_*_episodes.zip; do
        if [ ! -f "$ckpt" ]; then
            continue
        fi
        found_ckpt=true
        ckpt_name=$(basename "$ckpt" .zip)
        eval_dir="$EVAL_ROOT/$ckpt_name"
        if [ -f "$eval_dir/summary.tsv" ]; then
            continue
        fi

        echo "evaluating $ckpt_name"
        echo "checkpoint_path=$ckpt"
        echo "output_dir=$eval_dir"
        mkdir -p "$eval_dir"
        eval_log="$eval_dir/eval.log"
        if run_eval "$ckpt" "$eval_dir" >"$eval_log" 2>&1; then
            print_trace_summary "$eval_dir" "$ckpt_name" >>"$eval_log" 2>&1 || true
        else
            status=$?
            echo "eval_failed checkpoint=$ckpt_name status=$status log=$eval_log" >&2
        fi
    done

    if [ -f "$TRAIN_DIR/ppo_schedule.zip" ]; then
        final_eval_dir="$EVAL_ROOT/final"
        if [ ! -f "$final_eval_dir/summary.tsv" ]; then
            echo "evaluating final checkpoint"
            mkdir -p "$final_eval_dir"
            final_eval_log="$final_eval_dir/eval.log"
            if ! run_eval "$TRAIN_DIR/ppo_schedule.zip" "$final_eval_dir" >"$final_eval_log" 2>&1; then
                status=$?
                echo "eval_failed checkpoint=final status=$status log=$final_eval_log" >&2
                sleep "$POLL_SECONDS"
                continue
            fi
        fi
        exit 0
    fi

    if [ "$found_ckpt" = false ] && [ "$saw_training" = true ] && ! pgrep -f "$TRAIN_DIR" >/dev/null 2>&1; then
        exit 0
    fi

    sleep "$POLL_SECONDS"
done
