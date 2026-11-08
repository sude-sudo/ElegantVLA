#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/../common.sh"
REPO_ROOT=$(temporal_skip_resolve_repo_root "$SCRIPT_DIR")

CHECKPOINT_PATH=${CHECKPOINT_PATH:?set CHECKPOINT_PATH}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$REPO_ROOT/outputs/gr00t/rl_eval_multi"}
RUN_NAME=${RUN_NAME:-"resume_final_multi_$(date +%Y%m%d_%H%M%S)"}
RUN_DIR=${RUN_DIR:-"$OUTPUT_ROOT/$RUN_NAME"}
EPISODES=${EPISODES:-10}
SLOT_GPUS=${SLOT_GPUS:-"2 2 2 2 3 3 3 3"}
GOOGLE_MODEL=${GOOGLE_MODEL:?set GOOGLE_MODEL to the Google GR00T checkpoint}
WIDOWX_MODEL=${WIDOWX_MODEL:?set WIDOWX_MODEL to the WidowX GR00T checkpoint}

mkdir -p "$RUN_DIR/workers"

TASKS='
simpler_env_google/google_robot_pick_coke_can
simpler_env_google/google_robot_pick_object
simpler_env_google/google_robot_move_near
simpler_env_google/google_robot_open_drawer
simpler_env_google/google_robot_close_drawer
simpler_env_google/google_robot_place_in_closed_drawer
simpler_env_widowx/widowx_spoon_on_towel
simpler_env_widowx/widowx_carrot_on_plate
simpler_env_widowx/widowx_stack_cube
simpler_env_widowx/widowx_put_eggplant_in_basket
simpler_env_widowx/widowx_put_eggplant_in_sink
simpler_env_widowx/widowx_open_drawer
simpler_env_widowx/widowx_close_drawer
'

slot_count=0
for _gpu in $SLOT_GPUS; do
    : > "$RUN_DIR/workers/slot_${slot_count}.tasks"
    slot_count=$((slot_count + 1))
done

slot_index=0
printf '%s\n' "$TASKS" | while IFS= read -r task; do
    [ -n "$task" ] || continue
    printf '%s\n' "$task" >> "$RUN_DIR/workers/slot_${slot_index}.tasks"
    slot_index=$(( (slot_index + 1) % slot_count ))
done

run_task() {
    gpu="$1"
    task="$2"
    slug=$(printf '%s' "$task" | tr '/' '_')
    out_dir="$RUN_DIR/$slug"
    log_path="$RUN_DIR/${slug}.log"
    case "$task" in
        simpler_env_google/*) model_path="$GOOGLE_MODEL" ;;
        *) model_path="$WIDOWX_MODEL" ;;
    esac
    echo "=== $task ==="
    GPU="$gpu" \
    TASK="$task" \
    MODEL_PATH="$model_path" \
    CHECKPOINT_PATH="$CHECKPOINT_PATH" \
    EPISODES="$EPISODES" \
    OUTPUT_DIR="$out_dir" \
    sh "$SCRIPT_DIR/../launch_rl_eval.sh" >"$log_path" 2>&1
}

run_worker() {
    slot="$1"
    gpu="$2"
    task_file="$RUN_DIR/workers/slot_${slot}.tasks"
    while IFS= read -r task; do
        [ -n "$task" ] || continue
        run_task "$gpu" "$task"
    done < "$task_file"
}

echo "$RUN_DIR"

slot=0
worker_pids=""
for gpu in $SLOT_GPUS; do
    (run_worker "$slot" "$gpu") >"$RUN_DIR/runner_slot_${slot}_gpu${gpu}.log" 2>&1 &
    worker_pids="$worker_pids $!"
    slot=$((slot + 1))
done

for pid in $worker_pids; do
    wait "$pid"
done

echo "done: $RUN_DIR"
