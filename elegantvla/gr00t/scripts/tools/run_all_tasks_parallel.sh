#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/../common.sh"
REPO_ROOT=$(temporal_skip_resolve_repo_root "$SCRIPT_DIR")

PYTHON_BIN=${PYTHON_BIN:-python3}
GOOGLE_MODEL=${GOOGLE_MODEL:?set GOOGLE_MODEL to the Google GR00T checkpoint}
WIDOWX_MODEL=${WIDOWX_MODEL:?set WIDOWX_MODEL to the WidowX GR00T checkpoint}
SLOT_GPUS=${SLOT_GPUS:-"4 4 5 5 6 6 7 7"}

EPISODES=${EPISODES:-50}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-300}
N_ACTION_STEPS=${N_ACTION_STEPS:-1}
STEPS_PER_RENDER=${STEPS_PER_RENDER:-2}

OUTPUT_ROOT=${OUTPUT_ROOT:-"$REPO_ROOT/outputs/gr00t"}
RUN_NAME=${RUN_NAME:-"all_tasks_parallel_$(date +%Y%m%d_%H%M%S)"}
RUN_DIR=${RUN_DIR:-"$OUTPUT_ROOT/$RUN_NAME"}

temporal_skip_setup_python_env "$REPO_ROOT" /tmp/mplconfig_elegantvla_gr00t

mkdir -p "$RUN_DIR/tasks" "$RUN_DIR/workers"

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
printf '%s\n' "$TASKS" | while IFS= read -r env_name; do
    [ -n "$env_name" ] || continue
    printf '%s\n' "$env_name" >> "$RUN_DIR/workers/slot_${slot_index}.tasks"
    slot_index=$(( (slot_index + 1) % slot_count ))
done

run_task() {
    gpu="$1"
    env_name="$2"
    task_slug=$(printf '%s' "$env_name" | tr '/' '_')
    task_dir="$RUN_DIR/tasks/$task_slug"
    mkdir -p "$task_dir"
    log_path="$task_dir/run.log"

    case "$env_name" in
        simpler_env_google/*) model_path="$GOOGLE_MODEL" ;;
        *) model_path="$WIDOWX_MODEL" ;;
    esac

    echo "[$(date +%H:%M:%S)] start  gpu=$gpu task=$env_name"
    if CUDA_VISIBLE_DEVICES="$gpu" \
        "$PYTHON_BIN" -u -m elegantvla.gr00t.entrypoints.evaluate \
        --model-path "$model_path" \
        --env-name "$env_name" \
        --episodes "$EPISODES" \
        --max-episode-steps "$MAX_EPISODE_STEPS" \
        --n-action-steps "$N_ACTION_STEPS" \
        --steps-per-render "$STEPS_PER_RENDER" \
        --device cuda \
        --output-dir "$task_dir" \
        >"$log_path" 2>&1; then
        echo "[$(date +%H:%M:%S)] finish gpu=$gpu task=$env_name"
    else
        echo "[$(date +%H:%M:%S)] fail   gpu=$gpu task=$env_name"
    fi
}

run_worker() {
    slot="$1"
    gpu="$2"
    task_file="$RUN_DIR/workers/slot_${slot}.tasks"
    while IFS= read -r env_name; do
        [ -n "$env_name" ] || continue
        run_task "$gpu" "$env_name"
    done < "$task_file"
}

echo "run dir: $RUN_DIR"
echo "episodes per task: $EPISODES"
echo "slot gpus: $SLOT_GPUS"

slot=0
worker_pids=""
for gpu in $SLOT_GPUS; do
    (run_worker "$slot" "$gpu") >"$RUN_DIR/workers/slot_${slot}_gpu${gpu}.log" 2>&1 &
    worker_pids="$worker_pids $!"
    slot=$((slot + 1))
done

for pid in $worker_pids; do
    wait "$pid"
done

"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
rows = []

for task_dir in sorted((run_dir / "tasks").glob("*")):
    if not task_dir.is_dir():
        continue
    aggregate_path = task_dir / "aggregate.json"
    if not aggregate_path.exists():
        rows.append(
            {
                "task": task_dir.name,
                "status": "ERROR",
                "success_rate": "",
                "mean_return": "",
                "mean_decision_steps": "",
                "mean_decision_e2e_ms": "",
                "mean_scheduler_prepare_ms": "",
                "mean_policy_step_ms": "",
            }
        )
        continue
    aggregate = json.loads(aggregate_path.read_text())
    rows.append(
        {
            "task": task_dir.name,
            "status": "OK",
            "success_rate": aggregate.get("success_rate", ""),
            "mean_return": aggregate.get("mean_return", ""),
            "mean_decision_steps": aggregate.get("mean_decision_steps", ""),
            "mean_decision_e2e_ms": aggregate.get("mean_decision_e2e_ms", ""),
            "mean_scheduler_prepare_ms": aggregate.get("mean_scheduler_prepare_ms", ""),
            "mean_policy_step_ms": aggregate.get("mean_policy_step_ms", ""),
        }
    )

summary_path = run_dir / "summary.tsv"
with summary_path.open("w") as f:
    f.write(
        "task\tstatus\tsuccess_rate\tmean_return\tmean_decision_steps\tmean_decision_e2e_ms\tmean_scheduler_prepare_ms\tmean_policy_step_ms\n"
    )
    for row in rows:
        f.write(
            f"{row['task']}\t{row['status']}\t{row['success_rate']}\t{row['mean_return']}\t"
            f"{row['mean_decision_steps']}\t{row['mean_decision_e2e_ms']}\t{row['mean_scheduler_prepare_ms']}\t"
            f"{row['mean_policy_step_ms']}\n"
        )

ok_count = sum(row["status"] == "OK" for row in rows)
print(f"summary: {ok_count}/{len(rows)} finished")
print(f"summary file: {summary_path}")
for row in rows:
    print(
        "\t".join(
            str(row[key])
            for key in (
                "task",
                "status",
                "success_rate",
                "mean_return",
                "mean_decision_steps",
                "mean_decision_e2e_ms",
                "mean_scheduler_prepare_ms",
                "mean_policy_step_ms",
            )
        )
    )
PY
