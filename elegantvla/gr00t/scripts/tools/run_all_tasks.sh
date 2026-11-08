#!/bin/sh

# Run all 13 SimplerEnv tasks with elegantvla_gr00t.
# Default: Google on GPU 2, WidowX on GPU 3, one episode per task.
# Example:
#   EPISODES=10 GOOGLE_GPU=0 WIDOWX_GPU=1 sh elegantvla/gr00t/scripts/tools/run_all_tasks.sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/../common.sh"
REPO_ROOT=$(temporal_skip_resolve_repo_root "$SCRIPT_DIR")

PYTHON_BIN=${PYTHON_BIN:-python3}
GOOGLE_MODEL=${GOOGLE_MODEL:?set GOOGLE_MODEL to the Google GR00T checkpoint}
WIDOWX_MODEL=${WIDOWX_MODEL:?set WIDOWX_MODEL to the WidowX GR00T checkpoint}

GOOGLE_GPU=${GOOGLE_GPU:-2}
WIDOWX_GPU=${WIDOWX_GPU:-3}

EPISODES=${EPISODES:-1}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-300}
N_ACTION_STEPS=${N_ACTION_STEPS:-1}
STEPS_PER_RENDER=${STEPS_PER_RENDER:-2}

OUTPUT_ROOT=${OUTPUT_ROOT:-"$REPO_ROOT/outputs/gr00t"}
RUN_NAME=${RUN_NAME:-"all_tasks_$(date +%Y%m%d_%H%M%S)"}
RUN_DIR=${RUN_DIR:-"$OUTPUT_ROOT/$RUN_NAME"}

temporal_skip_setup_python_env "$REPO_ROOT" /tmp/mplconfig_elegantvla_gr00t

mkdir -p "$RUN_DIR"

GOOGLE_TASKS='
simpler_env_google/google_robot_pick_coke_can
simpler_env_google/google_robot_pick_object
simpler_env_google/google_robot_move_near
simpler_env_google/google_robot_open_drawer
simpler_env_google/google_robot_close_drawer
simpler_env_google/google_robot_place_in_closed_drawer
'

WIDOWX_TASKS='
simpler_env_widowx/widowx_spoon_on_towel
simpler_env_widowx/widowx_carrot_on_plate
simpler_env_widowx/widowx_stack_cube
simpler_env_widowx/widowx_put_eggplant_in_basket
simpler_env_widowx/widowx_put_eggplant_in_sink
simpler_env_widowx/widowx_open_drawer
simpler_env_widowx/widowx_close_drawer
'

run_task() {
    gpu="$1"
    model_path="$2"
    env_name="$3"
    task_slug=$(printf '%s' "$env_name" | tr '/' '_')
    log_path="$RUN_DIR/$task_slug.log"

    echo "[$(date +%H:%M:%S)] start  $env_name"
    CUDA_VISIBLE_DEVICES="$gpu" \
        "$PYTHON_BIN" -u -m elegantvla.gr00t.entrypoints.evaluate \
        --model-path "$model_path" \
        --env-name "$env_name" \
        --episodes "$EPISODES" \
        --max-episode-steps "$MAX_EPISODE_STEPS" \
        --n-action-steps "$N_ACTION_STEPS" \
        --steps-per-render "$STEPS_PER_RENDER" \
        --device cuda \
        >"$log_path" 2>&1
    echo "[$(date +%H:%M:%S)] finish $env_name"
}

run_group() {
    gpu="$1"
    model_path="$2"
    tasks="$3"

    printf '%s\n' "$tasks" | while IFS= read -r env_name; do
        [ -n "$env_name" ] || continue
        run_task "$gpu" "$model_path" "$env_name"
    done
}

echo "run dir: $RUN_DIR"
echo "google gpu: $GOOGLE_GPU"
echo "widowx gpu: $WIDOWX_GPU"
echo "episodes per task: $EPISODES"

(run_group "$GOOGLE_GPU" "$GOOGLE_MODEL" "$GOOGLE_TASKS") >"$RUN_DIR/google_runner.log" 2>&1 &
google_pid=$!

(run_group "$WIDOWX_GPU" "$WIDOWX_MODEL" "$WIDOWX_TASKS") >"$RUN_DIR/widowx_runner.log" 2>&1 &
widowx_pid=$!

wait "$google_pid"
wait "$widowx_pid"

"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import ast
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
rows = []

for log_path in sorted(run_dir.glob("*.log")):
    if log_path.name in {"google_runner.log", "widowx_runner.log"}:
        continue

    last_record = None
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                record = ast.literal_eval(line)
            except Exception:
                continue
            if isinstance(record, dict) and "episode" in record and "success" in record:
                last_record = record

    if last_record is None:
        rows.append((log_path.stem, "ERROR", "", "", ""))
        continue

    rows.append(
        (
            log_path.stem,
            str(bool(last_record.get("success", False))),
            str(last_record.get("return", "")),
            str(last_record.get("decision_steps", "")),
            str(last_record.get("absorbed_steps", "")),
        )
    )

summary_path = run_dir / "summary.tsv"
with summary_path.open("w") as f:
    f.write("task\tsuccess\treturn\tdecision_steps\tabsorbed_steps\n")
    for row in rows:
        f.write("\t".join(row) + "\n")

success_count = sum(row[1] == "True" for row in rows)
print(f"summary: {success_count}/{len(rows)} success")
print(f"summary file: {summary_path}")
for row in rows:
    print("\t".join(row))
PY
