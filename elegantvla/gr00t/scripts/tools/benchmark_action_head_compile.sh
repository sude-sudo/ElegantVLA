#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/../common.sh"
REPO_ROOT=$(temporal_skip_resolve_repo_root "$SCRIPT_DIR")

PYTHON_BIN=${PYTHON_BIN:-python3}
GPU=${GPU:-3}
TASK=${TASK:-simpler_env_widowx/widowx_spoon_on_towel}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the GR00T checkpoint directory}
EPISODES=${EPISODES:-1}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-300}
N_ACTION_STEPS=${N_ACTION_STEPS:-1}
STEPS_PER_RENDER=${STEPS_PER_RENDER:-2}

OUTPUT_ROOT=${OUTPUT_ROOT:-"$REPO_ROOT/outputs/gr00t"}
RUN_NAME=${RUN_NAME:-"compile_bench_$(date +%Y%m%d_%H%M%S)"}
RUN_DIR=${RUN_DIR:-"$OUTPUT_ROOT/$RUN_NAME"}

temporal_skip_setup_python_env "$REPO_ROOT" /tmp/mplconfig_elegantvla_gr00t

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig_elegantvla_gr00t}

mkdir -p "$RUN_DIR" "$MPLCONFIGDIR"

log_path="$RUN_DIR/patch.log"
trace_path="$RUN_DIR/decision_trace.jsonl"

echo "[$(date +%H:%M:%S)] start  compile-bench task=$TASK"
CUDA_VISIBLE_DEVICES="$GPU" \
"$PYTHON_BIN" -u -m elegantvla.gr00t.entrypoints.evaluate \
    --model-path "$MODEL_PATH" \
    --env-name "$TASK" \
    --episodes "$EPISODES" \
    --max-episode-steps "$MAX_EPISODE_STEPS" \
    --n-action-steps "$N_ACTION_STEPS" \
    --steps-per-render "$STEPS_PER_RENDER" \
    --device cuda \
    --output-dir "$RUN_DIR" \
    >"$log_path" 2>&1
echo "[$(date +%H:%M:%S)] finish compile-bench"

"$PYTHON_BIN" - "$trace_path" "$log_path" <<'PY'
import ast
import json
import pathlib
import statistics
import sys

trace_path = pathlib.Path(sys.argv[1])
log_path = pathlib.Path(sys.argv[2])
policy_step_ms = []
refresh_step_ms = []
reuse_step_ms = []
final_record = None

if trace_path.exists():
    for line in trace_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        ms = float(record.get("policy_step_ms", 0.0))
        policy_step_ms.append(ms)
        if bool(record.get("use_backbone_cache", False)):
            reuse_step_ms.append(ms)
        else:
            refresh_step_ms.append(ms)

for line in log_path.read_text().splitlines():
    line = line.strip()
    if not (line.startswith("{") and line.endswith("}")):
        continue
    try:
        record = ast.literal_eval(line)
    except Exception:
        continue
    if isinstance(record, dict) and "episode" in record and "success" in record:
        final_record = record

def mean(values):
    return 0.0 if not values else statistics.mean(values)

print(
    {
        "timed_steps": len(policy_step_ms),
        "refresh_steps": len(refresh_step_ms),
        "reuse_steps": len(reuse_step_ms),
        "policy_step_ms_mean": mean(policy_step_ms),
        "refresh_policy_step_ms_mean": mean(refresh_step_ms),
        "reuse_policy_step_ms_mean": mean(reuse_step_ms),
        "episode_return": None if final_record is None else final_record.get("return"),
        "decision_steps": None if final_record is None else final_record.get("decision_steps"),
        "success": None if final_record is None else final_record.get("success"),
    }
)
PY
