#!/usr/bin/env python3

"""Compare paired full-compute and temporal-skip rollout artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import tyro


@dataclass
class CompareConfig:
    baseline_dir: str
    candidate_dir: str
    warmup_episodes: int = 1
    output_path: str | None = None


def _read_tsv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("env_name", "")), int(row["seed"])


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else float(np.mean(values))


def _delta(candidate: float | None, baseline: float | None) -> dict[str, float | None]:
    if candidate is None or baseline is None:
        return {"baseline": baseline, "candidate": candidate, "delta": None, "delta_pct": None}
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": candidate - baseline,
        "delta_pct": None if baseline == 0.0 else (candidate / baseline - 1.0) * 100.0,
    }


def _trace_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    scheduler_ms = _mean(rows, "scheduler_prepare_ms")
    policy_ms = _mean(rows, "policy_step_ms")
    return {
        "decision_steps": len(rows),
        "loop_step_wall_ms": _mean(rows, "loop_step_wall_ms"),
        "model_decision_ms": (
            None if scheduler_ms is None or policy_ms is None else scheduler_ms + policy_ms
        ),
        "sim_env_step_ms": _mean(rows, "sim_env_step_ms"),
    }


def main(cfg: CompareConfig) -> None:
    baseline_dir = Path(cfg.baseline_dir)
    candidate_dir = Path(cfg.candidate_dir)
    baseline_rows = _read_tsv(baseline_dir / "summary.tsv")
    candidate_rows = _read_tsv(candidate_dir / "summary.tsv")
    baseline_by_key = {_key(row): row for row in baseline_rows}
    candidate_by_key = {_key(row): row for row in candidate_rows}
    warmup = max(int(cfg.warmup_episodes), 0)
    paired_keys = sorted(
        key
        for key in baseline_by_key.keys() & candidate_by_key.keys()
        if int(baseline_by_key[key]["episode"]) >= warmup
    )
    paired_rows: list[dict[str, Any]] = []
    for key in paired_keys:
        baseline = baseline_by_key[key]
        candidate = candidate_by_key[key]
        paired_rows.append(
            {
                "env_name": key[0],
                "seed": key[1],
                "baseline_success": _as_bool(baseline["success"]),
                "candidate_success": _as_bool(candidate["success"]),
                "baseline_steps": int(baseline["decision_steps"]),
                "candidate_steps": int(candidate["decision_steps"]),
                "baseline_control_rollout_wall_ms": float(
                    baseline["control_rollout_wall_ms"]
                ),
                "candidate_control_rollout_wall_ms": float(
                    candidate["control_rollout_wall_ms"]
                ),
            }
        )

    baseline_trace = [
        row
        for row in _read_jsonl(baseline_dir / "decision_trace.jsonl")
        if int(row["episode"]) >= warmup
    ]
    candidate_trace = [
        row
        for row in _read_jsonl(candidate_dir / "decision_trace.jsonl")
        if int(row["episode"]) >= warmup
    ]
    baseline_trace_metrics = _trace_metrics(baseline_trace)
    candidate_trace_metrics = _trace_metrics(candidate_trace)
    baseline_success_rows = [row for row in paired_rows if row["baseline_success"]]
    candidate_success_rows = [row for row in paired_rows if row["candidate_success"]]
    both_success_rows = [
        row
        for row in paired_rows
        if row["baseline_success"] and row["candidate_success"]
    ]
    result = {
        "warmup_episodes": warmup,
        "paired_episodes": len(paired_rows),
        "baseline_success_rate": _mean(paired_rows, "baseline_success"),
        "candidate_success_rate": _mean(paired_rows, "candidate_success"),
        "both_success_episodes": len(both_success_rows),
        "decision_steps": _delta(
            _mean(candidate_success_rows, "candidate_steps"),
            _mean(baseline_success_rows, "baseline_steps"),
        ),
        "both_success_decision_steps": _delta(
            _mean(both_success_rows, "candidate_steps"),
            _mean(both_success_rows, "baseline_steps"),
        ),
        "both_success_control_rollout_wall_ms": _delta(
            _mean(both_success_rows, "candidate_control_rollout_wall_ms"),
            _mean(both_success_rows, "baseline_control_rollout_wall_ms"),
        ),
        "step_weighted_loop_wall_ms": _delta(
            candidate_trace_metrics["loop_step_wall_ms"],
            baseline_trace_metrics["loop_step_wall_ms"],
        ),
        "step_weighted_model_decision_ms": _delta(
            candidate_trace_metrics["model_decision_ms"],
            baseline_trace_metrics["model_decision_ms"],
        ),
        "step_weighted_sim_env_step_ms": _delta(
            candidate_trace_metrics["sim_env_step_ms"],
            baseline_trace_metrics["sim_env_step_ms"],
        ),
        "paired_rows": paired_rows,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if cfg.output_path is not None:
        output_path = Path(cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


def cli_main() -> None:
    main(tyro.cli(CompareConfig))


if __name__ == "__main__":
    cli_main()
