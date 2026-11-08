#!/usr/bin/env python3

"""Summarize a paired full-compute versus temporal-reuse evaluation suite."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import tyro

from elegantvla.gr00t.entrypoints._common import write_json, write_tsv


@dataclass
class SuiteSummaryConfig:
    suite_dir: str
    warmup_episodes: int = 1
    compile_outlier_threshold_ms: float = 1000.0


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _is_latency_warmup(row: dict[str, Any], warmup: int) -> bool:
    if "latency_warmup" in row:
        return _as_bool(row["latency_warmup"])
    return int(row["episode"]) < warmup


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = min(max(float(quantile), 0.0), 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _delta(candidate: float | None, baseline: float | None) -> dict[str, float | None]:
    if candidate is None or baseline is None:
        return {"baseline": baseline, "candidate": candidate, "delta": None, "delta_pct": None}
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": candidate - baseline,
        "delta_pct": None if baseline == 0.0 else (candidate / baseline - 1.0) * 100.0,
    }


def _wilson_95(successes: int, trials: int) -> list[float] | None:
    if trials <= 0:
        return None
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _paired_exact_p(baseline_only: int, candidate_only: int) -> float | None:
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return None
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(baseline_only, candidate_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _trace_metrics(
    rows: list[dict[str, Any]],
    compile_outlier_threshold_ms: float,
) -> dict[str, float | int | None]:
    model_values = [
        float(row["scheduler_prepare_ms"]) + float(row["policy_step_ms"])
        for row in rows
    ]
    loop_values = [float(row["loop_step_wall_ms"]) for row in rows]
    no_compile_rows = [
        row
        for row in rows
        if float(row["policy_step_ms"]) < compile_outlier_threshold_ms
    ]
    no_compile_model_values = [
        float(row["scheduler_prepare_ms"]) + float(row["policy_step_ms"])
        for row in no_compile_rows
    ]
    no_compile_loop_values = [float(row["loop_step_wall_ms"]) for row in no_compile_rows]
    return {
        "decision_steps": len(rows),
        "compile_outliers": len(rows) - len(no_compile_rows),
        "loop_step_wall_ms": _mean(loop_values),
        "loop_step_wall_ms_no_compile": _mean(no_compile_loop_values),
        "loop_step_wall_ms_p50": _percentile(no_compile_loop_values, 0.50),
        "loop_step_wall_ms_p95": _percentile(no_compile_loop_values, 0.95),
        "model_decision_ms": _mean(model_values),
        "model_decision_ms_no_compile": _mean(no_compile_model_values),
        "model_decision_ms_p50": _percentile(no_compile_model_values, 0.50),
        "model_decision_ms_p95": _percentile(no_compile_model_values, 0.95),
        "sim_env_step_ms": _mean([float(row["sim_env_step_ms"]) for row in rows]),
    }


def _summarize_task(
    task_dir: Path,
    warmup: int,
    compile_outlier_threshold_ms: float,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    env_name = (task_dir / "task_name.txt").read_text(encoding="utf-8").strip()
    baseline_dir = task_dir / "baseline"
    candidate_dir = task_dir / "method"
    baseline_rows = _read_tsv(baseline_dir / "summary.tsv")
    candidate_rows = _read_tsv(candidate_dir / "summary.tsv")
    baseline_by_seed = {int(row["seed"]): row for row in baseline_rows}
    candidate_by_seed = {int(row["seed"]): row for row in candidate_rows}
    paired_seeds = sorted(baseline_by_seed.keys() & candidate_by_seed.keys())

    paired = [(baseline_by_seed[seed], candidate_by_seed[seed]) for seed in paired_seeds]
    baseline_successes = sum(_as_bool(baseline["success"]) for baseline, _ in paired)
    candidate_successes = sum(_as_bool(candidate["success"]) for _, candidate in paired)
    baseline_only = sum(
        _as_bool(baseline["success"]) and not _as_bool(candidate["success"])
        for baseline, candidate in paired
    )
    candidate_only = sum(
        not _as_bool(baseline["success"]) and _as_bool(candidate["success"])
        for baseline, candidate in paired
    )
    both_success = [
        (baseline, candidate)
        for baseline, candidate in paired
        if _as_bool(baseline["success"]) and _as_bool(candidate["success"])
    ]

    baseline_trace = [
        row
        for row in _read_jsonl(baseline_dir / "decision_trace.jsonl")
        if not _is_latency_warmup(row, warmup)
    ]
    candidate_trace = [
        row
        for row in _read_jsonl(candidate_dir / "decision_trace.jsonl")
        if not _is_latency_warmup(row, warmup)
    ]
    baseline_trace_metrics = _trace_metrics(baseline_trace, compile_outlier_threshold_ms)
    candidate_trace_metrics = _trace_metrics(candidate_trace, compile_outlier_threshold_ms)
    trials = len(paired)
    baseline_rate = baseline_successes / trials if trials else None
    candidate_rate = candidate_successes / trials if trials else None
    row = {
        "task": env_name,
        "paired_seeds": trials,
        "baseline_successes": baseline_successes,
        "candidate_successes": candidate_successes,
        "baseline_success_rate": baseline_rate,
        "candidate_success_rate": candidate_rate,
        "success_delta_pp": None
        if baseline_rate is None or candidate_rate is None
        else (candidate_rate - baseline_rate) * 100.0,
        "baseline_wilson_95": _wilson_95(baseline_successes, trials),
        "candidate_wilson_95": _wilson_95(candidate_successes, trials),
        "baseline_only_success": baseline_only,
        "candidate_only_success": candidate_only,
        "paired_exact_p": _paired_exact_p(baseline_only, candidate_only),
        "both_success_episodes": len(both_success),
        "both_success_decision_steps": _delta(
            _mean([float(candidate["decision_steps"]) for _, candidate in both_success]),
            _mean([float(baseline["decision_steps"]) for baseline, _ in both_success]),
        ),
        "both_success_control_rollout_wall_ms": _delta(
            _mean([float(candidate["control_rollout_wall_ms"]) for _, candidate in both_success]),
            _mean([float(baseline["control_rollout_wall_ms"]) for baseline, _ in both_success]),
        ),
        "step_weighted_loop_wall_ms": _delta(
            candidate_trace_metrics["loop_step_wall_ms"],
            baseline_trace_metrics["loop_step_wall_ms"],
        ),
        "step_weighted_loop_wall_ms_no_compile": _delta(
            candidate_trace_metrics["loop_step_wall_ms_no_compile"],
            baseline_trace_metrics["loop_step_wall_ms_no_compile"],
        ),
        "loop_wall_ms_p50": _delta(
            candidate_trace_metrics["loop_step_wall_ms_p50"],
            baseline_trace_metrics["loop_step_wall_ms_p50"],
        ),
        "loop_wall_ms_p95": _delta(
            candidate_trace_metrics["loop_step_wall_ms_p95"],
            baseline_trace_metrics["loop_step_wall_ms_p95"],
        ),
        "step_weighted_model_decision_ms": _delta(
            candidate_trace_metrics["model_decision_ms"],
            baseline_trace_metrics["model_decision_ms"],
        ),
        "step_weighted_model_decision_ms_no_compile": _delta(
            candidate_trace_metrics["model_decision_ms_no_compile"],
            baseline_trace_metrics["model_decision_ms_no_compile"],
        ),
        "model_decision_ms_p50": _delta(
            candidate_trace_metrics["model_decision_ms_p50"],
            baseline_trace_metrics["model_decision_ms_p50"],
        ),
        "model_decision_ms_p95": _delta(
            candidate_trace_metrics["model_decision_ms_p95"],
            baseline_trace_metrics["model_decision_ms_p95"],
        ),
        "compile_outliers": {
            "baseline": baseline_trace_metrics["compile_outliers"],
            "candidate": candidate_trace_metrics["compile_outliers"],
        },
        "step_weighted_sim_env_step_ms": _delta(
            candidate_trace_metrics["sim_env_step_ms"],
            baseline_trace_metrics["sim_env_step_ms"],
        ),
    }
    pooled = {
        "baseline_trace": baseline_trace,
        "candidate_trace": candidate_trace,
        "both_success": both_success,
    }
    return row, pooled


def main(cfg: SuiteSummaryConfig) -> None:
    suite_dir = Path(cfg.suite_dir)
    warmup = max(int(cfg.warmup_episodes), 0)
    compile_outlier_threshold_ms = float(cfg.compile_outlier_threshold_ms)
    task_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    pooled_baseline_trace: list[dict[str, Any]] = []
    pooled_candidate_trace: list[dict[str, Any]] = []
    pooled_both_success: list[tuple[dict[str, str], dict[str, str]]] = []

    for task_dir in sorted((suite_dir / "tasks").iterdir()):
        if not task_dir.is_dir():
            continue
        try:
            row, pooled = _summarize_task(
                task_dir,
                warmup,
                compile_outlier_threshold_ms,
            )
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
            errors.append({"task_dir": task_dir.name, "error": str(error)})
            continue
        task_rows.append(row)
        pooled_baseline_trace.extend(pooled["baseline_trace"])
        pooled_candidate_trace.extend(pooled["candidate_trace"])
        pooled_both_success.extend(pooled["both_success"])

    total_trials = sum(int(row["paired_seeds"]) for row in task_rows)
    baseline_successes = sum(int(row["baseline_successes"]) for row in task_rows)
    candidate_successes = sum(int(row["candidate_successes"]) for row in task_rows)
    baseline_only = sum(int(row["baseline_only_success"]) for row in task_rows)
    candidate_only = sum(int(row["candidate_only_success"]) for row in task_rows)
    baseline_trace_metrics = _trace_metrics(
        pooled_baseline_trace,
        compile_outlier_threshold_ms,
    )
    candidate_trace_metrics = _trace_metrics(
        pooled_candidate_trace,
        compile_outlier_threshold_ms,
    )
    baseline_macro = _mean([float(row["baseline_success_rate"]) for row in task_rows])
    candidate_macro = _mean([float(row["candidate_success_rate"]) for row in task_rows])
    summary = {
        "suite_dir": str(suite_dir),
        "warmup_episodes_for_latency": warmup,
        "compile_outlier_threshold_ms": compile_outlier_threshold_ms,
        "tasks_completed": len(task_rows),
        "tasks_with_errors": errors,
        "paired_seeds": total_trials,
        "baseline_successes": baseline_successes,
        "candidate_successes": candidate_successes,
        "baseline_micro_success_rate": baseline_successes / total_trials if total_trials else None,
        "candidate_micro_success_rate": candidate_successes / total_trials if total_trials else None,
        "baseline_macro_success_rate": baseline_macro,
        "candidate_macro_success_rate": candidate_macro,
        "baseline_wilson_95": _wilson_95(baseline_successes, total_trials),
        "candidate_wilson_95": _wilson_95(candidate_successes, total_trials),
        "baseline_only_success": baseline_only,
        "candidate_only_success": candidate_only,
        "paired_exact_p": _paired_exact_p(baseline_only, candidate_only),
        "task_success_wins": sum(row["success_delta_pp"] > 0 for row in task_rows),
        "task_success_ties": sum(row["success_delta_pp"] == 0 for row in task_rows),
        "task_success_losses": sum(row["success_delta_pp"] < 0 for row in task_rows),
        "both_success_decision_steps": _delta(
            _mean([float(candidate["decision_steps"]) for _, candidate in pooled_both_success]),
            _mean([float(baseline["decision_steps"]) for baseline, _ in pooled_both_success]),
        ),
        "both_success_control_rollout_wall_ms": _delta(
            _mean([float(candidate["control_rollout_wall_ms"]) for _, candidate in pooled_both_success]),
            _mean([float(baseline["control_rollout_wall_ms"]) for baseline, _ in pooled_both_success]),
        ),
        "step_weighted_loop_wall_ms": _delta(
            candidate_trace_metrics["loop_step_wall_ms"],
            baseline_trace_metrics["loop_step_wall_ms"],
        ),
        "step_weighted_loop_wall_ms_no_compile": _delta(
            candidate_trace_metrics["loop_step_wall_ms_no_compile"],
            baseline_trace_metrics["loop_step_wall_ms_no_compile"],
        ),
        "loop_wall_ms_p50": _delta(
            candidate_trace_metrics["loop_step_wall_ms_p50"],
            baseline_trace_metrics["loop_step_wall_ms_p50"],
        ),
        "loop_wall_ms_p95": _delta(
            candidate_trace_metrics["loop_step_wall_ms_p95"],
            baseline_trace_metrics["loop_step_wall_ms_p95"],
        ),
        "step_weighted_model_decision_ms": _delta(
            candidate_trace_metrics["model_decision_ms"],
            baseline_trace_metrics["model_decision_ms"],
        ),
        "step_weighted_model_decision_ms_no_compile": _delta(
            candidate_trace_metrics["model_decision_ms_no_compile"],
            baseline_trace_metrics["model_decision_ms_no_compile"],
        ),
        "model_decision_ms_p50": _delta(
            candidate_trace_metrics["model_decision_ms_p50"],
            baseline_trace_metrics["model_decision_ms_p50"],
        ),
        "model_decision_ms_p95": _delta(
            candidate_trace_metrics["model_decision_ms_p95"],
            baseline_trace_metrics["model_decision_ms_p95"],
        ),
        "compile_outliers": {
            "baseline": baseline_trace_metrics["compile_outliers"],
            "candidate": candidate_trace_metrics["compile_outliers"],
        },
        "step_weighted_sim_env_step_ms": _delta(
            candidate_trace_metrics["sim_env_step_ms"],
            baseline_trace_metrics["sim_env_step_ms"],
        ),
        "per_task": task_rows,
    }
    write_json(suite_dir / "suite_summary.json", summary)
    write_tsv(
        suite_dir / "per_task_summary.tsv",
        [
            "task",
            "paired_seeds",
            "baseline_successes",
            "candidate_successes",
            "baseline_success_rate",
            "candidate_success_rate",
            "success_delta_pp",
            "baseline_only_success",
            "candidate_only_success",
            "paired_exact_p",
            "both_success_episodes",
            "baseline_steps",
            "candidate_steps",
            "steps_delta_pct",
            "baseline_model_ms",
            "candidate_model_ms",
            "model_delta_pct",
            "baseline_model_ms_no_compile",
            "candidate_model_ms_no_compile",
            "model_no_compile_delta_pct",
            "baseline_model_p50_ms",
            "candidate_model_p50_ms",
            "model_p50_delta_pct",
            "baseline_compile_outliers",
            "candidate_compile_outliers",
            "baseline_loop_ms",
            "candidate_loop_ms",
            "loop_delta_pct",
            "baseline_loop_ms_no_compile",
            "candidate_loop_ms_no_compile",
            "loop_no_compile_delta_pct",
            "baseline_rollout_ms",
            "candidate_rollout_ms",
            "rollout_delta_pct",
        ],
        [
            {
                "task": row["task"],
                "paired_seeds": row["paired_seeds"],
                "baseline_successes": row["baseline_successes"],
                "candidate_successes": row["candidate_successes"],
                "baseline_success_rate": row["baseline_success_rate"],
                "candidate_success_rate": row["candidate_success_rate"],
                "success_delta_pp": row["success_delta_pp"],
                "baseline_only_success": row["baseline_only_success"],
                "candidate_only_success": row["candidate_only_success"],
                "paired_exact_p": row["paired_exact_p"],
                "both_success_episodes": row["both_success_episodes"],
                "baseline_steps": row["both_success_decision_steps"]["baseline"],
                "candidate_steps": row["both_success_decision_steps"]["candidate"],
                "steps_delta_pct": row["both_success_decision_steps"]["delta_pct"],
                "baseline_model_ms": row["step_weighted_model_decision_ms"]["baseline"],
                "candidate_model_ms": row["step_weighted_model_decision_ms"]["candidate"],
                "model_delta_pct": row["step_weighted_model_decision_ms"]["delta_pct"],
                "baseline_model_ms_no_compile": row["step_weighted_model_decision_ms_no_compile"]["baseline"],
                "candidate_model_ms_no_compile": row["step_weighted_model_decision_ms_no_compile"]["candidate"],
                "model_no_compile_delta_pct": row["step_weighted_model_decision_ms_no_compile"]["delta_pct"],
                "baseline_model_p50_ms": row["model_decision_ms_p50"]["baseline"],
                "candidate_model_p50_ms": row["model_decision_ms_p50"]["candidate"],
                "model_p50_delta_pct": row["model_decision_ms_p50"]["delta_pct"],
                "baseline_compile_outliers": row["compile_outliers"]["baseline"],
                "candidate_compile_outliers": row["compile_outliers"]["candidate"],
                "baseline_loop_ms": row["step_weighted_loop_wall_ms"]["baseline"],
                "candidate_loop_ms": row["step_weighted_loop_wall_ms"]["candidate"],
                "loop_delta_pct": row["step_weighted_loop_wall_ms"]["delta_pct"],
                "baseline_loop_ms_no_compile": row["step_weighted_loop_wall_ms_no_compile"]["baseline"],
                "candidate_loop_ms_no_compile": row["step_weighted_loop_wall_ms_no_compile"]["candidate"],
                "loop_no_compile_delta_pct": row["step_weighted_loop_wall_ms_no_compile"]["delta_pct"],
                "baseline_rollout_ms": row["both_success_control_rollout_wall_ms"]["baseline"],
                "candidate_rollout_ms": row["both_success_control_rollout_wall_ms"]["candidate"],
                "rollout_delta_pct": row["both_success_control_rollout_wall_ms"]["delta_pct"],
            }
            for row in task_rows
        ],
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "per_task"}, indent=2))


def cli_main() -> None:
    main(tyro.cli(SuiteSummaryConfig))


if __name__ == "__main__":
    cli_main()
