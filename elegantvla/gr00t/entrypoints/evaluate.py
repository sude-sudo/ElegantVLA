#!/usr/bin/env python3

"""Run the chunked temporal-skip env with a schedule policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import tyro

from elegantvla.gr00t.entrypoints._analysis import (
    FullComputeInstrumentedTemporalSkipPolicyWrapper,
    InstrumentedMultiTaskTemporalSkipEnv,
    InstrumentedTemporalSkipEnv,
    InstrumentedTemporalSkipPolicyWrapper,
)
from elegantvla.gr00t.entrypoints._common import append_jsonl, write_json, write_jsonl, write_tsv
from elegantvla.gr00t.entrypoints._runtime import (
    build_temporal_skip_env,
    extract_scheduler_observation,
    require_existing_model_path,
    seed_episode_rng,
)
from elegantvla.gr00t.core.types import ActionHeadAction, BackboneAction


@dataclass
class RunConfig:
    model_path: str
    env_name: str = "simpler_env_google/google_robot_pick_object"
    task_group: str = ""
    episodes: int = 1
    seed: int | None = 0
    device: str = "cuda"
    strict: bool = True
    compile_action_head: bool = True
    compile_mode: str = "max-autotune"
    static_inference_cache: bool = True
    schedule_mode: str = "rule"
    reward_mode: str = "stage1"
    stage2_efficiency_penalty_mode: str = "step"
    stage2_backbone_conservative_penalty: float = 0.05
    stage2_action_head_conservative_penalty: float = 0.025

    max_episode_steps: int = 300
    n_action_steps: int = 1
    steps_per_render: int = 2

    gate_threshold: float = 0.95
    free1_similarity_threshold: float | None = 0.98
    free2_similarity_threshold: float | None = 0.99
    free3_similarity_threshold: float | None = 0.995
    action_head_cache_selection: str = "motion_speed_inner"
    ah_speed_threshold: float = 0.005
    ah_secondary_threshold: float | None = 0.05
    action_head_gripper_speed_smoothing_window: int = 4
    action_head_eef_rotation_speed_weight: float = 1.0
    action_head_gripper_speed_weight: float = 1.0

    log_each_step: bool = False
    latency_warmup_episodes: int = 1
    output_dir: str | None = None
    resume: bool = True


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else float(np.mean(values))


def _episode_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful_rows = [row for row in rows if bool(row.get("success"))]
    failed_rows = [row for row in rows if not bool(row.get("success"))]
    return {
        "episodes": int(len(rows)),
        "success_rate": _mean_metric(rows, "success"),
        "mean_return": _mean_metric(rows, "return"),
        "mean_decision_steps": _mean_metric(rows, "decision_steps"),
        "mean_successful_decision_steps": _mean_metric(
            successful_rows,
            "decision_steps",
        ),
        "mean_failed_decision_steps": _mean_metric(failed_rows, "decision_steps"),
        "mean_reset_wall_ms": _mean_metric(rows, "reset_wall_ms"),
        "mean_control_rollout_wall_ms": _mean_metric(rows, "control_rollout_wall_ms"),
        "mean_episode_wall_ms": _mean_metric(rows, "episode_wall_ms"),
        "mean_loop_step_wall_ms": _mean_metric(rows, "loop_step_wall_ms_mean"),
        "mean_sim_env_step_ms": _mean_metric(rows, "sim_env_step_ms_mean"),
    }


def _trace_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scheduler_ms = _mean_metric(rows, "scheduler_prepare_ms")
    policy_ms = _mean_metric(rows, "policy_step_ms")
    return {
        "decision_steps": int(len(rows)),
        "mean_loop_step_wall_ms": _mean_metric(rows, "loop_step_wall_ms"),
        "mean_scheduler_prepare_ms": scheduler_ms,
        "mean_policy_step_ms": policy_ms,
        "mean_model_decision_ms": (
            None if scheduler_ms is None or policy_ms is None else scheduler_ms + policy_ms
        ),
        "mean_sim_env_step_ms": _mean_metric(rows, "sim_env_step_ms"),
        "mean_execute_total_ms": _mean_metric(rows, "execute_total_ms"),
    }


def _read_jsonl_prefix(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if not isinstance(row, dict):
                break
            rows.append(row)
    return rows


def _completed_episode_prefix(
    rows: list[dict[str, Any]],
    cfg: RunConfig,
) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for expected_episode, row in enumerate(rows[: max(int(cfg.episodes), 0)]):
        expected_seed = None if cfg.seed is None else int(cfg.seed) + expected_episode
        row_seed = row.get("seed")
        if int(row.get("episode", -1)) != expected_episode:
            break
        if row_seed != expected_seed:
            break
        completed.append(row)
    return completed


def _validate_resume_config(path: Path, cfg: RunConfig) -> None:
    if not path.exists():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    current = asdict(cfg)
    changed = {
        key: {"previous": value, "current": current.get(key)}
        for key, value in previous.items()
        if key != "resume" and current.get(key) != value
    }
    if changed:
        raise ValueError(
            "Cannot resume with a different run configuration: "
            + json.dumps(changed, sort_keys=True)
        )


def main(cfg: RunConfig) -> None:
    require_existing_model_path(cfg.model_path)
    schedule_mode = str(cfg.schedule_mode).strip().lower()
    if schedule_mode not in {"rule", "full_compute"}:
        raise ValueError("schedule_mode must be either 'rule' or 'full_compute'.")
    wrapper_cls = (
        FullComputeInstrumentedTemporalSkipPolicyWrapper
        if schedule_mode == "full_compute"
        else InstrumentedTemporalSkipPolicyWrapper
    )
    env = build_temporal_skip_env(
        cfg,
        device_attr="device",
        wrapper_cls=wrapper_cls,
        env_cls=InstrumentedTemporalSkipEnv,
        multi_task_env_cls=InstrumentedMultiTaskTemporalSkipEnv,
    )
    schedule_policy = env.reward_policy
    output_dir = None if cfg.output_dir is None else Path(cfg.output_dir)
    trace_path = None
    episode_checkpoint_path = None
    episode_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    start_episode = 0
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_config_path = output_dir / "run_config.json"
        trace_path = output_dir / "decision_trace.jsonl"
        episode_checkpoint_path = output_dir / "episodes.jsonl"
        if cfg.resume:
            _validate_resume_config(run_config_path, cfg)
            episode_rows = _completed_episode_prefix(
                _read_jsonl_prefix(episode_checkpoint_path),
                cfg,
            )
            start_episode = len(episode_rows)
            trace_rows = [
                row
                for row in _read_jsonl_prefix(trace_path)
                if 0 <= int(row.get("episode", -1)) < start_episode
            ]
        write_json(run_config_path, asdict(cfg))
        write_jsonl(episode_checkpoint_path, episode_rows)
        write_jsonl(trace_path, trace_rows)

    configured_warmup_episodes = max(int(cfg.latency_warmup_episodes), 0)
    resumed_segment = start_episode > 0

    try:
        for episode_idx in range(start_episode, cfg.episodes):
            episode_seed = None if cfg.seed is None else cfg.seed + episode_idx
            latency_warmup = bool(
                episode_idx < configured_warmup_episodes
                or (resumed_segment and episode_idx == start_episode)
            )
            seed_episode_rng(episode_seed)
            episode_started_at = time.perf_counter()
            reset_started_at = time.perf_counter()
            obs, info = env.reset(
                seed=episode_seed,
            )
            reset_wall_ms = float((time.perf_counter() - reset_started_at) * 1000.0)
            control_rollout_started_at = time.perf_counter()
            episode_env_name = str(info.get("env_name", cfg.env_name))
            done = False
            episode_return = 0.0
            decision_steps = 0
            episode_e2e_ms: list[float] = []
            episode_prepare_ms: list[float] = []
            episode_policy_ms: list[float] = []
            episode_loop_step_ms: list[float] = []
            episode_sim_env_step_ms: list[float] = []
            episode_execute_total_ms: list[float] = []
            episode_trace_start = len(trace_rows)

            while not done:
                loop_step_started_at = time.perf_counter()
                current_obs = extract_scheduler_observation(obs)
                current_mask = np.asarray(info.get("backbone_action_mask"), dtype=np.float32).reshape(-1)
                schedule_select_started_at = time.perf_counter()
                if schedule_mode == "full_compute":
                    action = np.asarray(
                        [BackboneAction.REFRESH, ActionHeadAction.REFRESH],
                        dtype=np.int64,
                    )
                else:
                    action = schedule_policy.select_action(
                        obs,
                        backbone_action_mask=current_mask,
                    ).as_array()
                action = np.asarray(action, dtype=np.int64)
                schedule_select_ms = float(
                    (time.perf_counter() - schedule_select_started_at) * 1000.0
                )
                env_step_started_at = time.perf_counter()
                obs, reward, terminated, truncated, step_info = env.step(action)
                env_step_wall_ms = float((time.perf_counter() - env_step_started_at) * 1000.0)
                loop_step_wall_ms = float(
                    (time.perf_counter() - loop_step_started_at) * 1000.0
                )
                done = bool(terminated or truncated)
                episode_return += float(reward)
                decision_steps += 1
                info = step_info
                temporal_skip = dict(step_info.get("temporal_skip") or {})
                record = {
                    "episode": episode_idx,
                    "seed": episode_seed,
                    "latency_warmup": latency_warmup,
                    "env_name": episode_env_name,
                    "decision_step": decision_steps,
                    **current_obs,
                    "backbone_action": int(action[0]),
                    "backbone_action_name": BackboneAction.from_any(action[0]).name,
                    "action_head_action": int(action[1]),
                    "action_head_action_name": ActionHeadAction(int(action[1])).name,
                    "backbone_action_mask": current_mask.astype(int).tolist(),
                    "env_reward": float(step_info.get("env_reward", 0.0)),
                    "schedule_reward": float(reward),
                    "schedule_mode": schedule_mode,
                    "schedule_select_ms": schedule_select_ms,
                    "env_step_wall_ms": env_step_wall_ms,
                    "loop_step_wall_ms": loop_step_wall_ms,
                    "scheduler_prepare_ms": float(step_info.get("scheduler_prepare_ms", 0.0)),
                    "schedule_policy_observation_ms": float(
                        temporal_skip.get("schedule_policy_observation_ms", 0.0)
                    ),
                    "schedule_wrapper_build_ms": float(
                        temporal_skip.get("schedule_wrapper_build_ms", 0.0)
                    ),
                    "schedule_prepare_inputs_ms": float(
                        temporal_skip.get("prepare_inputs_ms", 0.0)
                    ),
                    "schedule_first_block_probe_ms": float(
                        temporal_skip.get("first_block_probe_ms", 0.0)
                    ),
                    "schedule_build_observation_ms": float(
                        temporal_skip.get("schedule_build_observation_ms", 0.0)
                    ),
                    "decision_e2e_ms": float(step_info.get("decision_e2e_ms", 0.0)),
                    "decision_full_e2e_ms": float(
                        step_info.get("decision_full_e2e_ms", 0.0)
                    ),
                    "policy_step_ms": float(temporal_skip.get("policy_step_ms", 0.0)),
                    "policy_prepare_inputs_ms": float(
                        temporal_skip.get("policy_prepare_inputs_ms", 0.0)
                    ),
                    "policy_first_block_probe_ms": float(
                        temporal_skip.get("policy_first_block_probe_ms", 0.0)
                    ),
                    "policy_continue_backbone_ms": float(
                        temporal_skip.get("policy_continue_backbone_ms", 0.0)
                    ),
                    "policy_run_action_head_ms": float(
                        temporal_skip.get("policy_run_action_head_ms", 0.0)
                    ),
                    "policy_decode_action_ms": float(
                        temporal_skip.get("policy_decode_action_ms", 0.0)
                    ),
                    "sim_env_step_ms": float(temporal_skip.get("sim_env_step_ms", 0.0)),
                    "execute_total_ms": float(temporal_skip.get("execute_total_ms", 0.0)),
                    "execute_policy_observation_ms": float(
                        temporal_skip.get("execute_policy_observation_ms", 0.0)
                    ),
                    "execute_wrapper_ms": float(
                        temporal_skip.get("execute_wrapper_ms", 0.0)
                    ),
                    "use_backbone_cache": bool(temporal_skip.get("use_backbone_cache", False)),
                    "backbone_middle_cache_used": bool(
                        temporal_skip.get("backbone_middle_cache_used", False)
                    ),
                    "image_free_skip": bool(temporal_skip.get("image_free_skip", False)),
                    "success": bool(step_info.get("success", False)),
                }
                trace_rows.append(record)
                episode_e2e_ms.append(record["decision_e2e_ms"])
                episode_prepare_ms.append(record["scheduler_prepare_ms"])
                episode_policy_ms.append(record["policy_step_ms"])
                episode_loop_step_ms.append(record["loop_step_wall_ms"])
                episode_sim_env_step_ms.append(record["sim_env_step_ms"])
                episode_execute_total_ms.append(record["execute_total_ms"])

                if cfg.log_each_step:
                    print(record)
            control_rollout_wall_ms = float(
                (time.perf_counter() - control_rollout_started_at) * 1000.0
            )
            episode_wall_ms = float((time.perf_counter() - episode_started_at) * 1000.0)
            episode_row = {
                "episode": episode_idx,
                "seed": episode_seed,
                "latency_warmup": latency_warmup,
                "env_name": episode_env_name,
                "return": float(episode_return),
                "decision_steps": int(decision_steps),
                "success": bool(step_info.get("success", False)),
                "reset_wall_ms": reset_wall_ms,
                "control_rollout_wall_ms": control_rollout_wall_ms,
                "episode_wall_ms": episode_wall_ms,
                "loop_step_wall_ms_mean": float(np.mean(episode_loop_step_ms))
                if episode_loop_step_ms
                else 0.0,
                "sim_env_step_ms_mean": float(np.mean(episode_sim_env_step_ms))
                if episode_sim_env_step_ms
                else 0.0,
                "execute_total_ms_mean": float(np.mean(episode_execute_total_ms))
                if episode_execute_total_ms
                else 0.0,
                "decision_e2e_ms_mean": float(np.mean(episode_e2e_ms)) if episode_e2e_ms else 0.0,
                "scheduler_prepare_ms_mean": float(np.mean(episode_prepare_ms))
                if episode_prepare_ms
                else 0.0,
                "policy_step_ms_mean": float(np.mean(episode_policy_ms)) if episode_policy_ms else 0.0,
            }
            episode_rows.append(episode_row)
            if trace_path is not None and episode_checkpoint_path is not None:
                append_jsonl(trace_path, trace_rows[episode_trace_start:])
                append_jsonl(episode_checkpoint_path, [episode_row])
            print(episode_row)
        if output_dir is not None:
            write_tsv(
                output_dir / "summary.tsv",
                [
                    "episode",
                    "seed",
                    "latency_warmup",
                    "env_name",
                    "return",
                    "decision_steps",
                    "success",
                    "reset_wall_ms",
                    "control_rollout_wall_ms",
                    "episode_wall_ms",
                    "loop_step_wall_ms_mean",
                    "sim_env_step_ms_mean",
                    "execute_total_ms_mean",
                    "decision_e2e_ms_mean",
                    "scheduler_prepare_ms_mean",
                    "policy_step_ms_mean",
                ],
                episode_rows,
            )
            write_jsonl(trace_path, trace_rows)
            aggregate = {
                **_episode_aggregate(episode_rows),
                "schedule_mode": schedule_mode,
                "step_weighted": _trace_aggregate(trace_rows),
                "mean_decision_e2e_ms": float(np.mean([row["decision_e2e_ms_mean"] for row in episode_rows]))
                if episode_rows
                else 0.0,
                "mean_scheduler_prepare_ms": float(
                    np.mean([row["scheduler_prepare_ms_mean"] for row in episode_rows])
                )
                if episode_rows
                else 0.0,
                "mean_policy_step_ms": float(
                    np.mean([row["policy_step_ms_mean"] for row in episode_rows])
                )
                if episode_rows
                else 0.0,
                "summary_path": str(output_dir / "summary.tsv"),
                "decision_trace_path": str(trace_path),
            }
            warmup_episodes = configured_warmup_episodes
            steady_state_rows = [
                row
                for row in episode_rows
                if not bool(
                    row.get(
                        "latency_warmup",
                        int(row["episode"]) < warmup_episodes,
                    )
                )
            ]
            steady_state_trace_rows = [
                row
                for row in trace_rows
                if not bool(
                    row.get(
                        "latency_warmup",
                        int(row["episode"]) < warmup_episodes,
                    )
                )
            ]
            aggregate["latency_warmup_episodes"] = warmup_episodes
            aggregate["latency_warmup_episode_indices"] = [
                int(row["episode"])
                for row in episode_rows
                if bool(row.get("latency_warmup", False))
            ]
            aggregate["steady_state"] = {
                **_episode_aggregate(steady_state_rows),
                "step_weighted": _trace_aggregate(steady_state_trace_rows),
            }
            per_env_rows: dict[str, list[dict[str, Any]]] = {}
            for row in episode_rows:
                per_env_rows.setdefault(str(row["env_name"]), []).append(row)
            aggregate["per_env"] = {
                env_name: _episode_aggregate(rows)
                for env_name, rows in sorted(per_env_rows.items())
            }
            write_json(output_dir / "aggregate.json", aggregate)
            print(aggregate)
    finally:
        env.close()


def cli_main() -> None:
    main(tyro.cli(RunConfig))


if __name__ == "__main__":
    cli_main()
