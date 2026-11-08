#!/usr/bin/env python3

"""Evaluate a trained ElegantVLA scheduler on GR00T."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import pickle
import time
from typing import Any

import numpy as np
import tyro

from sb3_contrib import MaskablePPO

from elegantvla.gr00t.entrypoints._analysis import (
    InstrumentedMultiTaskTemporalSkipEnv,
    InstrumentedTemporalSkipEnv,
    InstrumentedTemporalSkipPolicyWrapper,
    TracingRLSchedulePolicy,
)
from elegantvla.gr00t.entrypoints._common import (
    finish_wandb_run,
    init_wandb_run,
    log_wandb,
    make_wandb_table,
    maybe_wandb_save,
    maybe_wandb_save_many,
    update_wandb_summary,
    write_json,
    write_jsonl,
    write_tsv,
)
from elegantvla.gr00t.entrypoints._runtime import (
    build_temporal_skip_env,
    extract_scheduler_observation,
    seed_episode_rng,
)
from elegantvla.gr00t.core.types import ActionHeadAction, BackboneAction


@dataclass
class EvalConfig:
    checkpoint_path: str
    model_path: str = "/var/tmp/isaac-gr00t-node7-server/models/GR00T-N1.6-fractal"
    env_name: str = "simpler_env_google/google_robot_pick_object"
    task_group: str = ""
    episodes: int = 5
    seed: int | None = 0
    gr00t_device: str = "cuda"
    ppo_device: str = "cpu"
    strict: bool = True
    compile_action_head: bool = True
    compile_mode: str = "max-autotune"
    static_inference_cache: bool = True
    deterministic: bool = True
    max_backbone_action_level: int | None = None
    max_action_head_action_level: int | None = None
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

    output_dir: str | None = None
    log_each_step: bool = False
    latency_warmup_episodes: int = 1
    use_wandb: bool = False
    wandb_project: str = "temporal-skip"
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    wandb_tags: str = ""
    wandb_dir: str | None = None
    wandb_job_type: str = "ppo-eval"
    wandb_extra_paths: str = ""
    wandb_log_step_history: bool = True
    wandb_save_tables: bool = True


def _load_obs_normalizer(checkpoint_path: Path) -> object | None:
    stem = checkpoint_path.stem if checkpoint_path.suffix == ".zip" else checkpoint_path.name
    candidates = [checkpoint_path.with_name(f"{stem}_vecnormalize.pkl")]
    if stem.startswith("ppo_schedule_"):
        suffix = stem[len("ppo_schedule_") :]
        candidates.append(checkpoint_path.with_name(f"ppo_schedule_vecnormalize_{suffix}.pkl"))
    for candidate in candidates:
        if not candidate.exists():
            continue
        with candidate.open("rb") as handle:
            normalizer = pickle.load(handle)
        setattr(normalizer, "training", False)
        setattr(normalizer, "norm_reward", False)
        return normalizer
    return None


def _normalize_observation(
    observation: dict[str, np.ndarray],
    obs_normalizer: object | None,
) -> dict[str, np.ndarray]:
    if obs_normalizer is None:
        return observation
    normalized = obs_normalizer.normalize_obs(observation)
    assert isinstance(normalized, dict)
    return normalized


def _selected_action_probability(probs: Any, action: Any) -> float | None:
    if probs is None or action is None:
        return None
    array = np.asarray(probs, dtype=np.float32).reshape(-1)
    index = int(action)
    if index < 0 or index >= array.size:
        return None
    return float(array[index])


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return 0.0
    return float(np.mean(values))


def _backbone_action_branch_count(backbone_action_mask: Any) -> int:
    if backbone_action_mask is None:
        return 0
    try:
        return sum(int(value) for value in backbone_action_mask)
    except TypeError:
        return 0


def _is_valid_backbone_decision_step(backbone_action_mask: Any) -> bool:
    return _backbone_action_branch_count(backbone_action_mask) > 1


def _ratio_metric(rows: list[dict[str, Any]], key: str, expected: str) -> float:
    if not rows:
        return 0.0
    matches = sum(row.get(key) == expected for row in rows)
    return float(matches / len(rows))


def _episode_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_rate = 0.0 if not rows else sum(bool(r["success"]) for r in rows) / len(rows)
    mean_return = 0.0 if not rows else float(np.mean([float(r["return"]) for r in rows]))
    mean_steps = 0.0 if not rows else float(np.mean([float(r["decision_steps"]) for r in rows]))
    return {
        "episodes": int(len(rows)),
        "success_rate": float(success_rate),
        "mean_return": float(mean_return),
        "mean_decision_steps": float(mean_steps),
        "mean_reset_wall_ms": _mean_metric(rows, "reset_wall_ms"),
        "mean_control_rollout_wall_ms": _mean_metric(
            rows,
            "control_rollout_wall_ms",
        ),
        "mean_episode_wall_ms": _mean_metric(rows, "episode_wall_ms"),
        "mean_loop_step_wall_ms": _mean_metric(rows, "loop_step_wall_ms_mean"),
        "mean_sim_env_step_ms": _mean_metric(rows, "sim_env_step_ms_mean"),
    }


def _wandb_step_payload(step_row: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    valid_backbone_decision = bool(step_row.get("is_valid_backbone_decision_step"))
    for key, value in step_row.items():
        if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
            if key == "global_decision_index":
                continue
            if not valid_backbone_decision and key in {
                "backbone_action",
                "resolved_backbone_action",
                "target_backbone_action",
            }:
                continue
            payload[f"eval/step/{key}"] = (
                value.item() if isinstance(value, np.generic) else value
            )
    return payload


def main(cfg: EvalConfig) -> None:
    checkpoint_path = Path(cfg.checkpoint_path)
    if not checkpoint_path.exists() and checkpoint_path.suffix != ".zip":
        checkpoint_path = checkpoint_path.with_suffix(".zip")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {cfg.checkpoint_path!r} does not exist.")

    env = build_temporal_skip_env(
        cfg,
        wrapper_cls=InstrumentedTemporalSkipPolicyWrapper,
        env_cls=InstrumentedTemporalSkipEnv,
        multi_task_env_cls=InstrumentedMultiTaskTemporalSkipEnv,
    )
    model = MaskablePPO.load(str(checkpoint_path), device=cfg.ppo_device)
    policy = TracingRLSchedulePolicy(
        model=model,
        deterministic=cfg.deterministic,
        max_backbone_action_level=cfg.max_backbone_action_level,
        max_action_head_action_level=cfg.max_action_head_action_level,
    )
    obs_normalizer = _load_obs_normalizer(checkpoint_path)

    output_dir = None if cfg.output_dir is None else Path(cfg.output_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb_run(
        enabled=cfg.use_wandb,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        group=cfg.wandb_group,
        name=cfg.wandb_run_name or (
            output_dir.name if output_dir is not None else f"{checkpoint_path.stem}_eval"
        ),
        job_type=cfg.wandb_job_type,
        tags=cfg.wandb_tags,
        config={
            **asdict(cfg),
            "resolved_checkpoint_path": str(checkpoint_path),
            "resolved_output_dir": None if output_dir is None else str(output_dir),
        },
        dir=cfg.wandb_dir,
        mode=cfg.wandb_mode,
    )

    rows: list[dict[str, object]] = []
    step_rows: list[dict[str, object]] = []
    global_decision_index = 0
    summary_path: Path | None = None
    decision_trace_path: Path | None = None
    episode_path: Path | None = None
    aggregate_path: Path | None = None
    try:
        for episode_idx in range(cfg.episodes):
            episode_seed = None if cfg.seed is None else cfg.seed + episode_idx
            seed_episode_rng(episode_seed)
            episode_started_at = time.perf_counter()
            reset_started_at = time.perf_counter()
            obs, info = env.reset(seed=episode_seed)
            reset_wall_ms = float((time.perf_counter() - reset_started_at) * 1000.0)
            control_rollout_started_at = time.perf_counter()
            episode_env_name = str(info.get("env_name", cfg.env_name))
            done = False
            episode_return = 0.0
            decision_steps = 0
            episode_step_row_start = len(step_rows)

            while not done:
                loop_step_started_at = time.perf_counter()
                current_obs = obs
                full_current_obs = info.get("full_schedule_observation")
                if not isinstance(full_current_obs, dict):
                    full_current_obs = current_obs
                current_backbone_action_mask = info.get("backbone_action_mask")
                normalized_obs = _normalize_observation(obs, obs_normalizer)
                target_select_started_at = time.perf_counter()
                target_schedule_action = env.reward_policy.select_action(
                    full_current_obs,
                    backbone_action_mask=current_backbone_action_mask,
                )
                target_select_ms = float(
                    (time.perf_counter() - target_select_started_at) * 1000.0
                )
                target_backbone_action = target_schedule_action.backbone_action
                target_action_head_action = target_schedule_action.action_head_action
                rl_select_started_at = time.perf_counter()
                action = policy.select_action(
                    normalized_obs,
                    backbone_action_mask=current_backbone_action_mask,
                ).as_array()
                rl_select_ms = float((time.perf_counter() - rl_select_started_at) * 1000.0)
                action = np.asarray(action, dtype=np.int64)
                scheduler_policy_select_ms = target_select_ms + rl_select_ms
                step_index = decision_steps + 1
                env_step_started_at = time.perf_counter()
                obs, reward, terminated, truncated, step_info = env.step(action)
                env_step_wall_ms = float((time.perf_counter() - env_step_started_at) * 1000.0)
                loop_step_wall_ms = float(
                    (time.perf_counter() - loop_step_started_at) * 1000.0
                )
                done = bool(terminated or truncated)
                episode_return += float(reward)
                decision_steps += 1
                global_decision_index += 1
                info = step_info
                backbone_matches_rule = int(action[0]) == int(target_backbone_action)
                action_head_matches_rule = int(action[1]) == int(target_action_head_action)
                valid_backbone_decision_step = _is_valid_backbone_decision_step(
                    current_backbone_action_mask
                )
                trace = dict(policy.last_trace())
                if not valid_backbone_decision_step:
                    trace["selected_backbone_action_prob"] = None
                temporal_skip = dict(step_info.get("temporal_skip") or {})
                step_row = {
                    "episode": int(episode_idx),
                    "seed": episode_seed,
                    "env_name": episode_env_name,
                    "decision_step": int(step_index),
                    "global_decision_index": int(global_decision_index),
                    **extract_scheduler_observation(full_current_obs),
                    **trace,
                    "target_backbone_action": int(target_backbone_action),
                    "target_backbone_action_name": BackboneAction.from_any(target_backbone_action).name,
                    "target_action_head_action": int(target_action_head_action),
                    "target_action_head_action_name": ActionHeadAction.from_any(target_action_head_action).name,
                    "target_backbone_action_prob": (
                        _selected_action_probability(
                            policy.last_backbone_action_probs,
                            target_backbone_action,
                        )
                        if valid_backbone_decision_step
                        else None
                    ),
                    "target_action_head_action_prob": _selected_action_probability(
                        policy.last_action_head_action_probs,
                        target_action_head_action,
                    ),
                    "backbone_matches_rule": bool(backbone_matches_rule),
                    "action_head_matches_rule": bool(action_head_matches_rule),
                    "matches_rule": bool(backbone_matches_rule and action_head_matches_rule),
                    "is_valid_backbone_decision_step": bool(valid_backbone_decision_step),
                    "env_reward": float(step_info.get("env_reward", 0.0)),
                    "schedule_reward": float(step_info.get("schedule_reward", reward)),
                    "target_select_ms": target_select_ms,
                    "rl_select_ms": rl_select_ms,
                    "scheduler_policy_select_ms": scheduler_policy_select_ms,
                    "env_step_wall_ms": env_step_wall_ms,
                    "loop_step_wall_ms": loop_step_wall_ms,
                    "scheduler_prepare_ms": float(
                        step_info.get("scheduler_prepare_ms", 0.0)
                    ),
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
                    "use_backbone_cache": bool(
                        temporal_skip.get("use_backbone_cache", False)
                    ),
                    "image_free_skip": bool(temporal_skip.get("image_free_skip", False)),
                    "backbone_middle_cache_used": bool(
                        temporal_skip.get("backbone_middle_cache_used", False)
                    ),
                    "success": bool(step_info.get("success", False)),
                }
                step_rows.append(step_row)
                if wandb_run is not None and cfg.wandb_log_step_history:
                    log_wandb(
                        _wandb_step_payload(step_row),
                        enabled=cfg.use_wandb,
                        step=global_decision_index,
                    )
                if cfg.log_each_step:
                    print(step_row)

            control_rollout_wall_ms = float(
                (time.perf_counter() - control_rollout_started_at) * 1000.0
            )
            episode_wall_ms = float((time.perf_counter() - episode_started_at) * 1000.0)
            episode_step_rows = step_rows[episode_step_row_start:]
            row = {
                "episode": int(episode_idx),
                "seed": episode_seed,
                "env_name": episode_env_name,
                "return": float(episode_return),
                "decision_steps": int(decision_steps),
                "success": bool(step_info.get("success", False)),
                "reset_wall_ms": reset_wall_ms,
                "control_rollout_wall_ms": control_rollout_wall_ms,
                "episode_wall_ms": episode_wall_ms,
                "loop_step_wall_ms_mean": _mean_metric(
                    episode_step_rows,
                    "loop_step_wall_ms",
                ),
                "scheduler_policy_select_ms_mean": _mean_metric(
                    episode_step_rows,
                    "scheduler_policy_select_ms",
                ),
                "scheduler_prepare_ms_mean": _mean_metric(
                    episode_step_rows,
                    "scheduler_prepare_ms",
                ),
                "policy_step_ms_mean": _mean_metric(
                    episode_step_rows,
                    "policy_step_ms",
                ),
                "sim_env_step_ms_mean": _mean_metric(
                    episode_step_rows,
                    "sim_env_step_ms",
                ),
                "execute_total_ms_mean": _mean_metric(
                    episode_step_rows,
                    "execute_total_ms",
                ),
            }
            rows.append(row)
            print(row)
            if output_dir is not None:
                live_episode_path = output_dir / "episodes_live.jsonl"
                live_aggregate_path = output_dir / "aggregate_live.json"
                write_jsonl(live_episode_path, rows)
                live_aggregate: dict[str, Any] = {
                    **_episode_aggregate(rows),
                    "completed_episodes": int(len(rows)),
                    "target_episodes": int(cfg.episodes),
                }
                env_names = sorted({str(r["env_name"]) for r in rows})
                live_aggregate["per_env"] = {
                    env_name: _episode_aggregate(
                        [r for r in rows if str(r["env_name"]) == env_name]
                    )
                    for env_name in env_names
                }
                write_json(live_aggregate_path, live_aggregate)
        backbone_match_rate = (
            0.0
            if not step_rows
            else sum(bool(r["backbone_matches_rule"]) for r in step_rows) / len(step_rows)
        )
        action_head_match_rate = (
            0.0
            if not step_rows
            else sum(bool(r["action_head_matches_rule"]) for r in step_rows) / len(step_rows)
        )
        valid_backbone_rows = [
            row for row in step_rows if bool(row.get("is_valid_backbone_decision_step"))
        ]
        full_match_rate = (
            0.0
            if not step_rows
            else sum(bool(r["matches_rule"]) for r in step_rows) / len(step_rows)
        )
        aggregate: dict[str, Any] = {
            **_episode_aggregate(rows),
            "logged_step_rows": int(len(step_rows)),
            "valid_backbone_decision_steps": int(len(valid_backbone_rows)),
            "valid_backbone_decision_step_rate": (
                0.0 if not step_rows else float(len(valid_backbone_rows) / len(step_rows))
            ),
            "backbone_match_rate": float(backbone_match_rate),
            "action_head_match_rate": float(action_head_match_rate),
            "full_match_rate": float(full_match_rate),
            "mean_backbone_entropy": _mean_metric(step_rows, "backbone_entropy"),
            "mean_action_head_entropy": _mean_metric(step_rows, "action_head_entropy"),
            "mean_policy_entropy": _mean_metric(step_rows, "policy_entropy"),
            "mean_selected_backbone_action_prob": _mean_metric(step_rows, "selected_backbone_action_prob"),
            "mean_selected_action_head_action_prob": _mean_metric(
                step_rows,
                "selected_action_head_action_prob",
            ),
            "mean_target_backbone_action_prob": _mean_metric(step_rows, "target_backbone_action_prob"),
            "mean_target_action_head_action_prob": _mean_metric(
                step_rows,
                "target_action_head_action_prob",
            ),
            "mean_scheduler_policy_select_ms": _mean_metric(
                step_rows,
                "scheduler_policy_select_ms",
            ),
            "mean_scheduler_prepare_ms": _mean_metric(step_rows, "scheduler_prepare_ms"),
            "mean_policy_step_ms": _mean_metric(step_rows, "policy_step_ms"),
            "mean_sim_env_step_ms": _mean_metric(step_rows, "sim_env_step_ms"),
            "mean_loop_step_wall_ms": _mean_metric(step_rows, "loop_step_wall_ms"),
        }
        warmup_episodes = max(int(cfg.latency_warmup_episodes), 0)
        steady_state_rows = [
            row for row in rows if int(row["episode"]) >= warmup_episodes
        ]
        steady_state_step_rows = [
            row for row in step_rows if int(row["episode"]) >= warmup_episodes
        ]
        aggregate["latency_warmup_episodes"] = warmup_episodes
        aggregate["steady_state"] = {
            **_episode_aggregate(steady_state_rows),
            "mean_scheduler_policy_select_ms": _mean_metric(
                steady_state_step_rows,
                "scheduler_policy_select_ms",
            ),
            "mean_scheduler_prepare_ms": _mean_metric(
                steady_state_step_rows,
                "scheduler_prepare_ms",
            ),
            "mean_policy_step_ms": _mean_metric(steady_state_step_rows, "policy_step_ms"),
            "mean_sim_env_step_ms": _mean_metric(
                steady_state_step_rows,
                "sim_env_step_ms",
            ),
            "mean_loop_step_wall_ms": _mean_metric(
                steady_state_step_rows,
                "loop_step_wall_ms",
            ),
        }
        per_env_rows: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            per_env_rows.setdefault(str(row.get("env_name", "")), []).append(row)
        aggregate["per_env"] = {
            env_name: _episode_aggregate(env_rows)
            for env_name, env_rows in sorted(per_env_rows.items())
        }
        for action in BackboneAction:
            aggregate[f"mean_backbone_prob_{str(action.name).lower()}"] = _mean_metric(
                step_rows,
                f"backbone_prob_{str(action.name).lower()}",
            )
            aggregate[f"backbone_ratio_{str(action.name).lower()}"] = _ratio_metric(
                valid_backbone_rows,
                "backbone_action_name",
                action.name,
            )
        for action in ActionHeadAction:
            aggregate[f"mean_action_head_prob_{str(action.name).lower()}"] = _mean_metric(
                step_rows,
                f"action_head_prob_{str(action.name).lower()}",
            )
            aggregate[f"action_head_ratio_{str(action.name).lower()}"] = _ratio_metric(
                step_rows,
                "action_head_action_name",
                action.name,
            )

        if output_dir is not None:
            summary_path = output_dir / "summary.tsv"
            write_tsv(
                summary_path,
                [
                    "episode",
                    "seed",
                    "env_name",
                    "return",
                    "decision_steps",
                    "success",
                    "reset_wall_ms",
                    "control_rollout_wall_ms",
                    "episode_wall_ms",
                    "loop_step_wall_ms_mean",
                    "scheduler_policy_select_ms_mean",
                    "scheduler_prepare_ms_mean",
                    "policy_step_ms_mean",
                    "sim_env_step_ms_mean",
                    "execute_total_ms_mean",
                ],
                rows,
            )
            decision_trace_path = output_dir / "decision_trace.jsonl"
            write_jsonl(decision_trace_path, step_rows)
            episode_path = output_dir / "episodes.jsonl"
            write_jsonl(episode_path, rows)
            aggregate_path = output_dir / "aggregate.json"
            aggregate_with_paths = dict(aggregate)
            aggregate_with_paths["summary_path"] = str(summary_path)
            aggregate_with_paths["decision_trace_path"] = str(decision_trace_path)
            aggregate_with_paths["episode_path"] = str(episode_path)
            write_json(aggregate_path, aggregate_with_paths)
            aggregate = aggregate_with_paths

        print(aggregate)

        if wandb_run is not None:
            update_wandb_summary(wandb_run, aggregate)
            payload = {f"eval/aggregate/{key}": value for key, value in aggregate.items()}
            if cfg.wandb_save_tables:
                step_table = make_wandb_table(step_rows)
                if step_table is not None:
                    payload["eval/step_table"] = step_table
                episode_table = make_wandb_table([dict(row) for row in rows])
                if episode_table is not None:
                    payload["eval/episode_table"] = episode_table
            log_wandb(payload, enabled=cfg.use_wandb, step=max(global_decision_index, 0))
            maybe_wandb_save(summary_path, enabled=cfg.use_wandb)
            maybe_wandb_save(decision_trace_path, enabled=cfg.use_wandb)
            maybe_wandb_save(episode_path, enabled=cfg.use_wandb)
            maybe_wandb_save(aggregate_path, enabled=cfg.use_wandb)
            maybe_wandb_save_many(cfg.wandb_extra_paths, enabled=cfg.use_wandb)
    finally:
        env.close()
        finish_wandb_run(wandb_run)


def cli_main() -> None:
    main(tyro.cli(EvalConfig))


if __name__ == "__main__":
    cli_main()
