#!/usr/bin/env python3

"""Compare temporal skip with full compute on fixed SimplerEnv trajectories."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import pickle
import random
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from sb3_contrib import MaskablePPO

from gr00t.eval.rollout_policy import MultiStepConfig, VideoConfig, WrapperConfigs, create_eval_env
from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.policy.gr00t_policy import Gr00tPolicy
from elegantvla.gr00t.entrypoints._analysis import (
    InstrumentedTemporalSkipPolicyWrapper,
    TracingRLSchedulePolicy,
)
from elegantvla.gr00t.entrypoints._common import write_json, write_jsonl
from elegantvla.gr00t.analysis.open_loop_error.metrics import action_dict_error, mean_metrics
from elegantvla.gr00t.core.types import ActionHeadAction, BackboneAction, ScheduleAction


DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "")
DEFAULT_CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "")
DEFAULT_ENVS = [
    "simpler_env_widowx/widowx_spoon_on_towel",
    "simpler_env_widowx/widowx_carrot_on_plate",
    "simpler_env_widowx/widowx_put_eggplant_in_basket",
]


@dataclass
class OpenLoopConfig:
    model_path: str = DEFAULT_MODEL_PATH
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH
    output_dir: str = "outputs/gr00t/open_loop_error/widowx_balanced_bc"
    env_names: list[str] = field(default_factory=lambda: list(DEFAULT_ENVS))
    trajectories_per_env: int = 1
    seed: int = 0
    max_trajectory_steps: int = 40
    max_episode_steps: int = 300
    n_action_steps: int = 1
    steps_per_render: int = 2
    inference_seed: int = 20260722
    trajectory_driver: str = "physics_only"
    gr00t_device: str = "cuda"
    ppo_device: str = "cpu"
    strict: bool = True
    deterministic: bool = True
    compile_action_head: bool = True
    compile_mode: str = "max-autotune"
    static_inference_cache: bool = True
    speed_smoothing_window: int = 4


def _set_inference_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_obs_normalizer(checkpoint_path: Path) -> object | None:
    stem = checkpoint_path.stem if checkpoint_path.suffix == ".zip" else checkpoint_path.name
    candidates = [checkpoint_path.with_name(f"{stem}_vecnormalize.pkl")]
    if stem.startswith("ppo_schedule_"):
        suffix = stem[len("ppo_schedule_") :]
        candidates.append(checkpoint_path.with_name(f"ppo_schedule_vecnormalize_{suffix}.pkl"))
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("rb") as handle:
                normalizer = pickle.load(handle)
            setattr(normalizer, "training", False)
            setattr(normalizer, "norm_reward", False)
            return normalizer
    return None


def _normalize_observation(
    observation: dict[str, np.ndarray],
    normalizer: object | None,
) -> dict[str, np.ndarray]:
    if normalizer is None:
        return observation
    normalized = normalizer.normalize_obs(observation)
    if not isinstance(normalized, dict):
        raise TypeError("VecNormalize must return a dict observation.")
    return normalized


def _create_sim_env(cfg: OpenLoopConfig, env_name: str) -> Any:
    return create_eval_env(
        env_name=env_name,
        env_idx=0,
        total_n_envs=1,
        wrapper_configs=WrapperConfigs(
            video=VideoConfig(
                video_dir=None,
                steps_per_render=cfg.steps_per_render,
                max_episode_steps=cfg.max_episode_steps,
            ),
            multistep=MultiStepConfig(
                n_action_steps=cfg.n_action_steps,
                max_episode_steps=cfg.max_episode_steps,
                terminate_on_success=True,
            ),
        ),
    )


def _build_policy_observation(
    sim_obs: dict[str, Any],
    modality_configs: dict[str, Any],
) -> dict[str, Any]:
    policy_obs: dict[str, dict[str, Any]] = {"video": {}, "state": {}, "language": {}}
    for modality in ("video", "state", "language"):
        for key in modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
                if key == "task" and "annotation.human.coarse_action" in sim_obs:
                    parsed_key = "annotation.human.coarse_action"
                values = sim_obs[parsed_key]
                if isinstance(values, str):
                    policy_obs[modality][key] = [[values]]
                elif isinstance(values, np.ndarray):
                    policy_obs[modality][key] = [[str(item)] for item in values.reshape(-1)]
                elif isinstance(values, (list, tuple)):
                    policy_obs[modality][key] = [[str(item)] for item in values]
                else:
                    policy_obs[modality][key] = [[str(values)]]
            elif modality == "video":
                policy_obs[modality][key] = np.expand_dims(
                    np.asarray(sim_obs[f"video.{key}"], dtype=np.uint8),
                    axis=0,
                )
            else:
                policy_obs[modality][key] = np.expand_dims(
                    np.asarray(sim_obs[f"state.{key}"], dtype=np.float32),
                    axis=0,
                )
    return policy_obs


def _to_sim_action(policy_action: dict[str, Any]) -> dict[str, Any]:
    sim_action: dict[str, Any] = {}
    for key, value in policy_action.items():
        array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] == 1:
            array = array[0]
        if np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32, copy=False)
        key_str = str(key)
        sim_action[key_str if key_str.startswith("action.") else f"action.{key_str}"] = array
    return sim_action


def _find_observation_adapter(sim_env: Any) -> Any:
    current = sim_env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if callable(getattr(current, "_process_observation", None)) and hasattr(current, "env"):
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("Could not find the SimplerEnv observation adapter in the wrapper chain.")


def _single_step_observation(processed_observation: dict[str, Any]) -> dict[str, Any]:
    observation: dict[str, Any] = {}
    for key, value in processed_observation.items():
        if isinstance(value, str):
            observation[key] = value
        else:
            observation[key] = np.expand_dims(np.asarray(value), axis=0)
    return observation


def _advance_simulation(
    sim_env: Any,
    *,
    trajectory_driver: str,
    reference_action: dict[str, Any],
) -> tuple[dict[str, Any], bool, bool]:
    driver = str(trajectory_driver).strip().lower()
    if driver == "policy_action":
        observation, _, terminated, truncated, _ = sim_env.step(
            _to_sim_action(reference_action)
        )
        return observation, bool(terminated), bool(truncated)
    if driver != "physics_only":
        raise ValueError(
            "trajectory_driver must be either 'physics_only' or 'policy_action'."
        )

    adapter = _find_observation_adapter(sim_env)
    raw_observation, _, terminated, truncated, _ = adapter.env.step(None)
    processed_observation = adapter._process_observation(raw_observation)
    return (
        _single_step_observation(processed_observation),
        bool(terminated),
        bool(truncated),
    )


def _copy_nested_arrays(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_nested_arrays(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return [_copy_nested_arrays(item) for item in value]
    return value


def _set_step_progress(schedule_state: Any, step: int, max_episode_steps: int) -> None:
    schedule_state.schedule_obs = dict(schedule_state.schedule_obs)
    schedule_state.schedule_obs["step_progress"] = np.asarray(
        [np.clip(float(step) / float(max(max_episode_steps, 1)), 0.0, 1.0)],
        dtype=np.float32,
    )


def _save_trajectory(
    path: Path,
    *,
    observations: list[dict[str, Any]],
    reference_actions: list[dict[str, Any]],
    inference_seeds: list[int],
    metadata: dict[str, Any],
) -> None:
    payload: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "inference_seeds": np.asarray(inference_seeds, dtype=np.int64),
    }
    for modality in ("video", "state"):
        for key in sorted(observations[0][modality]):
            payload[f"observation.{modality}.{key}"] = np.concatenate(
                [np.asarray(observation[modality][key]) for observation in observations],
                axis=0,
            )
    for key in sorted(observations[0]["language"]):
        payload[f"observation.language.{key}"] = np.asarray(
            [str(observation["language"][key][0][0]) for observation in observations]
        )
    for key in sorted(reference_actions[0]):
        payload[f"reference_action.{key}"] = np.concatenate(
            [np.asarray(action[key]) for action in reference_actions],
            axis=0,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _trajectory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    backbone_counts = Counter(str(row["backbone_action_name"]) for row in rows)
    action_head_counts = Counter(str(row["action_head_action_name"]) for row in rows)
    total = max(len(rows), 1)
    return {
        "steps": len(rows),
        "chunk_error": mean_metrics(rows, metric_key="chunk_error"),
        "executed_prefix_error": mean_metrics(rows, metric_key="executed_prefix_error"),
        "backbone_action_counts": dict(backbone_counts),
        "backbone_action_ratios": {key: value / total for key, value in backbone_counts.items()},
        "action_head_action_counts": dict(action_head_counts),
        "action_head_action_ratios": {key: value / total for key, value in action_head_counts.items()},
    }


def _plot_errors(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    x = np.arange(1, len(rows) + 1)
    chunk_mae = [float(row["chunk_error"]["overall"]["mae"]) for row in rows]
    prefix_mae = [float(row["executed_prefix_error"]["overall"]["mae"]) for row in rows]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, chunk_mae, label="full action chunk MAE", linewidth=1.5)
    ax.plot(x, prefix_mae, label="executed prefix MAE", linewidth=1.5)
    ax.set_xlabel("fixed-trajectory decision step")
    ax.set_ylabel("unnormalized action MAE")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main(cfg: OpenLoopConfig) -> None:
    print("Open-loop evaluator main started.", flush=True)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(cfg.checkpoint_path)
    if not checkpoint_path.exists() and checkpoint_path.suffix != ".zip":
        checkpoint_path = checkpoint_path.with_suffix(".zip")
    if not Path(cfg.model_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {cfg.model_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Scheduler checkpoint does not exist: {cfg.checkpoint_path}")
    if not cfg.env_names:
        raise ValueError("At least one env name is required.")
    if str(cfg.trajectory_driver).strip().lower() not in {"physics_only", "policy_action"}:
        raise ValueError(
            "trajectory_driver must be either 'physics_only' or 'policy_action'."
        )

    write_json(output_dir / "config.json", {**asdict(cfg), "checkpoint_path": str(checkpoint_path)})
    print(f"Loading GR00T from {cfg.model_path}", flush=True)
    base_policy = Gr00tPolicy(
        embodiment_tag=get_embodiment_tag_from_env_name(cfg.env_names[0]),
        model_path=cfg.model_path,
        device=cfg.gr00t_device,
        strict=cfg.strict,
    )
    wrapper = InstrumentedTemporalSkipPolicyWrapper(
        base_policy,
        speed_smoothing_window=cfg.speed_smoothing_window,
        compile_action_head=cfg.compile_action_head,
        compile_mode=cfg.compile_mode,
        static_inference_cache=cfg.static_inference_cache,
        strict=cfg.strict,
    )
    model = MaskablePPO.load(str(checkpoint_path), device=cfg.ppo_device)
    scheduler = TracingRLSchedulePolicy(model=model, deterministic=cfg.deterministic)
    obs_normalizer = _load_obs_normalizer(checkpoint_path)

    all_rows: list[dict[str, Any]] = []
    trajectory_summaries: list[dict[str, Any]] = []
    trajectory_index = 0
    started_at = time.perf_counter()
    for env_name in cfg.env_names:
        sim_env = _create_sim_env(cfg, env_name)
        try:
            for local_trajectory_index in range(cfg.trajectories_per_env):
                trajectory_seed = int(cfg.seed + local_trajectory_index)
                sim_obs, reset_info = sim_env.reset(seed=trajectory_seed)
                wrapper.reset()
                observations: list[dict[str, Any]] = []
                reference_actions: list[dict[str, Any]] = []
                inference_seeds: list[int] = []
                terminated = False
                truncated = False
                for step in range(cfg.max_trajectory_steps):
                    policy_obs = _build_policy_observation(sim_obs, base_policy.modality_configs)
                    schedule_state = wrapper.build_schedule_state(policy_obs)
                    _set_step_progress(schedule_state, step, cfg.max_episode_steps)
                    step_inference_seed = int(
                        cfg.inference_seed + trajectory_index * cfg.max_trajectory_steps + step
                    )
                    _set_inference_seed(step_inference_seed)
                    reference_action, _ = wrapper.execute_schedule_action(
                        policy_obs,
                        ScheduleAction(BackboneAction.REFRESH, ActionHeadAction.REFRESH),
                        schedule_state,
                    )
                    observations.append(_copy_nested_arrays(policy_obs))
                    reference_actions.append(_copy_nested_arrays(reference_action))
                    inference_seeds.append(step_inference_seed)
                    sim_obs, terminated, truncated = _advance_simulation(
                        sim_env,
                        trajectory_driver=cfg.trajectory_driver,
                        reference_action=reference_action,
                    )
                    if terminated or truncated:
                        break

                trajectory_name = f"trajectory_{trajectory_index:04d}"
                trajectory_metadata = {
                    "trajectory_index": trajectory_index,
                    "env_name": env_name,
                    "seed": trajectory_seed,
                    "steps": len(observations),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "reset_info": str(reset_info),
                }
                _save_trajectory(
                    output_dir / "trajectories" / f"{trajectory_name}.npz",
                    observations=observations,
                    reference_actions=reference_actions,
                    inference_seeds=inference_seeds,
                    metadata=trajectory_metadata,
                )

                wrapper.reset()
                trajectory_rows: list[dict[str, Any]] = []
                for step, (policy_obs, reference_action, step_inference_seed) in enumerate(
                    zip(observations, reference_actions, inference_seeds, strict=True)
                ):
                    schedule_state = wrapper.build_schedule_state(policy_obs)
                    _set_step_progress(schedule_state, step, cfg.max_episode_steps)
                    normalized_obs = _normalize_observation(
                        schedule_state.schedule_obs,
                        obs_normalizer,
                    )
                    selected_action = scheduler.select_action(
                        normalized_obs,
                        backbone_action_mask=schedule_state.backbone_action_mask,
                    )
                    _set_inference_seed(step_inference_seed)
                    candidate_action, policy_info = wrapper.execute_schedule_action(
                        policy_obs,
                        selected_action,
                        schedule_state,
                    )
                    row = {
                        "trajectory_index": trajectory_index,
                        "env_name": env_name,
                        "seed": trajectory_seed,
                        "step": step,
                        "inference_seed": step_inference_seed,
                        "backbone_action": int(selected_action.backbone_action),
                        "backbone_action_name": selected_action.backbone_action.name,
                        "action_head_action": int(selected_action.action_head_action),
                        "action_head_action_name": selected_action.action_head_action.name,
                        "use_backbone_cache": bool(policy_info.get("use_backbone_cache", False)),
                        "image_free_skip": bool(policy_info.get("image_free_skip", False)),
                        "backbone_middle_cache_used": bool(
                            policy_info.get("backbone_middle_cache_used", False)
                        ),
                        "chunk_error": action_dict_error(reference_action, candidate_action),
                        "executed_prefix_error": action_dict_error(
                            reference_action,
                            candidate_action,
                            prefix_steps=cfg.n_action_steps,
                        ),
                    }
                    trajectory_rows.append(row)
                    all_rows.append(row)

                summary = {
                    **trajectory_metadata,
                    **_trajectory_summary(trajectory_rows),
                }
                trajectory_summaries.append(summary)
                write_json(output_dir / "summaries" / f"{trajectory_name}.json", summary)
                write_jsonl(output_dir / "steps.jsonl", all_rows)
                print(
                    f"[{trajectory_name}] {env_name} seed={trajectory_seed} "
                    f"steps={len(trajectory_rows)} chunk_mae={summary['chunk_error']['mae']:.6g} "
                    f"prefix_mae={summary['executed_prefix_error']['mae']:.6g}",
                    flush=True,
                )
                trajectory_index += 1
        finally:
            sim_env.close()

    aggregate = {
        "trajectories": len(trajectory_summaries),
        "decision_steps": len(all_rows),
        "wall_seconds": float(time.perf_counter() - started_at),
        "chunk_error": mean_metrics(all_rows, metric_key="chunk_error"),
        "executed_prefix_error": mean_metrics(all_rows, metric_key="executed_prefix_error"),
        "by_env": {
            env_name: _trajectory_summary([row for row in all_rows if row["env_name"] == env_name])
            for env_name in cfg.env_names
        },
        "schedule_distribution": _trajectory_summary(all_rows),
        "trajectory_summaries": trajectory_summaries,
    }
    write_json(output_dir / "summary.json", aggregate)
    _plot_errors(all_rows, output_dir / "action_error.png")
    print(json.dumps(aggregate["chunk_error"], sort_keys=True), flush=True)


def cli_main() -> None:
    main(tyro.cli(OpenLoopConfig))


if __name__ == "__main__":
    cli_main()
