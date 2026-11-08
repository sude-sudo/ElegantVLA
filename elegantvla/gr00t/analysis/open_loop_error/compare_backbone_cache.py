#!/usr/bin/env python3

"""Compare full cache reuse and first/last correction against full compute."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.policy.gr00t_policy import Gr00tPolicy
from elegantvla.gr00t.core.adapter import Gr00tPolicyTemporalSkipAdapter
from elegantvla.gr00t.entrypoints._common import write_json, write_jsonl
from elegantvla.gr00t.analysis.open_loop_error.evaluate import (
    DEFAULT_ENVS,
    DEFAULT_MODEL_PATH,
    _advance_simulation,
    _build_policy_observation,
    _copy_nested_arrays,
    _create_sim_env,
    _save_trajectory,
    _set_inference_seed,
)
from elegantvla.gr00t.analysis.open_loop_error.metrics import (
    action_dict_error,
    array_error,
    mean_metrics,
)
from elegantvla.gr00t.core.wrapper import SkipController


@dataclass
class CacheComparisonConfig:
    model_path: str = DEFAULT_MODEL_PATH
    output_dir: str = "outputs/gr00t/open_loop_error/backbone_cache_comparison"
    env_names: list[str] = field(default_factory=lambda: list(DEFAULT_ENVS))
    trajectories_per_env: int = 1
    seed: int = 0
    max_trajectory_steps: int = 40
    max_episode_steps: int = 300
    n_action_steps: int = 1
    steps_per_render: int = 2
    inference_seed: int = 20260722
    trajectory_driver: str = "rendered_frame_sweep"
    frame_shift_pixels_per_step: int = 1
    qpos_sweep_amplitude: float = 0.0
    object_sweep_amplitude: float = 0.0
    camera_sweep_amplitude: float = 0.05
    render_flush_frames: int = 1
    render_pipeline_warmup_steps: int = 4
    min_changed_frame_fraction: float = 0.8
    changed_frame_mae_threshold: float = 0.01
    gr00t_device: str = "cuda"
    strict: bool = True


def _tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    return array_error(
        reference.detach().float().cpu().numpy(),
        candidate.detach().float().cpu().numpy(),
    )


def _method_summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    return {
        "evaluated_steps": len(rows),
        "representation_error": mean_metrics(
            rows,
            metric_key=f"{prefix}_representation_error_wrapped",
        ),
        "action_chunk_error": mean_metrics(
            rows,
            metric_key=f"{prefix}_action_chunk_error",
        ),
        "executed_prefix_error": mean_metrics(
            rows,
            metric_key=f"{prefix}_executed_prefix_error",
        ),
    }


def _reduction(full_cache: float, corrected: float) -> float | None:
    if full_cache == 0.0:
        return None
    return float((full_cache - corrected) / full_cache)


def _comparison_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated_rows = [row for row in rows if not bool(row["cache_warmup"])]
    full_cache = _method_summary(evaluated_rows, "full_cache")
    corrected = _method_summary(evaluated_rows, "first_last")
    reductions: dict[str, Any] = {}
    for metric_group in (
        "representation_error",
        "action_chunk_error",
        "executed_prefix_error",
    ):
        reductions[metric_group] = {
            metric: _reduction(
                float(full_cache[metric_group][metric]),
                float(corrected[metric_group][metric]),
            )
            for metric in ("mae", "mse")
        }
    return {
        "ground_truth": "current_frame_full_backbone_and_full_action_head_compute",
        "full_cache_method": "previous_frame_full_backbone_cache_plus_current_state",
        "first_last_method": (
            "current_first_layer_plus_previous_middle_delta_cache_plus_current_last_layer"
        ),
        "action_head_control": "full_compute_for_all_three_paths_with_paired_random_seed",
        "warmup_steps_excluded": int(len(rows) - len(evaluated_rows)),
        "observation_change": {
            "transitions": len(evaluated_rows),
            "mean_image_mae": float(
                np.mean([float(row["previous_frame_image_mae"]) for row in evaluated_rows])
            )
            if evaluated_rows
            else 0.0,
            "changed_frame_fraction": float(
                np.mean([bool(row["image_changed"]) for row in evaluated_rows])
            )
            if evaluated_rows
            else 0.0,
        },
        "full_cache": full_cache,
        "first_last_correction": corrected,
        "first_last_relative_error_reduction": reductions,
    }


def _find_raw_simpler_env(sim_env: Any) -> tuple[Any, Any]:
    observation_adapter = sim_env
    visited: set[int] = set()
    while observation_adapter is not None and id(observation_adapter) not in visited:
        visited.add(id(observation_adapter))
        if callable(getattr(observation_adapter, "_process_observation", None)):
            raw_env = getattr(observation_adapter, "env", None)
            if raw_env is not None and hasattr(raw_env.unwrapped, "agent"):
                return observation_adapter, raw_env.unwrapped
        observation_adapter = getattr(observation_adapter, "env", None)
    raise RuntimeError("Could not find the raw SimplerEnv environment and observation adapter.")


def _scene_sweep_observation(
    sim_env: Any,
    *,
    baseline_qpos: np.ndarray,
    sweep_actor: Any,
    baseline_actor_position: np.ndarray,
    baseline_actor_quaternion: np.ndarray,
    sweep_camera: Any,
    baseline_camera_position: np.ndarray,
    baseline_camera_quaternion: np.ndarray,
    step: int,
    total_steps: int,
    qpos_amplitude: float,
    object_amplitude: float,
    camera_amplitude: float,
    render_flush_frames: int,
) -> dict[str, Any]:
    observation_adapter, raw_env = _find_raw_simpler_env(sim_env)
    qpos = np.asarray(baseline_qpos, dtype=np.float64).copy()
    arm_dofs = min(6, qpos.size)
    phase = 2.0 * np.pi * float(step) / float(max(total_steps, 2))
    joint_phases = np.arange(arm_dofs, dtype=np.float64) * 0.37
    qpos[:arm_dofs] += float(qpos_amplitude) * (
        np.sin(phase + joint_phases) - np.sin(joint_phases)
    )
    raw_env.agent.robot.set_qpos(qpos)
    actor_position = np.asarray(baseline_actor_position, dtype=np.float64).copy()
    actor_position[0] += float(object_amplitude) * np.sin(phase)
    actor_position[1] += float(object_amplitude) * (np.cos(phase) - 1.0)
    sweep_actor.set_pose(
        type(sweep_actor.pose)(actor_position, baseline_actor_quaternion)
    )
    if callable(getattr(sweep_actor, "set_velocity", None)):
        sweep_actor.set_velocity(np.zeros(3, dtype=np.float32))
    if callable(getattr(sweep_actor, "set_angular_velocity", None)):
        sweep_actor.set_angular_velocity(np.zeros(3, dtype=np.float32))
    camera_position = np.asarray(baseline_camera_position, dtype=np.float64).copy()
    camera_position[0] += float(camera_amplitude) * np.sin(phase)
    camera_position[1] += float(camera_amplitude) * (np.cos(phase) - 1.0)
    sweep_camera.camera.set_local_pose(
        type(sweep_camera.camera_cfg.pose)(
            camera_position,
            baseline_camera_quaternion,
        )
    )
    scene = getattr(raw_env, "scene", getattr(raw_env, "_scene", None))
    if scene is None:
        raise RuntimeError("Raw SimplerEnv environment does not expose its render scene.")
    raw_observation = None
    for _ in range(max(int(render_flush_frames), 1)):
        scene.step()
        scene.update_render()
        raw_observation = raw_env.get_obs()
    assert raw_observation is not None
    for camera_images in raw_observation.get("image", {}).values():
        if "rgb" not in camera_images and "Color" in camera_images:
            camera_images["rgb"] = np.clip(
                np.asarray(camera_images["Color"])[..., :3] * 255.0,
                0.0,
                255.0,
            ).astype(np.uint8)
    processed_observation = observation_adapter._process_observation(raw_observation)
    return {
        key: value if isinstance(value, str) else np.expand_dims(np.asarray(value), axis=0)
        for key, value in processed_observation.items()
    }


def _policy_image_mae(
    previous_observation: dict[str, Any] | None,
    current_observation: dict[str, Any],
) -> float:
    if previous_observation is None:
        return 0.0
    errors = []
    for key in sorted(current_observation["video"]):
        previous = np.asarray(previous_observation["video"][key], dtype=np.float32)
        current = np.asarray(current_observation["video"][key], dtype=np.float32)
        errors.append(float(np.mean(np.abs(current - previous))))
    return float(np.mean(errors)) if errors else 0.0


def _rendered_frame_sweep_observation(
    base_observation: dict[str, Any],
    *,
    step: int,
    pixels_per_step: int,
) -> dict[str, Any]:
    observation = _copy_nested_arrays(base_observation)
    shift = max(int(step) * int(pixels_per_step), 0)
    for key, value in observation.items():
        if not str(key).startswith("video.") or shift == 0:
            continue
        frames = np.asarray(value)
        width = int(frames.shape[-2])
        bounded_shift = min(shift, max(width - 1, 0))
        shifted = np.roll(frames, shift=bounded_shift, axis=-2)
        if bounded_shift > 0:
            edge = np.take(frames, indices=[0], axis=-2)
            repeat_shape = [1] * frames.ndim
            repeat_shape[-2] = bounded_shift
            leading_edge = np.tile(edge, repeat_shape)
            leading_slice = [slice(None)] * frames.ndim
            leading_slice[-2] = slice(0, bounded_shift)
            shifted[tuple(leading_slice)] = leading_edge
        observation[key] = shifted
    return observation


def _plot_comparison(rows: list[dict[str, Any]], path: Path) -> None:
    rows = [row for row in rows if not bool(row["cache_warmup"])]
    if not rows:
        return
    x = [int(row["step"]) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(
        x,
        [row["full_cache_representation_error"]["mae"] for row in rows],
        label="full cache reuse",
    )
    axes[0].plot(
        x,
        [row["first_last_representation_error"]["mae"] for row in rows],
        label="first/last correction",
    )
    axes[0].set_ylabel("backbone representation MAE")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        x,
        [row["full_cache_executed_prefix_error"]["overall"]["mae"] for row in rows],
        label="full cache reuse",
    )
    axes[1].plot(
        x,
        [row["first_last_executed_prefix_error"]["overall"]["mae"] for row in rows],
        label="first/last correction",
    )
    axes[1].set_xlabel("fixed-trajectory decision step")
    axes[1].set_ylabel("executed action MAE")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(cfg: CacheComparisonConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not Path(cfg.model_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {cfg.model_path}")
    if not cfg.env_names:
        raise ValueError("At least one env name is required.")
    if str(cfg.trajectory_driver).strip().lower() not in {
        "reset_sequence",
        "physics_only",
        "policy_action",
        "qpos_sweep",
        "scene_sweep",
        "rendered_frame_sweep",
    }:
        raise ValueError(
            "trajectory_driver must be 'rendered_frame_sweep', 'scene_sweep', "
            "'reset_sequence', "
            "'physics_only', or 'policy_action'."
        )
    write_json(output_dir / "config.json", asdict(cfg))

    print(f"Loading GR00T from {cfg.model_path}", flush=True)
    policy = Gr00tPolicy(
        embodiment_tag=get_embodiment_tag_from_env_name(cfg.env_names[0]),
        model_path=cfg.model_path,
        device=cfg.gr00t_device,
        strict=cfg.strict,
    )
    adapter = Gr00tPolicyTemporalSkipAdapter(policy)
    adapter.configure_static_inference_cache(True)

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
                observations: list[dict[str, Any]] = []
                ground_truth_actions: list[dict[str, Any]] = []
                inference_seeds: list[int] = []
                rows: list[dict[str, Any]] = []
                previous_ground_truth_cache = None
                previous_middle_cache = None
                previous_policy_obs = None
                previous_probe_feature = None
                rendered_frame_sweep_base = None
                qpos_baseline = None
                sweep_actor = None
                baseline_actor_position = None
                baseline_actor_quaternion = None
                sweep_camera = None
                baseline_camera_position = None
                baseline_camera_quaternion = None
                driver = str(cfg.trajectory_driver).strip().lower()
                if driver in {
                    "scene_sweep",
                    "qpos_sweep",
                }:
                    _, raw_env = _find_raw_simpler_env(sim_env)
                    qpos_baseline = np.asarray(
                        raw_env.agent.robot.get_qpos(), dtype=np.float64
                    ).copy()
                    sweep_actor = getattr(raw_env, "episode_source_obj", None)
                    if sweep_actor is None:
                        raise RuntimeError(
                            "Scene-sweep driver requires an episode_source_obj actor."
                        )
                    baseline_actor_position = np.asarray(
                        sweep_actor.pose.p, dtype=np.float64
                    ).copy()
                    baseline_actor_quaternion = np.asarray(
                        sweep_actor.pose.q, dtype=np.float64
                    ).copy()
                    sweep_camera = raw_env._cameras.get("3rd_view_camera")
                    if sweep_camera is None:
                        raise RuntimeError(
                            "Scene-sweep driver requires the 3rd_view_camera."
                        )
                    baseline_camera_position = np.asarray(
                        sweep_camera.camera_cfg.pose.p, dtype=np.float64
                    ).copy()
                    baseline_camera_quaternion = np.asarray(
                        sweep_camera.camera_cfg.pose.q, dtype=np.float64
                    ).copy()
                    for warmup_step in range(-cfg.render_pipeline_warmup_steps, 1):
                        sim_obs = _scene_sweep_observation(
                            sim_env,
                            baseline_qpos=qpos_baseline,
                            sweep_actor=sweep_actor,
                            baseline_actor_position=baseline_actor_position,
                            baseline_actor_quaternion=baseline_actor_quaternion,
                            sweep_camera=sweep_camera,
                            baseline_camera_position=baseline_camera_position,
                            baseline_camera_quaternion=baseline_camera_quaternion,
                            step=warmup_step,
                            total_steps=cfg.max_trajectory_steps,
                            qpos_amplitude=cfg.qpos_sweep_amplitude,
                            object_amplitude=cfg.object_sweep_amplitude,
                            camera_amplitude=cfg.camera_sweep_amplitude,
                            render_flush_frames=cfg.render_flush_frames,
                        )
                elif driver == "rendered_frame_sweep":
                    rendered_frame_sweep_base = _copy_nested_arrays(sim_obs)
                terminated = False
                truncated = False

                for step in range(cfg.max_trajectory_steps):
                    policy_obs = _build_policy_observation(sim_obs, policy.modality_configs)
                    previous_frame_image_mae = _policy_image_mae(
                        previous_policy_obs,
                        policy_obs,
                    )
                    prepared = adapter.prepare_inputs(policy_obs)
                    with torch.inference_mode():
                        probe_output, action_input = adapter.get_first_block_gate_probe(
                            prepared["collated_inputs"]
                        )
                        ground_truth_backbone, current_middle_cache = (
                            adapter.continue_backbone_from_probe_with_middle_cache(probe_output)
                        )
                    current_probe_feature = probe_output["hidden_states"]
                    previous_frame_probe_cka = (
                        None
                        if previous_probe_feature is None
                        else SkipController._feature_cka_similarity(
                            previous_probe_feature,
                            current_probe_feature,
                        )
                    )

                    step_inference_seed = int(
                        cfg.inference_seed + trajectory_index * cfg.max_trajectory_steps + step
                    )
                    _set_inference_seed(step_inference_seed)
                    with torch.inference_mode():
                        ground_truth_pred, ground_truth_cache = adapter.run_action_head(
                            backbone_output=ground_truth_backbone,
                            action_input=action_input,
                            reuse_strategy=None,
                        )
                    ground_truth_action = adapter.decode_action(
                        ground_truth_pred,
                        prepared["state"],
                    )

                    cache_warmup = previous_ground_truth_cache is None
                    if cache_warmup:
                        full_cache_action = _copy_nested_arrays(ground_truth_action)
                        first_last_action = _copy_nested_arrays(ground_truth_action)
                        full_cache_features = ground_truth_cache.backbone_features
                        first_last_features = ground_truth_cache.backbone_features
                        middle_cache_used = False
                    else:
                        _set_inference_seed(step_inference_seed)
                        with torch.inference_mode():
                            full_cache_pred, _ = adapter.run_action_head(
                                backbone_cache=previous_ground_truth_cache,
                                action_input=action_input,
                                reuse_strategy=None,
                            )
                        full_cache_action = adapter.decode_action(
                            full_cache_pred,
                            prepared["state"],
                        )
                        full_cache_features = previous_ground_truth_cache.backbone_features

                        with torch.inference_mode():
                            first_last_backbone, _, middle_cache_used = (
                                adapter.continue_backbone_from_middle_cache(
                                    probe_output,
                                    previous_middle_cache,
                                )
                            )
                        _set_inference_seed(step_inference_seed)
                        with torch.inference_mode():
                            first_last_pred, first_last_cache = adapter.run_action_head(
                                backbone_output=first_last_backbone,
                                action_input=action_input,
                                reuse_strategy=None,
                            )
                        first_last_action = adapter.decode_action(
                            first_last_pred,
                            prepared["state"],
                        )
                        first_last_features = first_last_cache.backbone_features

                    full_cache_representation_error = _tensor_error(
                        ground_truth_cache.backbone_features,
                        full_cache_features,
                    )
                    first_last_representation_error = _tensor_error(
                        ground_truth_cache.backbone_features,
                        first_last_features,
                    )
                    row = {
                        "trajectory_index": trajectory_index,
                        "env_name": env_name,
                        "seed": trajectory_seed,
                        "step": step,
                        "inference_seed": step_inference_seed,
                        "cache_warmup": cache_warmup,
                        "middle_cache_used": bool(middle_cache_used),
                        "previous_frame_image_mae": previous_frame_image_mae,
                        "previous_frame_probe_cka": previous_frame_probe_cka,
                        "image_changed": bool(
                            previous_frame_image_mae > cfg.changed_frame_mae_threshold
                        ),
                        "full_cache_representation_error": full_cache_representation_error,
                        "first_last_representation_error": first_last_representation_error,
                        "full_cache_representation_error_wrapped": {
                            "overall": full_cache_representation_error
                        },
                        "first_last_representation_error_wrapped": {
                            "overall": first_last_representation_error
                        },
                        "full_cache_action_chunk_error": action_dict_error(
                            ground_truth_action,
                            full_cache_action,
                        ),
                        "first_last_action_chunk_error": action_dict_error(
                            ground_truth_action,
                            first_last_action,
                        ),
                        "full_cache_executed_prefix_error": action_dict_error(
                            ground_truth_action,
                            full_cache_action,
                            prefix_steps=cfg.n_action_steps,
                        ),
                        "first_last_executed_prefix_error": action_dict_error(
                            ground_truth_action,
                            first_last_action,
                            prefix_steps=cfg.n_action_steps,
                        ),
                    }
                    rows.append(row)
                    all_rows.append(row)
                    observations.append(_copy_nested_arrays(policy_obs))
                    ground_truth_actions.append(_copy_nested_arrays(ground_truth_action))
                    inference_seeds.append(step_inference_seed)
                    previous_ground_truth_cache = ground_truth_cache
                    previous_middle_cache = current_middle_cache
                    previous_policy_obs = policy_obs
                    previous_probe_feature = current_probe_feature.detach()

                    driver = str(cfg.trajectory_driver).strip().lower()
                    if driver == "reset_sequence":
                        sim_obs, _ = sim_env.reset(seed=trajectory_seed + step + 1)
                        terminated = False
                        truncated = False
                    elif driver == "rendered_frame_sweep":
                        assert rendered_frame_sweep_base is not None
                        sim_obs = _rendered_frame_sweep_observation(
                            rendered_frame_sweep_base,
                            step=step + 1,
                            pixels_per_step=cfg.frame_shift_pixels_per_step,
                        )
                        terminated = False
                        truncated = False
                    elif driver in {"scene_sweep", "qpos_sweep"}:
                        assert qpos_baseline is not None
                        assert sweep_actor is not None
                        assert baseline_actor_position is not None
                        assert baseline_actor_quaternion is not None
                        assert sweep_camera is not None
                        assert baseline_camera_position is not None
                        assert baseline_camera_quaternion is not None
                        sim_obs = _scene_sweep_observation(
                            sim_env,
                            baseline_qpos=qpos_baseline,
                            sweep_actor=sweep_actor,
                            baseline_actor_position=baseline_actor_position,
                            baseline_actor_quaternion=baseline_actor_quaternion,
                            sweep_camera=sweep_camera,
                            baseline_camera_position=baseline_camera_position,
                            baseline_camera_quaternion=baseline_camera_quaternion,
                            step=step + 1,
                            total_steps=cfg.max_trajectory_steps,
                            qpos_amplitude=cfg.qpos_sweep_amplitude,
                            object_amplitude=cfg.object_sweep_amplitude,
                            camera_amplitude=cfg.camera_sweep_amplitude,
                            render_flush_frames=cfg.render_flush_frames,
                        )
                        terminated = False
                        truncated = False
                    else:
                        sim_obs, terminated, truncated = _advance_simulation(
                            sim_env,
                            trajectory_driver=cfg.trajectory_driver,
                            reference_action=ground_truth_action,
                        )
                    if terminated or truncated:
                        break

                name = f"trajectory_{trajectory_index:04d}"
                metadata = {
                    "trajectory_index": trajectory_index,
                    "env_name": env_name,
                    "seed": trajectory_seed,
                    "steps": len(rows),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "reset_info": str(reset_info),
                }
                _save_trajectory(
                    output_dir / "trajectories" / f"{name}.npz",
                    observations=observations,
                    reference_actions=ground_truth_actions,
                    inference_seeds=inference_seeds,
                    metadata=metadata,
                )
                trajectory_summary = {**metadata, **_comparison_summary(rows)}
                changed_fraction = float(
                    trajectory_summary["observation_change"]["changed_frame_fraction"]
                )
                trajectory_summary["observation_change"]["quality_check_passed"] = bool(
                    changed_fraction >= cfg.min_changed_frame_fraction
                )
                trajectory_summaries.append(trajectory_summary)
                write_json(output_dir / "summaries" / f"{name}.json", trajectory_summary)
                write_jsonl(output_dir / "steps.jsonl", all_rows)
                print(
                    f"[{name}] steps={len(rows)} "
                    f"full_cache_action_mae="
                    f"{trajectory_summary['full_cache']['executed_prefix_error']['mae']:.6g} "
                    f"first_last_action_mae="
                    f"{trajectory_summary['first_last_correction']['executed_prefix_error']['mae']:.6g} "
                    f"changed_frames={changed_fraction:.1%}",
                    flush=True,
                )
                if changed_fraction < cfg.min_changed_frame_fraction:
                    raise RuntimeError(
                        "Open-loop trajectory is visually degenerate: "
                        f"only {changed_fraction:.1%} of transitions exceeded image MAE "
                        f"{cfg.changed_frame_mae_threshold}."
                    )
                trajectory_index += 1
        finally:
            sim_env.close()

    summary = {
        "trajectories": len(trajectory_summaries),
        "decision_steps": len(all_rows),
        "wall_seconds": float(time.perf_counter() - started_at),
        **_comparison_summary(all_rows),
        "by_env": {
            env_name: _comparison_summary(
                [row for row in all_rows if row["env_name"] == env_name]
            )
            for env_name in cfg.env_names
        },
        "trajectory_summaries": trajectory_summaries,
    }
    write_json(output_dir / "summary.json", summary)
    _plot_comparison(all_rows, output_dir / "cache_comparison.png")
    print(json.dumps(summary["first_last_relative_error_reduction"], sort_keys=True), flush=True)


def cli_main() -> None:
    main(tyro.cli(CacheComparisonConfig))


if __name__ == "__main__":
    cli_main()
