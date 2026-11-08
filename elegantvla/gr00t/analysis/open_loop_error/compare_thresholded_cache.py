#!/usr/bin/env python3

"""Replay saved observations through the threshold scheduler and compare caches."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import tyro

from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.policy.gr00t_policy import Gr00tPolicy
from elegantvla.gr00t.core.adapter import Gr00tPolicyTemporalSkipAdapter
from elegantvla.gr00t.entrypoints._common import write_json, write_jsonl
from elegantvla.gr00t.analysis.open_loop_error.compare_backbone_cache import (
    DEFAULT_MODEL_PATH,
    _comparison_summary,
    _copy_nested_arrays,
    _method_summary,
    _policy_image_mae,
    _reduction,
    _set_inference_seed,
    _tensor_error,
)
from elegantvla.gr00t.analysis.open_loop_error.metrics import action_dict_error
from elegantvla.gr00t.core.rule_policy import RuleSchedulePolicy
from elegantvla.gr00t.core.types import BackboneAction
from elegantvla.gr00t.core.wrapper import SkipController


DEFAULT_INPUT_DIR = (
    "outputs/gr00t/open_loop_error/"
    "20260722_widowx3_rendered_sweep_gt_vs_cache20"
)


@dataclass
class Config:
    input_dir: str = DEFAULT_INPUT_DIR
    output_dir: str = f"{DEFAULT_INPUT_DIR}/threshold_replay"
    model_path: str = DEFAULT_MODEL_PATH
    gate_threshold: float = 0.95
    free1_similarity_threshold: float | None = 0.98
    free2_similarity_threshold: float | None = 0.99
    free3_similarity_threshold: float | None = 0.995
    max_reuse_steps: int | None = 5
    speed_smoothing_window: int = 4
    n_action_steps: int = 1
    gr00t_device: str = "cuda"
    strict: bool = True


def _load_trajectory(path: Path) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    payload = np.load(path, allow_pickle=False)
    metadata = json.loads(str(payload["metadata_json"].item()))
    inference_seeds = np.asarray(payload["inference_seeds"], dtype=np.int64)
    observations: list[dict[str, Any]] = []
    for step in range(len(inference_seeds)):
        observation: dict[str, dict[str, Any]] = {
            "video": {},
            "state": {},
            "language": {},
        }
        for key in payload.files:
            if not key.startswith("observation."):
                continue
            _, modality, modality_key = key.split(".", 2)
            value = payload[key]
            if modality == "language":
                observation[modality][modality_key] = [[str(value[step])]]
            else:
                observation[modality][modality_key] = value[step : step + 1].copy()
        observations.append(observation)
    return observations, inference_seeds, metadata


def _error_row(
    *,
    metadata: dict[str, Any],
    step: int,
    inference_seed: int,
    feature_similarity: float,
    backbone_action: BackboneAction,
    image_free_skip: bool,
    previous_frame_image_mae: float,
    ground_truth_cache: Any,
    full_cache_features: torch.Tensor,
    first_last_features: torch.Tensor,
    first_only_features: torch.Tensor,
    ground_truth_action: dict[str, Any],
    full_cache_action: dict[str, Any],
    first_last_action: dict[str, Any],
    first_only_action: dict[str, Any],
    n_action_steps: int,
) -> dict[str, Any]:
    full_representation_error = _tensor_error(
        ground_truth_cache.backbone_features,
        full_cache_features,
    )
    first_last_representation_error = _tensor_error(
        ground_truth_cache.backbone_features,
        first_last_features,
    )
    first_only_representation_error = _tensor_error(
        ground_truth_cache.backbone_features,
        first_only_features,
    )
    use_cache = backbone_action != BackboneAction.REFRESH
    return {
        "trajectory_index": int(metadata["trajectory_index"]),
        "env_name": str(metadata["env_name"]),
        "seed": int(metadata["seed"]),
        "step": int(step),
        "inference_seed": int(inference_seed),
        "cache_warmup": bool(step == 0),
        "feature_similarity": float(feature_similarity),
        "backbone_action": int(backbone_action),
        "backbone_action_name": backbone_action.name,
        "threshold_selected_reuse": bool(use_cache),
        "image_free_skip": bool(image_free_skip),
        "probe_reuse": bool(use_cache and not image_free_skip),
        "previous_frame_image_mae": float(previous_frame_image_mae),
        "image_changed": bool(previous_frame_image_mae > 0.01),
        "full_cache_representation_error": full_representation_error,
        "first_last_representation_error": first_last_representation_error,
        "full_cache_representation_error_wrapped": {
            "overall": full_representation_error,
        },
        "first_last_representation_error_wrapped": {
            "overall": first_last_representation_error,
        },
        "first_only_representation_error": first_only_representation_error,
        "first_only_representation_error_wrapped": {
            "overall": first_only_representation_error,
        },
        "full_cache_action_chunk_error": action_dict_error(
            ground_truth_action,
            full_cache_action,
        ),
        "first_last_action_chunk_error": action_dict_error(
            ground_truth_action,
            first_last_action,
        ),
        "first_only_action_chunk_error": action_dict_error(
            ground_truth_action,
            first_only_action,
        ),
        "full_cache_executed_prefix_error": action_dict_error(
            ground_truth_action,
            full_cache_action,
            prefix_steps=n_action_steps,
        ),
        "first_last_executed_prefix_error": action_dict_error(
            ground_truth_action,
            first_last_action,
            prefix_steps=n_action_steps,
        ),
        "first_only_executed_prefix_error": action_dict_error(
            ground_truth_action,
            first_only_action,
            prefix_steps=n_action_steps,
        ),
    }


def _three_way_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _comparison_summary(rows)
    evaluated_rows = [row for row in rows if not bool(row["cache_warmup"])]
    first_only = _method_summary(evaluated_rows, "first_only")
    summary["first_only_method"] = (
        "current_first_layer_plus_previous_post_first_delta_without_last_recompute"
    )
    summary["first_only_correction"] = first_only
    summary["first_only_relative_error_reduction"] = {
        metric_group: {
            metric: _reduction(
                float(summary["full_cache"][metric_group][metric]),
                float(first_only[metric_group][metric]),
            )
            for metric in ("mae", "mse")
        }
        for metric_group in (
            "representation_error",
            "action_chunk_error",
            "executed_prefix_error",
        )
    }
    summary["first_only_vs_first_last_relative_error_reduction"] = {
        metric_group: {
            metric: _reduction(
                float(summary["first_last_correction"][metric_group][metric]),
                float(first_only[metric_group][metric]),
            )
            for metric in ("mae", "mse")
        }
        for metric_group in (
            "representation_error",
            "action_chunk_error",
            "executed_prefix_error",
        )
    }
    return summary


def _subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all_scheduled_steps": _three_way_summary(rows),
        "selected_reuse_steps": _three_way_summary(
            [row for row in rows if row["threshold_selected_reuse"]]
        ),
        "probe_reuse_steps": _three_way_summary(
            [row for row in rows if row["probe_reuse"]]
        ),
        "image_free_reuse_steps": _three_way_summary(
            [row for row in rows if row["image_free_skip"]]
        ),
    }


def main(cfg: Config) -> None:
    input_dir = Path(cfg.input_dir)
    trajectory_paths = sorted((input_dir / "trajectories").glob("trajectory_*.npz"))
    if not trajectory_paths:
        raise FileNotFoundError(f"No saved trajectories found under {input_dir / 'trajectories'}")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", asdict(cfg))

    _, _, first_metadata = _load_trajectory(trajectory_paths[0])
    policy = Gr00tPolicy(
        embodiment_tag=get_embodiment_tag_from_env_name(first_metadata["env_name"]),
        model_path=cfg.model_path,
        device=cfg.gr00t_device,
        strict=cfg.strict,
    )
    adapter = Gr00tPolicyTemporalSkipAdapter(policy)
    adapter.configure_static_inference_cache(True)
    rule_policy = RuleSchedulePolicy(
        gate_threshold=cfg.gate_threshold,
        free1_similarity_threshold=cfg.free1_similarity_threshold,
        free2_similarity_threshold=cfg.free2_similarity_threshold,
        free3_similarity_threshold=cfg.free3_similarity_threshold,
    )

    all_rows: list[dict[str, Any]] = []
    trajectory_summaries: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    for trajectory_path in trajectory_paths:
        observations, inference_seeds, metadata = _load_trajectory(trajectory_path)
        ctrl = SkipController(
            enabled=True,
            max_reuse_steps=cfg.max_reuse_steps,
            speed_smoothing_window=cfg.speed_smoothing_window,
        )
        full_cache_state = None
        first_last_cache_state = None
        first_only_cache_state = None
        last_feature_similarity = 0.0
        previous_observation = None
        rows: list[dict[str, Any]] = []

        for step, (observation, inference_seed) in enumerate(
            zip(observations, inference_seeds, strict=True)
        ):
            prepared = adapter.prepare_inputs(observation)
            previous_frame_image_mae = _policy_image_mae(
                previous_observation,
                observation,
            )
            with torch.inference_mode():
                probe_output, action_input = adapter.get_first_block_gate_probe(
                    prepared["collated_inputs"]
                )
                ground_truth_backbone, current_middle_cache = (
                    adapter.continue_backbone_from_probe_with_middle_cache(probe_output)
                )

            if ctrl._no_image_budget > 0:
                schedule_observation = ctrl.build_schedule_observation(
                    observation,
                    feature_similarity=last_feature_similarity,
                )
            else:
                schedule_observation = ctrl.build_schedule_observation(
                    observation,
                    current_gate_feature=probe_output["hidden_states"],
                )
                last_feature_similarity = float(schedule_observation.feature_similarity)
            schedule_action = rule_policy.select_action(
                schedule_observation.as_gym_observation(),
                backbone_action_mask=ctrl.build_backbone_action_mask(),
            )
            backbone_action = ctrl.resolve_backbone_action(schedule_action.backbone_action)
            use_cache = backbone_action != BackboneAction.REFRESH
            image_free_skip = bool(use_cache and ctrl._no_image_budget > 0)

            _set_inference_seed(int(inference_seed))
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

            if not use_cache:
                full_cache_action = _copy_nested_arrays(ground_truth_action)
                first_last_action = _copy_nested_arrays(ground_truth_action)
                first_only_action = _copy_nested_arrays(ground_truth_action)
                full_cache_state = ground_truth_cache
                first_last_cache_state = ground_truth_cache
                first_only_cache_state = ground_truth_cache
                full_cache_features = ground_truth_cache.backbone_features
                first_last_features = ground_truth_cache.backbone_features
                first_only_features = ground_truth_cache.backbone_features
                ctrl._backbone_middle_cache = current_middle_cache
            else:
                if (
                    full_cache_state is None
                    or first_last_cache_state is None
                    or first_only_cache_state is None
                ):
                    raise RuntimeError("Threshold scheduler selected reuse before cache warmup.")
                _set_inference_seed(int(inference_seed))
                with torch.inference_mode():
                    full_cache_pred, _ = adapter.run_action_head(
                        backbone_cache=full_cache_state,
                        action_input=action_input,
                        reuse_strategy=None,
                    )
                full_cache_action = adapter.decode_action(
                    full_cache_pred,
                    prepared["state"],
                )
                full_cache_features = full_cache_state.backbone_features

                if image_free_skip:
                    first_last_backbone = None
                else:
                    with torch.inference_mode():
                        first_last_backbone, refreshed_middle_cache, middle_cache_used = (
                            adapter.continue_backbone_from_middle_cache(
                                probe_output,
                                ctrl._backbone_middle_cache,
                            )
                        )
                    if refreshed_middle_cache is not None or not middle_cache_used:
                        raise RuntimeError("Saved trajectory replay could not use middle cache.")
                _set_inference_seed(int(inference_seed))
                with torch.inference_mode():
                    if image_free_skip:
                        first_last_pred, next_first_last_cache = adapter.run_action_head(
                            backbone_cache=first_last_cache_state,
                            action_input=action_input,
                            reuse_strategy=None,
                        )
                    else:
                        first_last_pred, next_first_last_cache = adapter.run_action_head(
                            backbone_output=first_last_backbone,
                            action_input=action_input,
                            reuse_strategy=None,
                        )
                first_last_action = adapter.decode_action(
                    first_last_pred,
                    prepared["state"],
                )
                first_last_features = next_first_last_cache.backbone_features
                if not image_free_skip:
                    first_last_cache_state = next_first_last_cache

                if image_free_skip:
                    first_only_backbone = None
                else:
                    with torch.inference_mode():
                        first_only_backbone, refreshed_post_first_cache, post_first_used = (
                            adapter.continue_backbone_from_post_first_cache(
                                probe_output,
                                ctrl._backbone_middle_cache,
                            )
                        )
                    if refreshed_post_first_cache is not None or not post_first_used:
                        raise RuntimeError("Saved trajectory replay could not use post-first cache.")
                _set_inference_seed(int(inference_seed))
                with torch.inference_mode():
                    if image_free_skip:
                        first_only_pred, next_first_only_cache = adapter.run_action_head(
                            backbone_cache=first_only_cache_state,
                            action_input=action_input,
                            reuse_strategy=None,
                        )
                    else:
                        first_only_pred, next_first_only_cache = adapter.run_action_head(
                            backbone_output=first_only_backbone,
                            action_input=action_input,
                            reuse_strategy=None,
                        )
                first_only_action = adapter.decode_action(
                    first_only_pred,
                    prepared["state"],
                )
                first_only_features = next_first_only_cache.backbone_features
                if not image_free_skip:
                    first_only_cache_state = next_first_only_cache

            row = _error_row(
                metadata=metadata,
                step=step,
                inference_seed=int(inference_seed),
                feature_similarity=float(schedule_observation.feature_similarity),
                backbone_action=backbone_action,
                image_free_skip=image_free_skip,
                previous_frame_image_mae=previous_frame_image_mae,
                ground_truth_cache=ground_truth_cache,
                full_cache_features=full_cache_features,
                first_last_features=first_last_features,
                first_only_features=first_only_features,
                ground_truth_action=ground_truth_action,
                full_cache_action=full_cache_action,
                first_last_action=first_last_action,
                first_only_action=first_only_action,
                n_action_steps=cfg.n_action_steps,
            )
            rows.append(row)
            all_rows.append(row)
            ctrl._backbone_cache = first_last_cache_state
            ctrl.record_motion(observation)
            ctrl.record_step(
                use_backbone_cache=use_cache,
                image_free_skip=image_free_skip,
                gate_feature=None if image_free_skip else probe_output["hidden_states"],
                backbone_action=backbone_action,
            )
            previous_observation = observation

        trajectory_summary = {
            **metadata,
            "schedule_counts": dict(Counter(row["backbone_action_name"] for row in rows)),
            **_subset_summary(rows),
        }
        trajectory_summaries.append(trajectory_summary)
        write_json(
            output_dir / "summaries" / f"trajectory_{metadata['trajectory_index']:04d}.json",
            trajectory_summary,
        )
        write_jsonl(output_dir / "steps.jsonl", all_rows)
        selected = trajectory_summary["selected_reuse_steps"]
        print(
            f"[{metadata['env_name']}] reuse={selected['full_cache']['evaluated_steps']} "
            f"full={selected['full_cache']['executed_prefix_error']['mae']:.6g} "
            f"first_last={selected['first_last_correction']['executed_prefix_error']['mae']:.6g} "
            f"first_only={selected['first_only_correction']['executed_prefix_error']['mae']:.6g}",
            flush=True,
        )

    summary = {
        "source": "offline_replay_of_saved_npz_trajectories",
        "trajectories": len(trajectory_paths),
        "steps": len(all_rows),
        "wall_seconds": float(time.perf_counter() - started_at),
        "schedule_counts": dict(Counter(row["backbone_action_name"] for row in all_rows)),
        **_subset_summary(all_rows),
        "by_env": {
            env_name: _subset_summary(
                [row for row in all_rows if row["env_name"] == env_name]
            )
            for env_name in sorted({row["env_name"] for row in all_rows})
        },
        "trajectory_summaries": trajectory_summaries,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary["schedule_counts"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
