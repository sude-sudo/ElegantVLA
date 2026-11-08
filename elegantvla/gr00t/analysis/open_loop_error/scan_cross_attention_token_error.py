#!/usr/bin/env python3

"""Scan layer subsets using representations actually consumed by DiT cross-attention."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import tyro

from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.policy.gr00t_policy import Gr00tPolicy
from elegantvla.gr00t.core.adapter import Gr00tPolicyTemporalSkipAdapter
from elegantvla.gr00t.entrypoints._common import write_json, write_jsonl
from elegantvla.gr00t.analysis.open_loop_error.compare_backbone_cache import DEFAULT_MODEL_PATH
from elegantvla.gr00t.analysis.open_loop_error.compare_thresholded_cache import _load_trajectory
from elegantvla.gr00t.analysis.open_loop_error.scan_layer_recompute_subsets import (
    DEFAULT_INPUT_DIR,
    DEFAULT_SCHEDULE_PATH,
    _candidate_subsets,
    _load_schedule,
)


DEFAULT_ACTION_SUMMARIES = [
    f"{DEFAULT_INPUT_DIR}/layer_subset_scan_k1_k2_k3/summary.json",
    f"{DEFAULT_INPUT_DIR}/layer_subset_scan_k4/summary.json",
]


@dataclass
class Config:
    input_dir: str = DEFAULT_INPUT_DIR
    schedule_path: str = DEFAULT_SCHEDULE_PATH
    output_dir: str = f"{DEFAULT_INPUT_DIR}/cross_attention_token_scan_k1_k4"
    action_summary_paths: list[str] = field(
        default_factory=lambda: list(DEFAULT_ACTION_SUMMARIES)
    )
    model_path: str = DEFAULT_MODEL_PATH
    min_total_recompute_layers: int = 1
    max_total_recompute_layers: int = 4
    explicit_recompute_layers: str | None = None
    cached_prefix_max_layer: int | None = None
    evaluate_all_non_refresh_steps: bool = False
    gr00t_device: str = "cuda"
    strict: bool = True


def _masked_token_error(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    token_mask: torch.Tensor,
) -> dict[str, float | int]:
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError(
            f"Representation shapes differ: reference={reference.shape}, candidate={candidate.shape}"
        )
    if tuple(token_mask.shape) != tuple(reference.shape[:2]):
        raise ValueError(
            f"Token mask {token_mask.shape} does not match representation {reference.shape[:2]}."
        )
    selected_reference = reference.float()[token_mask.bool()]
    selected_candidate = candidate.float()[token_mask.bool()]
    if selected_reference.numel() == 0:
        return {
            "tokens": 0,
            "hidden_dim": int(reference.shape[-1]),
            "mae": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "mean_token_cosine_distance": 0.0,
            "mean_token_relative_l2": 0.0,
            "max_token_mae": 0.0,
        }

    delta = selected_candidate - selected_reference
    per_token_mae = delta.abs().mean(dim=-1)
    per_token_mse = delta.square().mean(dim=-1)
    cosine_distance = 1.0 - F.cosine_similarity(
        selected_reference,
        selected_candidate,
        dim=-1,
        eps=1e-12,
    )
    relative_l2 = torch.linalg.vector_norm(delta, dim=-1) / torch.linalg.vector_norm(
        selected_reference,
        dim=-1,
    ).clamp_min(1e-12)
    mse = per_token_mse.mean()
    return {
        "tokens": int(selected_reference.shape[0]),
        "hidden_dim": int(selected_reference.shape[-1]),
        "mae": float(per_token_mae.mean().item()),
        "mse": float(mse.item()),
        "rmse": float(torch.sqrt(mse).item()),
        "mean_token_cosine_distance": float(cosine_distance.mean().item()),
        "mean_token_relative_l2": float(relative_l2.mean().item()),
        "max_token_mae": float(per_token_mae.max().item()),
    }


def _cross_attention_errors(
    reference_features: torch.Tensor,
    candidate_features: torch.Tensor,
    *,
    backbone_attention_mask: torch.Tensor,
    image_mask: torch.Tensor,
) -> dict[str, Any]:
    valid_mask = backbone_attention_mask.bool()
    image_attention_mask = valid_mask & image_mask.bool()
    non_image_attention_mask = valid_mask & ~image_mask.bool()
    image_error = _masked_token_error(
        reference_features,
        candidate_features,
        image_attention_mask,
    )
    non_image_error = _masked_token_error(
        reference_features,
        candidate_features,
        non_image_attention_mask,
    )
    return {
        "all_cross_attention_tokens": _masked_token_error(
            reference_features,
            candidate_features,
            valid_mask,
        ),
        "image_tokens": image_error,
        "non_image_tokens": non_image_error,
        "group_balanced_mae": float(
            0.5 * (float(image_error["mae"]) + float(non_image_error["mae"]))
        ),
    }


def _mean_error(rows: list[dict[str, Any]], group: str) -> dict[str, float | int]:
    if not rows:
        return {"steps": 0}
    names = (
        "mae",
        "mse",
        "rmse",
        "mean_token_cosine_distance",
        "mean_token_relative_l2",
        "max_token_mae",
    )
    return {
        "steps": len(rows),
        "tokens_per_step": float(
            np.mean([float(row["cross_attention_error"][group]["tokens"]) for row in rows])
        ),
        **{
            name: float(
                np.mean(
                    [float(row["cross_attention_error"][group][name]) for row in rows]
                )
            )
            for name in names
        },
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "evaluated_steps": len(rows),
        "all_cross_attention_tokens": _mean_error(rows, "all_cross_attention_tokens"),
        "image_tokens": _mean_error(rows, "image_tokens"),
        "non_image_tokens": _mean_error(rows, "non_image_tokens"),
        "group_balanced_mae": float(
            np.mean([float(row["cross_attention_error"]["group_balanced_mae"]) for row in rows])
        )
        if rows
        else 0.0,
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    left_centered = left_array - left_array.mean()
    right_centered = right_array - right_array.mean()
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    return None if denominator == 0.0 else float(
        np.dot(left_centered, right_centered) / denominator
    )


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    for rank, index in enumerate(order, start=1):
        ranks[index] = float(rank)
    return ranks


def _load_action_summaries(paths: list[str]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        for candidate in payload["all_candidates"]:
            summaries[str(candidate["subset_key"])] = candidate
    return summaries


def main(cfg: Config) -> None:
    input_dir = Path(cfg.input_dir)
    trajectory_paths = sorted((input_dir / "trajectories").glob("trajectory_*.npz"))
    if not trajectory_paths:
        raise FileNotFoundError(f"No saved trajectories found under {input_dir / 'trajectories'}")
    schedule = _load_schedule(Path(cfg.schedule_path))
    action_summaries = _load_action_summaries(cfg.action_summary_paths)
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
    num_layers = len(adapter.model.backbone.model.language_model.model.layers)
    if (
        cfg.explicit_recompute_layers is not None
        and cfg.cached_prefix_max_layer is not None
    ):
        raise ValueError(
            "explicit_recompute_layers and cached_prefix_max_layer are mutually exclusive."
        )
    if cfg.cached_prefix_max_layer is not None:
        max_cached_layer = int(cfg.cached_prefix_max_layer)
        if max_cached_layer < 1 or max_cached_layer >= num_layers:
            raise ValueError(
                "cached_prefix_max_layer must be between 1 and "
                f"{num_layers - 1}, got {max_cached_layer}."
            )
        subsets = [
            (0, *range(cached_layer + 1, num_layers))
            for cached_layer in range(1, max_cached_layer + 1)
        ]
    elif cfg.explicit_recompute_layers is None:
        subsets = _candidate_subsets(
            num_layers,
            cfg.min_total_recompute_layers,
            cfg.max_total_recompute_layers,
        )
    else:
        explicit_layers = tuple(
            sorted(
                {
                    int(value.strip())
                    for value in cfg.explicit_recompute_layers.split(",")
                    if value.strip()
                }
            )
        )
        if not explicit_layers or explicit_layers[0] != 0:
            raise ValueError("Explicit recomputation must include probe layer 0.")
        if explicit_layers[-1] >= num_layers:
            raise ValueError(
                f"Explicit recompute layer {explicit_layers[-1]} is outside a "
                f"{num_layers}-layer backbone."
            )
        subsets = [explicit_layers]
    print(
        f"Scanning {len(subsets)} subsets with DiT cross-attention token masks",
        flush=True,
    )

    candidate_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    for trajectory_path in trajectory_paths:
        observations, inference_seeds, metadata = _load_trajectory(trajectory_path)
        anchor_layer_cache = None
        anchor_features = None
        anchor_step = None
        scanned_steps = 0
        for step, (observation, inference_seed) in enumerate(
            zip(observations, inference_seeds, strict=True)
        ):
            schedule_row = schedule[(int(metadata["trajectory_index"]), step)]
            is_refresh = schedule_row["backbone_action_name"] == "REFRESH"
            is_probe_reuse = bool(schedule_row["probe_reuse"])
            if (
                not is_refresh
                and not is_probe_reuse
                and not cfg.evaluate_all_non_refresh_steps
            ):
                continue

            prepared = adapter.prepare_inputs(observation)
            with torch.inference_mode():
                probe_output, _ = adapter.get_first_block_gate_probe(
                    prepared["collated_inputs"]
                )
                if is_refresh:
                    backbone_output, anchor_layer_cache = (
                        adapter.continue_backbone_from_probe_with_layer_boundary_cache(
                            probe_output
                        )
                    )
                    anchor_output = adapter.process_backbone_output_for_action_head(
                        backbone_output
                    )
                    anchor_features = anchor_output.backbone_features.detach()
                    anchor_step = step
                    continue
                ground_truth_backbone = adapter.continue_backbone_from_probe(probe_output)
                ground_truth_output = adapter.process_backbone_output_for_action_head(
                    ground_truth_backbone
                )

            if anchor_layer_cache is None or anchor_features is None or anchor_step is None:
                raise RuntimeError("Probe reuse was selected before a refresh anchor was cached.")
            reference_features = ground_truth_output.backbone_features
            attention_mask = ground_truth_output.backbone_attention_mask
            image_mask = ground_truth_output.image_mask
            baseline_rows.append(
                {
                    "trajectory_index": int(metadata["trajectory_index"]),
                    "env_name": str(metadata["env_name"]),
                    "step": step,
                    "schedule_action": str(schedule_row["backbone_action_name"]),
                    "probe_reuse": is_probe_reuse,
                    "anchor_step": int(anchor_step),
                    "cache_age": int(step - anchor_step),
                    "cross_attention_error": _cross_attention_errors(
                        reference_features,
                        anchor_features,
                        backbone_attention_mask=attention_mask,
                        image_mask=image_mask,
                    ),
                }
            )

            for recompute_layers in subsets:
                with torch.inference_mode():
                    candidate_backbone, refreshed_cache, cache_used = (
                        adapter.continue_backbone_from_layer_subset_cache(
                            probe_output,
                            anchor_layer_cache,
                            recompute_layers,
                        )
                    )
                    candidate_output = adapter.process_backbone_output_for_action_head(
                        candidate_backbone
                    )
                if refreshed_cache is not None or not cache_used:
                    raise RuntimeError(
                        f"Layer cache could not serve subset {recompute_layers}."
                    )
                candidate_rows.append(
                    {
                        "trajectory_index": int(metadata["trajectory_index"]),
                        "env_name": str(metadata["env_name"]),
                        "step": step,
                        "inference_seed": int(inference_seed),
                        "schedule_action": str(schedule_row["backbone_action_name"]),
                        "probe_reuse": is_probe_reuse,
                        "anchor_step": int(anchor_step),
                        "cache_age": int(step - anchor_step),
                        "subset_key": "-".join(str(index) for index in recompute_layers),
                        "recompute_layers": list(recompute_layers),
                        "total_recompute_layers": len(recompute_layers),
                        "cross_attention_error": _cross_attention_errors(
                            reference_features,
                            candidate_output.backbone_features,
                            backbone_attention_mask=attention_mask,
                            image_mask=image_mask,
                        ),
                    }
                )
            scanned_steps += 1
            print(
                f"[{metadata['env_name']}] step={step} "
                f"probe_scan={scanned_steps} subsets={len(subsets)}",
                flush=True,
            )
        write_jsonl(output_dir / "candidate_rows.jsonl", candidate_rows)
        write_jsonl(output_dir / "full_cache_rows.jsonl", baseline_rows)

    candidates: list[dict[str, Any]] = []
    env_names = sorted({row["env_name"] for row in candidate_rows})
    for recompute_layers in subsets:
        subset_key = "-".join(str(index) for index in recompute_layers)
        rows = [row for row in candidate_rows if row["subset_key"] == subset_key]
        action_summary = action_summaries.get(subset_key)
        candidates.append(
            {
                "subset_key": subset_key,
                "recompute_layers": list(recompute_layers),
                "total_recompute_layers": len(recompute_layers),
                "summary": _summary(rows),
                "by_env": {
                    env_name: _summary([row for row in rows if row["env_name"] == env_name])
                    for env_name in env_names
                },
                "existing_action_summary": (
                    None
                    if action_summary is None
                    else {
                        "executed_prefix_error": action_summary["summary"][
                            "executed_prefix_error"
                        ],
                        "action_chunk_error": action_summary["summary"][
                            "action_chunk_error"
                        ],
                    }
                ),
            }
        )

    ranking = sorted(
        candidates,
        key=lambda item: (
            float(item["summary"]["all_cross_attention_tokens"]["mae"]),
            float(item["summary"]["group_balanced_mae"]),
        ),
    )
    joined = [item for item in ranking if item["existing_action_summary"] is not None]
    token_mae = [
        float(item["summary"]["all_cross_attention_tokens"]["mae"])
        for item in joined
    ]
    action_mae = [
        float(item["existing_action_summary"]["executed_prefix_error"]["mae"])
        for item in joined
    ]
    summary = {
        "source": "paired_offline_scan_of_tokens_used_as_dit_cross_attention_kv",
        "token_definition": {
            "all_cross_attention_tokens": "backbone_attention_mask",
            "image_tokens": "image_mask AND backbone_attention_mask",
            "non_image_tokens": "NOT image_mask AND backbone_attention_mask",
            "feature_location": "after_final_qwen_norm_and_action_head_vlln",
            "aggregation": "mean_hidden_error_per_token_then_mean_over_selected_tokens",
        },
        "num_language_layers": num_layers,
        "candidate_subsets": len(subsets),
        "evaluated_probe_reuse_steps": len(baseline_rows),
        "evaluation_scope": (
            "all_non_refresh_steps"
            if cfg.evaluate_all_non_refresh_steps
            else "probe_reuse_steps"
        ),
        "schedule_counts": dict(
            Counter(row["backbone_action_name"] for row in schedule.values())
        ),
        "wall_seconds_excluding_model_load": float(time.perf_counter() - started_at),
        "full_cache_baseline": _summary(baseline_rows),
        "best_overall": ranking[0],
        "top_20": ranking[:20],
        "correlation_with_existing_action_scan": {
            "candidates": len(joined),
            "pearson_token_mae_vs_executed_action_mae": _pearson(token_mae, action_mae),
            "spearman_token_mae_vs_executed_action_mae": _pearson(
                _ranks(token_mae),
                _ranks(action_mae),
            ),
        },
        "all_candidates": ranking,
    }
    write_json(output_dir / "summary.json", summary)
    print(
        f"Best subset={ranking[0]['subset_key']} "
        "cross_attention_token_mae="
        f"{ranking[0]['summary']['all_cross_attention_tokens']['mae']:.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
