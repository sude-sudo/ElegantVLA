#!/usr/bin/env python3

"""Exhaustively scan small language-layer recomputation subsets on saved frames."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import csv
from itertools import combinations
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
    _set_inference_seed,
    _tensor_error,
)
from elegantvla.gr00t.analysis.open_loop_error.compare_thresholded_cache import _load_trajectory
from elegantvla.gr00t.analysis.open_loop_error.metrics import action_dict_error


DEFAULT_INPUT_DIR = (
    "outputs/gr00t/open_loop_error/"
    "20260722_widowx3_rendered_sweep_gt_vs_cache20"
)
DEFAULT_SCHEDULE_PATH = f"{DEFAULT_INPUT_DIR}/threshold_replay_first_only_gpu2/steps.jsonl"


@dataclass
class Config:
    input_dir: str = DEFAULT_INPUT_DIR
    schedule_path: str = DEFAULT_SCHEDULE_PATH
    output_dir: str = f"{DEFAULT_INPUT_DIR}/layer_subset_scan_k1_k2_k3"
    model_path: str = DEFAULT_MODEL_PATH
    min_total_recompute_layers: int = 1
    max_total_recompute_layers: int = 3
    n_action_steps: int = 1
    gr00t_device: str = "cuda"
    strict: bool = True


def _load_schedule(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    schedule = {
        (int(row["trajectory_index"]), int(row["step"])): row
        for row in rows
    }
    if len(schedule) != len(rows):
        raise ValueError(f"Duplicate trajectory/step keys in {path}")
    return schedule


def _candidate_subsets(
    num_layers: int,
    min_total_recompute_layers: int,
    max_total_recompute_layers: int,
) -> list[tuple[int, ...]]:
    if num_layers < 1:
        raise ValueError("Backbone must expose at least one language layer.")
    min_layers = max(int(min_total_recompute_layers), 1)
    max_layers = min(max(int(max_total_recompute_layers), 1), num_layers)
    if min_layers > max_layers:
        raise ValueError(
            "min_total_recompute_layers cannot exceed max_total_recompute_layers."
        )
    subsets: list[tuple[int, ...]] = []
    for total_layers in range(min_layers, max_layers + 1):
        for additional_layers in combinations(range(1, num_layers), total_layers - 1):
            subsets.append((0, *additional_layers))
    return subsets


def _mean_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    if not rows:
        return {"steps": 0}
    metric_names = (
        "mae",
        "mse",
        "rmse",
        "max_abs",
        "reference_rms",
        "nrmse_by_reference_rms",
        "cosine_similarity",
    )
    return {
        "steps": len(rows),
        **{
            name: float(np.mean([float(row[key][name]) for row in rows]))
            for name in metric_names
        },
    }


def _method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "evaluated_steps": len(rows),
        "representation_error": _mean_metrics(rows, "representation_error"),
        "action_chunk_error": _mean_metrics(rows, "action_chunk_error"),
        "executed_prefix_error": _mean_metrics(rows, "executed_prefix_error"),
    }


def _relative_reduction(reference: float, candidate: float) -> float | None:
    return None if reference == 0.0 else float((reference - candidate) / reference)


def _summarize_candidate(
    rows: list[dict[str, Any]],
    baseline_summary: dict[str, Any],
) -> dict[str, Any]:
    summary = _method_summary(rows)
    summary["relative_error_reduction_vs_full_cache"] = {
        group: {
            metric: _relative_reduction(
                float(baseline_summary[group][metric]),
                float(summary[group][metric]),
            )
            for metric in ("mae", "mse")
        }
        for group in (
            "representation_error",
            "action_chunk_error",
            "executed_prefix_error",
        )
    }
    return summary


def _write_ranking_csv(path: Path, ranking: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "rank",
                "subset_key",
                "recompute_layers",
                "total_recompute_layers",
                "executed_prefix_mae",
                "action_chunk_mae",
                "representation_mae",
                "executed_prefix_reduction_vs_full_cache",
            ),
        )
        writer.writeheader()
        for rank, item in enumerate(ranking, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "subset_key": item["subset_key"],
                    "recompute_layers": " ".join(
                        str(index) for index in item["recompute_layers"]
                    ),
                    "total_recompute_layers": item["total_recompute_layers"],
                    "executed_prefix_mae": item["summary"]["executed_prefix_error"]["mae"],
                    "action_chunk_mae": item["summary"]["action_chunk_error"]["mae"],
                    "representation_mae": item["summary"]["representation_error"]["mae"],
                    "executed_prefix_reduction_vs_full_cache": item["summary"][
                        "relative_error_reduction_vs_full_cache"
                    ]["executed_prefix_error"]["mae"],
                }
            )


def main(cfg: Config) -> None:
    input_dir = Path(cfg.input_dir)
    trajectory_paths = sorted((input_dir / "trajectories").glob("trajectory_*.npz"))
    if not trajectory_paths:
        raise FileNotFoundError(f"No saved trajectories found under {input_dir / 'trajectories'}")
    schedule = _load_schedule(Path(cfg.schedule_path))
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
    subsets = _candidate_subsets(
        num_layers,
        cfg.min_total_recompute_layers,
        cfg.max_total_recompute_layers,
    )
    print(
        f"Scanning {len(subsets)} subsets over {num_layers} layers: "
        "total recompute layers "
        f"{cfg.min_total_recompute_layers}..{cfg.max_total_recompute_layers}",
        flush=True,
    )

    candidate_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    for trajectory_path in trajectory_paths:
        observations, inference_seeds, metadata = _load_trajectory(trajectory_path)
        anchor_layer_cache = None
        anchor_action_cache = None
        scanned_steps = 0
        for step, (observation, inference_seed) in enumerate(
            zip(observations, inference_seeds, strict=True)
        ):
            schedule_row = schedule.get((int(metadata["trajectory_index"]), step))
            if schedule_row is None:
                raise KeyError(
                    f"Schedule is missing trajectory={metadata['trajectory_index']} step={step}"
                )
            is_refresh = schedule_row["backbone_action_name"] == "REFRESH"
            is_probe_reuse = bool(schedule_row["probe_reuse"])
            if not is_refresh and not is_probe_reuse:
                continue

            prepared = adapter.prepare_inputs(observation)
            with torch.inference_mode():
                probe_output, action_input = adapter.get_first_block_gate_probe(
                    prepared["collated_inputs"]
                )
                if is_refresh:
                    ground_truth_backbone, anchor_layer_cache = (
                        adapter.continue_backbone_from_probe_with_layer_boundary_cache(
                            probe_output
                        )
                    )
                else:
                    ground_truth_backbone = adapter.continue_backbone_from_probe(probe_output)

            if is_refresh:
                _set_inference_seed(int(inference_seed))
                with torch.inference_mode():
                    _, anchor_action_cache = adapter.run_action_head(
                        backbone_output=ground_truth_backbone,
                        action_input=action_input,
                        reuse_strategy=None,
                    )
                continue
            if anchor_layer_cache is None or anchor_action_cache is None:
                raise RuntimeError("Probe reuse was selected before a refresh anchor was cached.")

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

            _set_inference_seed(int(inference_seed))
            with torch.inference_mode():
                full_cache_pred, _ = adapter.run_action_head(
                    backbone_cache=anchor_action_cache,
                    action_input=action_input,
                    reuse_strategy=None,
                )
            full_cache_action = adapter.decode_action(full_cache_pred, prepared["state"])
            baseline_rows.append(
                {
                    "trajectory_index": int(metadata["trajectory_index"]),
                    "env_name": str(metadata["env_name"]),
                    "step": step,
                    "representation_error": _tensor_error(
                        ground_truth_cache.backbone_features,
                        anchor_action_cache.backbone_features,
                    ),
                    "action_chunk_error": action_dict_error(
                        ground_truth_action,
                        full_cache_action,
                    )["overall"],
                    "executed_prefix_error": action_dict_error(
                        ground_truth_action,
                        full_cache_action,
                        prefix_steps=cfg.n_action_steps,
                    )["overall"],
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
                if refreshed_cache is not None or not cache_used:
                    raise RuntimeError(
                        f"Layer cache could not serve subset {recompute_layers}."
                    )
                _set_inference_seed(int(inference_seed))
                with torch.inference_mode():
                    candidate_pred, candidate_cache = adapter.run_action_head(
                        backbone_output=candidate_backbone,
                        action_input=action_input,
                        reuse_strategy=None,
                    )
                candidate_action = adapter.decode_action(
                    candidate_pred,
                    prepared["state"],
                )
                subset_key = "-".join(str(index) for index in recompute_layers)
                candidate_rows.append(
                    {
                        "trajectory_index": int(metadata["trajectory_index"]),
                        "env_name": str(metadata["env_name"]),
                        "step": step,
                        "inference_seed": int(inference_seed),
                        "feature_similarity": float(schedule_row["feature_similarity"]),
                        "subset_key": subset_key,
                        "recompute_layers": list(recompute_layers),
                        "total_recompute_layers": len(recompute_layers),
                        "representation_error": _tensor_error(
                            ground_truth_cache.backbone_features,
                            candidate_cache.backbone_features,
                        ),
                        "action_chunk_error": action_dict_error(
                            ground_truth_action,
                            candidate_action,
                        )["overall"],
                        "executed_prefix_error": action_dict_error(
                            ground_truth_action,
                            candidate_action,
                            prefix_steps=cfg.n_action_steps,
                        )["overall"],
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

    baseline_summary = _method_summary(baseline_rows)
    summaries: list[dict[str, Any]] = []
    for recompute_layers in subsets:
        subset_key = "-".join(str(index) for index in recompute_layers)
        rows = [row for row in candidate_rows if row["subset_key"] == subset_key]
        summary = _summarize_candidate(rows, baseline_summary)
        by_env = {
            env_name: _method_summary(
                [row for row in rows if row["env_name"] == env_name]
            )
            for env_name in sorted({row["env_name"] for row in rows})
        }
        summaries.append(
            {
                "subset_key": subset_key,
                "recompute_layers": list(recompute_layers),
                "total_recompute_layers": len(recompute_layers),
                "summary": summary,
                "by_env": by_env,
            }
        )

    ranking = sorted(
        summaries,
        key=lambda item: (
            float(item["summary"]["executed_prefix_error"]["mae"]),
            float(item["summary"]["representation_error"]["mae"]),
        ),
    )
    by_recompute_count = {}
    for total_layers in sorted({item["total_recompute_layers"] for item in ranking}):
        group = [item for item in ranking if item["total_recompute_layers"] == total_layers]
        by_recompute_count[str(total_layers)] = {
            "candidates": len(group),
            "best": group[0],
            "mean_executed_prefix_mae_across_candidates": float(
                np.mean(
                    [item["summary"]["executed_prefix_error"]["mae"] for item in group]
                )
            ),
        }

    summary = {
        "source": "paired_offline_probe_reuse_scan_on_saved_npz_trajectories",
        "num_language_layers": num_layers,
        "mandatory_probe_layer": 0,
        "candidate_subsets": len(subsets),
        "evaluated_probe_reuse_steps": len(baseline_rows),
        "schedule_counts": dict(
            Counter(row["backbone_action_name"] for row in schedule.values())
        ),
        "wall_seconds_excluding_model_load": float(time.perf_counter() - started_at),
        "full_cache_baseline": baseline_summary,
        "best_overall": ranking[0],
        "top_20": ranking[:20],
        "by_recompute_count": by_recompute_count,
        "all_candidates": ranking,
    }
    write_json(output_dir / "summary.json", summary)
    _write_ranking_csv(output_dir / "ranking.csv", ranking)
    print(
        f"Best subset={ranking[0]['subset_key']} "
        f"action_mae={ranking[0]['summary']['executed_prefix_error']['mae']:.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
