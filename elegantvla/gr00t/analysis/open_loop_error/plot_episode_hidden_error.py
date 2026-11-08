#!/usr/bin/env python3

"""Plot per-step hidden-state error over saved episodes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if np.std(left_array) == 0.0 or np.std(right_array) == 0.0:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _series_summary(rows: list[dict], key: str) -> dict:
    non_refresh = [row for row in rows if not row["is_refresh"]]
    values = [float(row[key]) for row in non_refresh]
    minimum = min(non_refresh, key=lambda row: float(row[key]))
    maximum = max(non_refresh, key=lambda row: float(row[key]))
    return {
        "steps": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(minimum[key]),
        "min_step": int(minimum["step"]),
        "max": float(maximum[key]),
        "max_step": int(maximum["step"]),
        "range": float(maximum[key] - minimum[key]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--schedule-path", type=Path, required=True)
    args = parser.parse_args()

    candidate_rows = _read_jsonl(args.result_dir / "candidate_rows.jsonl")
    full_cache_rows = _read_jsonl(args.result_dir / "full_cache_rows.jsonl")
    schedule_rows = _read_jsonl(args.schedule_path)
    candidate_by_key = {
        (int(row["trajectory_index"]), int(row["step"])): row for row in candidate_rows
    }
    full_cache_by_key = {
        (int(row["trajectory_index"]), int(row["step"])): row for row in full_cache_rows
    }

    timeline: list[dict] = []
    anchor_steps: dict[int, int] = {}
    for schedule_row in schedule_rows:
        trajectory_index = int(schedule_row["trajectory_index"])
        step = int(schedule_row["step"])
        key = (trajectory_index, step)
        is_refresh = schedule_row["backbone_action_name"] == "REFRESH"
        if is_refresh:
            anchor_steps[trajectory_index] = step
            candidate_mae = 0.0
            full_cache_mae = 0.0
            image_mae = 0.0
        else:
            candidate = candidate_by_key[key]
            full_cache = full_cache_by_key[key]
            candidate_mae = float(
                candidate["cross_attention_error"]["all_cross_attention_tokens"]["mae"]
            )
            full_cache_mae = float(
                full_cache["cross_attention_error"]["all_cross_attention_tokens"]["mae"]
            )
            image_mae = float(
                candidate["cross_attention_error"]["image_tokens"]["mae"]
            )
        anchor_step = anchor_steps[trajectory_index]
        timeline.append(
            {
                "trajectory_index": trajectory_index,
                "env_name": str(schedule_row["env_name"]),
                "step": step,
                "action": str(schedule_row["backbone_action_name"]),
                "is_refresh": is_refresh,
                "probe_reuse": bool(schedule_row["probe_reuse"]),
                "anchor_step": anchor_step,
                "cache_age": step - anchor_step,
                "previous_frame_image_mae": float(
                    schedule_row.get("previous_frame_image_mae", 0.0)
                ),
                "candidate_mae": candidate_mae,
                "candidate_image_mae": image_mae,
                "full_cache_mae": full_cache_mae,
                "improvement_vs_full_cache": full_cache_mae - candidate_mae,
            }
        )

    csv_path = args.result_dir / "episode_error_timeline.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timeline[0]))
        writer.writeheader()
        writer.writerows(timeline)

    trajectory_indices = sorted({row["trajectory_index"] for row in timeline})
    per_episode = {}
    all_non_refresh = [row for row in timeline if not row["is_refresh"]]
    for trajectory_index in trajectory_indices:
        rows = [row for row in timeline if row["trajectory_index"] == trajectory_index]
        non_refresh = [row for row in rows if not row["is_refresh"]]
        per_episode[str(trajectory_index)] = {
            "env_name": rows[0]["env_name"],
            "candidate": _series_summary(rows, "candidate_mae"),
            "full_cache": _series_summary(rows, "full_cache_mae"),
            "pearson_error_vs_cache_age": _pearson(
                [float(row["candidate_mae"]) for row in non_refresh],
                [float(row["cache_age"]) for row in non_refresh],
            ),
            "pearson_error_vs_previous_frame_image_mae": _pearson(
                [float(row["candidate_mae"]) for row in non_refresh],
                [float(row["previous_frame_image_mae"]) for row in non_refresh],
            ),
            "candidate_better_steps": sum(
                float(row["candidate_mae"]) < float(row["full_cache_mae"])
                for row in non_refresh
            ),
        }

    summary = {
        "candidate": _series_summary(timeline, "candidate_mae"),
        "full_cache": _series_summary(timeline, "full_cache_mae"),
        "pearson_error_vs_cache_age": _pearson(
            [float(row["candidate_mae"]) for row in all_non_refresh],
            [float(row["cache_age"]) for row in all_non_refresh],
        ),
        "pearson_error_vs_previous_frame_image_mae": _pearson(
            [float(row["candidate_mae"]) for row in all_non_refresh],
            [float(row["previous_frame_image_mae"]) for row in all_non_refresh],
        ),
        "candidate_better_steps": sum(
            float(row["candidate_mae"]) < float(row["full_cache_mae"])
            for row in all_non_refresh
        ),
        "non_refresh_steps": len(all_non_refresh),
        "per_episode": per_episode,
    }
    (args.result_dir / "episode_error_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    fig, axes = plt.subplots(len(trajectory_indices), 1, figsize=(11, 8.5), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, trajectory_index in zip(axes, trajectory_indices, strict=True):
        rows = [row for row in timeline if row["trajectory_index"] == trajectory_index]
        steps = [row["step"] for row in rows]
        axis.plot(
            steps,
            [row["full_cache_mae"] for row in rows],
            color="#7a7a7a",
            linestyle="--",
            marker="o",
            markersize=3,
            linewidth=1.4,
            label="Full cache",
        )
        axis.plot(
            steps,
            [row["candidate_mae"] for row in rows],
            color="#147d64",
            marker="o",
            markersize=4,
            linewidth=1.8,
            label="Recompute {0,12,13,14,15}",
        )
        refresh_steps = [row["step"] for row in rows if row["is_refresh"]]
        for refresh_step in refresh_steps:
            axis.axvline(refresh_step, color="#d2ddd8", linewidth=0.8, zorder=0)
        axis.set_title(rows[0]["env_name"].split("/")[-1], fontsize=10)
        axis.set_ylabel("Hidden MAE")
        axis.grid(axis="y", alpha=0.22)
    axes[0].legend(loc="upper right", frameon=False, ncol=2)
    axes[-1].set_xlabel("Episode step (vertical lines are refreshes)")
    fig.tight_layout()
    fig.savefig(args.result_dir / "episode_error_dynamics.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
