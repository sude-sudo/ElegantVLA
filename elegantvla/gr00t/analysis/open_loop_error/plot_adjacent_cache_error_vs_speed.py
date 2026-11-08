#!/usr/bin/env python3

"""Plot previous-frame full-cache representation error against robot motion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _state_series(payload: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    values = np.asarray(payload[f"observation.state.{key}"], dtype=np.float64)
    return values.reshape(values.shape[0], -1)[:, 0]


def _smoothed_delta(values: np.ndarray, window: int, *, vector: bool) -> np.ndarray:
    speed = np.zeros(values.shape[0], dtype=np.float64)
    for step in range(window, values.shape[0]):
        delta = values[step] - values[step - window]
        speed[step] = np.linalg.norm(delta) / float(window) if vector else abs(delta) / float(window)
    return speed


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2 or np.std(x[valid]) == 0.0 or np.std(y[valid]) == 0.0:
        return None
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def _metric(row: dict[str, Any], group: str, name: str) -> float:
    return float(row[group][name])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoothing-window", type=int, default=4)
    parser.add_argument(
        "--primary-speed",
        choices=("scheduler_motion_signal", "eef_translation_speed"),
        default="scheduler_motion_signal",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "error_vs_speed"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_step_rows = _read_jsonl(input_dir / "steps.jsonl")
    trajectory_paths = sorted((input_dir / "trajectories").glob("trajectory_*.npz"))
    if not trajectory_paths:
        raise FileNotFoundError(f"No trajectories found under {input_dir / 'trajectories'}")

    timeline_rows: list[dict[str, Any]] = []
    trajectory_summaries: list[dict[str, Any]] = []
    for trajectory_path in trajectory_paths:
        with np.load(trajectory_path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            trajectory_index = int(metadata["trajectory_index"])
            rows = sorted(
                (
                    row
                    for row in all_step_rows
                    if int(row["trajectory_index"]) == trajectory_index
                ),
                key=lambda row: int(row["step"]),
            )
            if not rows:
                continue

            position = np.stack([_state_series(payload, key) for key in ("x", "y", "z")], axis=1)
            orientation = np.stack(
                [_state_series(payload, key) for key in ("roll", "pitch", "yaw")], axis=1
            )
            gripper = _state_series(payload, "gripper")
            images = np.asarray(payload["observation.video.image_0"], dtype=np.float32)

        count = min(len(rows), len(position))
        rows = rows[:count]
        position = position[:count]
        orientation = orientation[:count]
        gripper = gripper[:count]
        images = images[:count]
        window = max(int(args.smoothing_window), 1)
        translation_speed = _smoothed_delta(position, window, vector=True)
        rotation_speed = _smoothed_delta(orientation, window, vector=True)
        gripper_speed = _smoothed_delta(gripper, window, vector=False)
        motion_signal = np.maximum.reduce([translation_speed, rotation_speed, gripper_speed])
        if args.primary_speed == "eef_translation_speed":
            primary_speed = translation_speed
            primary_speed_label = f"EEF translation speed ({window}-step average)"
            primary_speed_axis_label = "EEF translation / decision step"
        else:
            primary_speed = motion_signal
            primary_speed_label = f"Motion signal ({window}-step smoothed)"
            primary_speed_axis_label = "Scheduler motion signal / decision step"

        hidden_mae = np.asarray(
            [_metric(row, "full_cache_representation_error", "mae") for row in rows],
            dtype=np.float64,
        )
        hidden_rmse = np.asarray(
            [_metric(row, "full_cache_representation_error", "rmse") for row in rows],
            dtype=np.float64,
        )
        hidden_cosine = np.asarray(
            [_metric(row, "full_cache_representation_error", "cosine_similarity") for row in rows],
            dtype=np.float64,
        )
        image_mae = np.zeros(count, dtype=np.float64)
        if count > 1:
            image_mae[1:] = np.mean(
                np.abs(images[1:].astype(np.float64) - images[:-1].astype(np.float64)),
                axis=tuple(range(1, images.ndim)),
            )
        for index, row in enumerate(rows):
            if "previous_frame_image_mae" in row:
                image_mae[index] = float(row["previous_frame_image_mae"])
        probe_cka = np.asarray(
            [
                np.nan
                if row.get("previous_frame_probe_cka") is None
                else float(row["previous_frame_probe_cka"])
                for row in rows
            ],
            dtype=np.float64,
        )
        probe_distance = 1.0 - probe_cka
        evaluated = np.asarray([not bool(row["cache_warmup"]) for row in rows], dtype=bool)
        motion_ready = np.arange(count) >= window
        correlation_mask = evaluated & motion_ready

        for index, row in enumerate(rows):
            timeline_rows.append(
                {
                    "trajectory_index": trajectory_index,
                    "env_name": str(row["env_name"]),
                    "seed": int(row["seed"]),
                    "step": int(row["step"]),
                    "cache_warmup": bool(row["cache_warmup"]),
                    "speed_window_ready": bool(motion_ready[index]),
                    "previous_frame_full_cache_hidden_mae": float(hidden_mae[index]),
                    "previous_frame_full_cache_hidden_rmse": float(hidden_rmse[index]),
                    "previous_frame_full_cache_hidden_cosine_similarity": float(
                        hidden_cosine[index]
                    ),
                    "previous_frame_image_mae": float(image_mae[index]),
                    "previous_frame_probe_cka": float(probe_cka[index]),
                    "previous_frame_probe_cka_distance": float(probe_distance[index]),
                    "eef_translation_speed": float(translation_speed[index]),
                    "eef_rotation_speed": float(rotation_speed[index]),
                    "gripper_speed": float(gripper_speed[index]),
                    "scheduler_motion_signal": float(motion_signal[index]),
                }
            )

        summary = {
            "trajectory_index": trajectory_index,
            "env_name": str(rows[0]["env_name"]),
            "seed": int(rows[0]["seed"]),
            "steps": count,
            "evaluated_adjacent_transitions": int(evaluated.sum()),
            "correlation_steps_after_speed_warmup": int(correlation_mask.sum()),
            "hidden_mae_mean": float(hidden_mae[evaluated].mean()),
            "hidden_mae_max": float(hidden_mae[evaluated].max()),
            "correlations_with_hidden_mae": {
                "scheduler_motion_signal": _pearson(
                    hidden_mae[correlation_mask], motion_signal[correlation_mask]
                ),
                "eef_translation_speed": _pearson(
                    hidden_mae[correlation_mask], translation_speed[correlation_mask]
                ),
                "eef_rotation_speed": _pearson(
                    hidden_mae[correlation_mask], rotation_speed[correlation_mask]
                ),
                "gripper_speed": _pearson(
                    hidden_mae[correlation_mask], gripper_speed[correlation_mask]
                ),
                "previous_frame_image_mae": _pearson(hidden_mae[evaluated], image_mae[evaluated]),
                "previous_frame_probe_cka": _pearson(
                    hidden_mae[evaluated], probe_cka[evaluated]
                ),
                "previous_frame_probe_cka_distance": _pearson(
                    hidden_mae[evaluated], probe_distance[evaluated]
                ),
            },
            "probe_cka_mean": float(np.nanmean(probe_cka[evaluated])),
            "probe_cka_min": float(np.nanmin(probe_cka[evaluated])),
        }
        trajectory_summaries.append(summary)

        steps = np.arange(count)
        fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.4), sharex=True)
        error_axis = axes[0]
        speed_axis = error_axis.twinx()
        cka_axis = error_axis.twinx()
        cka_axis.spines["right"].set_position(("axes", 1.11))
        error_line = error_axis.plot(
            steps,
            hidden_mae,
            color="#c43c39",
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label="Previous-frame full-cache hidden MAE",
        )[0]
        speed_line = speed_axis.plot(
            steps,
            primary_speed,
            color="#2467a5",
            linewidth=1.8,
            label=primary_speed_label,
        )[0]
        cka_line = cka_axis.plot(
            steps,
            probe_cka,
            color="#6a4c93",
            linewidth=1.7,
            linestyle="--",
            label="Adjacent probe CKA",
        )[0]
        error_axis.set_ylabel("DiT memory hidden-state MAE", color="#c43c39")
        speed_axis.set_ylabel(primary_speed_axis_label, color="#2467a5")
        cka_axis.set_ylabel("Probe CKA vs. previous frame", color="#6a4c93")
        finite_cka = probe_cka[np.isfinite(probe_cka)]
        if finite_cka.size:
            cka_lower = max(0.0, float(finite_cka.min()) - 0.01)
            cka_upper = min(1.0, float(finite_cka.max()) + 0.005)
            if cka_upper - cka_lower < 0.02:
                cka_lower = max(0.0, cka_upper - 0.02)
            cka_axis.set_ylim(cka_lower, cka_upper)
        error_axis.tick_params(axis="y", colors="#c43c39")
        speed_axis.tick_params(axis="y", colors="#2467a5")
        cka_axis.tick_params(axis="y", colors="#6a4c93")
        error_axis.grid(alpha=0.22)
        error_axis.legend(
            [error_line, speed_line, cka_line],
            [error_line.get_label(), speed_line.get_label(), cka_line.get_label()],
            loc="upper right",
        )
        error_axis.set_title(
            f"Adjacent full-cache error vs. robot motion\n{str(rows[0]['env_name']).split('/')[-1]}, seed {int(rows[0]['seed'])}"
        )

        eef_axis = axes[1]
        gripper_axis = eef_axis.twinx()
        translation_line = eef_axis.plot(
            steps, translation_speed, label="EEF translation", color="#2a9d8f"
        )[0]
        rotation_line = eef_axis.plot(
            steps, rotation_speed, label="EEF rotation", color="#6a4c93"
        )[0]
        gripper_line = gripper_axis.plot(
            steps, gripper_speed, label="Gripper", color="#e09f3e"
        )[0]
        eef_axis.set_xlabel("Closed-loop decision step")
        eef_axis.set_ylabel("EEF motion / decision step")
        gripper_axis.set_ylabel("Gripper motion / decision step", color="#e09f3e")
        gripper_axis.tick_params(axis="y", colors="#e09f3e")
        eef_axis.grid(alpha=0.22)
        eef_axis.legend(
            [translation_line, rotation_line, gripper_line],
            [
                translation_line.get_label(),
                rotation_line.get_label(),
                gripper_line.get_label(),
            ],
            ncol=3,
            loc="upper right",
        )
        fig.subplots_adjust(right=0.82, top=0.9, hspace=0.08)
        fig.savefig(output_dir / f"trajectory_{trajectory_index:04d}_error_vs_speed.png", dpi=200)
        plt.close(fig)

    if not timeline_rows:
        raise RuntimeError("No matching trajectory rows were found.")
    fieldnames = list(timeline_rows[0])
    with (output_dir / "error_vs_speed_timeline.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timeline_rows)

    evaluated_rows = [row for row in timeline_rows if not bool(row["cache_warmup"])]
    aggregate_hidden_mae = np.asarray(
        [row["previous_frame_full_cache_hidden_mae"] for row in evaluated_rows],
        dtype=np.float64,
    )

    def aggregate_values(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in evaluated_rows], dtype=np.float64)

    summary = {
        "definition": {
            "ground_truth": "current-frame full VLM compute, after vlln, as DiT cross-attention memory",
            "candidate": "immediately previous frame's full VLM cache reused without correction",
            "error": "elementwise hidden-state MAE",
            "speed": (
                "max of EEF translation, EEF Euler rotation, and gripper speeds; each is "
                f"a {max(int(args.smoothing_window), 1)}-step displacement average matching SkipController"
            ),
            "primary_speed_curve": args.primary_speed,
        },
        "aggregate": {
            "evaluated_adjacent_transitions": len(evaluated_rows),
            "hidden_mae_mean": float(aggregate_hidden_mae.mean()),
            "hidden_mae_max": float(aggregate_hidden_mae.max()),
            "probe_cka_mean": float(aggregate_values("previous_frame_probe_cka").mean()),
            "correlations_with_hidden_mae": {
                "eef_translation_speed": _pearson(
                    aggregate_hidden_mae, aggregate_values("eef_translation_speed")
                ),
                "eef_rotation_speed": _pearson(
                    aggregate_hidden_mae, aggregate_values("eef_rotation_speed")
                ),
                "gripper_speed": _pearson(
                    aggregate_hidden_mae, aggregate_values("gripper_speed")
                ),
                "previous_frame_image_mae": _pearson(
                    aggregate_hidden_mae, aggregate_values("previous_frame_image_mae")
                ),
                "previous_frame_probe_cka": _pearson(
                    aggregate_hidden_mae, aggregate_values("previous_frame_probe_cka")
                ),
                "previous_frame_probe_cka_distance": _pearson(
                    aggregate_hidden_mae,
                    aggregate_values("previous_frame_probe_cka_distance"),
                ),
            },
        },
        "trajectories": trajectory_summaries,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
