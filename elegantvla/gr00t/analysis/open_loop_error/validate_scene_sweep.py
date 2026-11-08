#!/usr/bin/env python3

"""Validate that the controller-free SimplerEnv scene sweep changes every frame."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import tyro

from elegantvla.gr00t.analysis.open_loop_error.compare_backbone_cache import (
    CacheComparisonConfig,
    _find_raw_simpler_env,
    _scene_sweep_observation,
)
from elegantvla.gr00t.analysis.open_loop_error.evaluate import _create_sim_env


@dataclass
class Config:
    env_name: str = "simpler_env_widowx/widowx_spoon_on_towel"
    output_path: str = "/tmp/elegantvla_gr00t_scene_sweep.npz"
    seed: int = 0
    steps: int = 8
    qpos_sweep_amplitude: float = 0.0
    object_sweep_amplitude: float = 0.0
    camera_sweep_amplitude: float = 0.05
    render_flush_frames: int = 1
    render_pipeline_warmup_steps: int = 4


def main(cfg: Config) -> None:
    env_cfg = CacheComparisonConfig(
        env_names=[cfg.env_name],
        max_trajectory_steps=cfg.steps,
    )
    sim_env = _create_sim_env(env_cfg, cfg.env_name)
    try:
        observation, _ = sim_env.reset(seed=cfg.seed)
        _, raw_env = _find_raw_simpler_env(sim_env)
        baseline_qpos = np.asarray(raw_env.agent.robot.get_qpos(), dtype=np.float64).copy()
        sweep_actor = raw_env.episode_source_obj
        baseline_actor_position = np.asarray(sweep_actor.pose.p, dtype=np.float64).copy()
        baseline_actor_quaternion = np.asarray(sweep_actor.pose.q, dtype=np.float64).copy()
        sweep_camera = raw_env._cameras["3rd_view_camera"]
        baseline_camera_position = np.asarray(
            sweep_camera.camera_cfg.pose.p, dtype=np.float64
        ).copy()
        baseline_camera_quaternion = np.asarray(
            sweep_camera.camera_cfg.pose.q, dtype=np.float64
        ).copy()
        for warmup_step in range(-cfg.render_pipeline_warmup_steps, 1):
            observation = _scene_sweep_observation(
                sim_env,
                baseline_qpos=baseline_qpos,
                sweep_actor=sweep_actor,
                baseline_actor_position=baseline_actor_position,
                baseline_actor_quaternion=baseline_actor_quaternion,
                sweep_camera=sweep_camera,
                baseline_camera_position=baseline_camera_position,
                baseline_camera_quaternion=baseline_camera_quaternion,
                step=warmup_step,
                total_steps=cfg.steps,
                qpos_amplitude=cfg.qpos_sweep_amplitude,
                object_amplitude=cfg.object_sweep_amplitude,
                camera_amplitude=cfg.camera_sweep_amplitude,
                render_flush_frames=cfg.render_flush_frames,
            )
        frames = [np.asarray(observation["video.image_0"])]
        for step in range(1, cfg.steps):
            observation = _scene_sweep_observation(
                sim_env,
                baseline_qpos=baseline_qpos,
                sweep_actor=sweep_actor,
                baseline_actor_position=baseline_actor_position,
                baseline_actor_quaternion=baseline_actor_quaternion,
                sweep_camera=sweep_camera,
                baseline_camera_position=baseline_camera_position,
                baseline_camera_quaternion=baseline_camera_quaternion,
                step=step,
                total_steps=cfg.steps,
                qpos_amplitude=cfg.qpos_sweep_amplitude,
                object_amplitude=cfg.object_sweep_amplitude,
                camera_amplitude=cfg.camera_sweep_amplitude,
                render_flush_frames=cfg.render_flush_frames,
            )
            frames.append(np.asarray(observation["video.image_0"]))
        frame_array = np.concatenate(frames, axis=0)
        maes = [
            float(
                np.mean(
                    np.abs(
                        frame_array[index].astype(np.float32)
                        - frame_array[index - 1].astype(np.float32)
                    )
                )
            )
            for index in range(1, len(frame_array))
        ]
        hashes = [
            hashlib.sha1(frame.tobytes()).hexdigest()[:12] for frame in frame_array
        ]
        output_path = Path(cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, frames=frame_array, adjacent_frame_mae=maes)
        print(f"hashes={hashes}")
        print(f"adjacent_frame_mae={maes}")
        print(f"unique_frames={len(set(hashes))}/{len(hashes)}")
    finally:
        sim_env.close()


if __name__ == "__main__":
    main(tyro.cli(Config))
