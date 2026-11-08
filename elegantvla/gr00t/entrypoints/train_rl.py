#!/usr/bin/env python3

"""Train an ElegantVLA scheduler for GR00T."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
import tyro
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import (
    MaskableDictRolloutBuffer,
    MaskableRolloutBuffer,
)
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.utils import obs_as_tensor

from elegantvla.gr00t.entrypoints._common import (
    finish_wandb_run,
    init_wandb_run,
    maybe_wandb_save,
    maybe_wandb_save_many,
    update_wandb_summary,
    write_json,
)
from elegantvla.gr00t.entrypoints._runtime import (
    build_temporal_skip_env,
    require_existing_model_path,
    resolve_env_names,
)


@dataclass
class TrainConfig:
    model_path: str
    save_dir: str = "outputs/gr00t/rl_train"
    resume_checkpoint: str | None = None
    resume_vecnormalize: str | None = None
    env_name: str = "simpler_env_widowx/widowx_spoon_on_towel"
    task_group: str = ""
    schedule_mode: str = "step"
    reward_mode: str = "stage1"
    stage2_efficiency_penalty_mode: str = "step"
    stage2_backbone_conservative_penalty: float = 0.05
    stage2_action_head_conservative_penalty: float = 0.025
    total_timesteps: int = 20000
    total_episodes: int = 0
    checkpoint_every: int = 1024
    seed: int = 0
    gr00t_device: str = "cuda"
    ppo_device: str = "cpu"
    tensorboard_log: str | None = None
    strict: bool = True
    compile_action_head: bool = True
    compile_mode: str = "max-autotune"

    max_episode_steps: int = 300
    n_action_steps: int = 1
    steps_per_render: int = 2

    learning_rate: float = 1e-4
    n_steps: int = 128
    episodes_per_update: int = 0
    batch_size: int = 128
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    verbose: int = 1

    use_wandb: bool = False
    wandb_project: str = "temporal-skip"
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    wandb_tags: str = ""
    wandb_dir: str | None = None
    wandb_job_type: str = "ppo-train"
    wandb_model_save_freq: int = 0
    wandb_gradient_save_freq: int = 0
    wandb_log: str = "all"
    wandb_sync_tensorboard: bool = True
    wandb_extra_paths: str = ""

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


def _resolve_schedule(cfg: TrainConfig) -> tuple[str, int, int]:
    schedule_mode = str(cfg.schedule_mode).strip().lower()
    if schedule_mode not in {"step", "episode"}:
        raise ValueError(f"Unsupported schedule_mode={cfg.schedule_mode!r}. Expected 'step' or 'episode'.")
    total_episodes = 0
    if schedule_mode == "episode":
        if int(cfg.episodes_per_update) <= 0:
            raise ValueError("schedule_mode='episode' requires episodes_per_update > 0.")
        total_episodes = max(int(cfg.total_episodes), 0)
        if total_episodes <= 0:
            raise ValueError("schedule_mode='episode' requires total_episodes > 0.")
        rollout_steps = max(int(cfg.episodes_per_update) * int(cfg.max_episode_steps), 1)
    else:
        rollout_steps = max(int(cfg.n_steps), 1)
    return schedule_mode, rollout_steps, total_episodes


def _resolve_tensorboard_log(cfg: TrainConfig, save_dir: Path) -> str | None:
    if cfg.tensorboard_log is not None:
        return cfg.tensorboard_log
    if cfg.use_wandb and cfg.wandb_sync_tensorboard:
        return str(save_dir / "tensorboard")
    return None


def _resolve_resume_vecnormalize(cfg: TrainConfig) -> str | None:
    if cfg.resume_vecnormalize:
        return cfg.resume_vecnormalize
    if not cfg.resume_checkpoint:
        return None
    checkpoint_path = Path(cfg.resume_checkpoint)
    candidates: list[Path] = []
    stem = checkpoint_path.stem if checkpoint_path.suffix == ".zip" else checkpoint_path.name
    candidates.append(checkpoint_path.with_name(f"{stem}_vecnormalize.pkl"))
    if stem.startswith("ppo_schedule_"):
        suffix = stem[len("ppo_schedule_") :]
        candidates.append(checkpoint_path.with_name(f"ppo_schedule_vecnormalize_{suffix}.pkl"))
    candidates.append(checkpoint_path.with_name("ppo_schedule_vecnormalize.pkl"))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _safe_train_batch_size(buffer_size: int, batch_size: int) -> int:
    """Avoid a final singleton minibatch that can poison advantage normalization."""
    buffer_size = int(buffer_size)
    batch_size = max(int(batch_size), 1)
    if buffer_size <= 1 or batch_size <= 2:
        return batch_size
    if buffer_size % batch_size == 1:
        return batch_size - 1
    return batch_size


def _make_wandb_callback(
    cfg: TrainConfig,
    *,
    save_dir: Path,
) -> Any | None:
    if not cfg.use_wandb:
        return None

    from wandb.integration.sb3 import WandbCallback

    wandb_log: str | None = None if cfg.wandb_log.strip().lower() == "none" else cfg.wandb_log
    return WandbCallback(
        verbose=1,
        model_save_path=str(save_dir / "wandb_checkpoints"),
        model_save_freq=max(int(cfg.wandb_model_save_freq), 0),
        gradient_save_freq=max(int(cfg.wandb_gradient_save_freq), 0),
        log=wandb_log,
    )


class EpisodeCheckpointCallback(BaseCallback):
    def __init__(
        self,
        *,
        save_freq_episodes: int,
        save_path: str,
        name_prefix: str = "ppo_schedule",
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.save_freq_episodes = max(int(save_freq_episodes), 1)
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_vecnormalize = bool(save_vecnormalize)
        self.completed_episodes = 0
        self.last_saved_episode = 0

    def _init_callback(self) -> None:
        Path(self.save_path).mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(self, episode_index: int) -> None:
        model_path = Path(self.save_path) / f"{self.name_prefix}_{int(episode_index)}_episodes.zip"
        self.model.save(str(model_path))
        if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
            vecnormalize_path = (
                Path(self.save_path) / f"{self.name_prefix}_vecnormalize_{int(episode_index)}_episodes.pkl"
            )
            self.model.get_vec_normalize_env().save(str(vecnormalize_path))
        if self.verbose >= 2:
            print(f"Saving episode checkpoint to {model_path}")

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is None:
            return True
        self.completed_episodes += int(np.asarray(dones, dtype=np.bool_).sum())
        while self.completed_episodes - self.last_saved_episode >= self.save_freq_episodes:
            self.last_saved_episode += self.save_freq_episodes
            self._save_checkpoint(self.last_saved_episode)
        return True


class _PartialRolloutBufferMixin:
    def __init__(self, buffer_size: int, *args: Any, **kwargs: Any) -> None:
        self._max_buffer_size = int(buffer_size)
        super().__init__(buffer_size, *args, **kwargs)

    def reset(self) -> None:
        self.buffer_size = int(self._max_buffer_size)
        super().reset()

    def finalize_for_training(self) -> None:
        active_size = int(self.pos)
        if active_size <= 0:
            raise ValueError("Cannot finalize an empty rollout buffer.")
        self._slice_active_storage(active_size)
        self.buffer_size = active_size
        self.pos = active_size
        self.full = True
        self.generator_ready = False

    def _slice_active_storage(self, active_size: int) -> None:
        raise NotImplementedError


class PartialMaskableRolloutBuffer(_PartialRolloutBufferMixin, MaskableRolloutBuffer):
    def _slice_active_storage(self, active_size: int) -> None:
        self.observations = self.observations[:active_size]
        self.actions = self.actions[:active_size]
        self.rewards = self.rewards[:active_size]
        self.returns = self.returns[:active_size]
        self.episode_starts = self.episode_starts[:active_size]
        self.values = self.values[:active_size]
        self.log_probs = self.log_probs[:active_size]
        self.advantages = self.advantages[:active_size]
        self.action_masks = self.action_masks[:active_size]


class PartialMaskableDictRolloutBuffer(_PartialRolloutBufferMixin, MaskableDictRolloutBuffer):
    def _slice_active_storage(self, active_size: int) -> None:
        self.observations = {
            key: value[:active_size]
            for key, value in self.observations.items()
        }
        self.actions = self.actions[:active_size]
        self.rewards = self.rewards[:active_size]
        self.returns = self.returns[:active_size]
        self.episode_starts = self.episode_starts[:active_size]
        self.values = self.values[:active_size]
        self.log_probs = self.log_probs[:active_size]
        self.advantages = self.advantages[:active_size]
        self.action_masks = self.action_masks[:active_size]


class EpisodeMaskablePPO(MaskablePPO):
    def __init__(self, *args: Any, episodes_per_update: int = 0, **kwargs: Any) -> None:
        self.episodes_per_update = max(int(episodes_per_update), 0)
        self.completed_episodes_total = 0
        self.last_rollout_completed_episodes = 0
        super().__init__(*args, **kwargs)

    def collect_rollouts(
        self,
        env: Any,
        callback: Any,
        rollout_buffer: Any,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        if int(self.episodes_per_update) <= 0:
            return super().collect_rollouts(
                env,
                callback,
                rollout_buffer,
                n_rollout_steps,
                use_masking=use_masking,
            )

        assert self._last_obs is not None, "No previous observation was provided"
        if env.num_envs != 1:
            raise ValueError("episodes_per_update only supports a single training env.")
        if not isinstance(
            rollout_buffer,
            (PartialMaskableRolloutBuffer, PartialMaskableDictRolloutBuffer),
        ):
            raise TypeError("episodes_per_update requires a partial rollout buffer.")

        self.policy.set_training_mode(False)
        action_masks = None
        completed_episodes = 0
        dones = np.zeros(env.num_envs, dtype=np.bool_)
        new_obs = self._last_obs
        rollout_buffer.reset()

        if use_masking and not is_masking_supported(env):
            raise ValueError(
                "Environment does not support action masking. Consider using ActionMasker wrapper"
            )

        callback.on_rollout_start()

        while completed_episodes < int(self.episodes_per_update):
            if rollout_buffer.pos >= rollout_buffer.buffer_size:
                raise RuntimeError(
                    "Episode rollout buffer overflowed before collecting the requested episodes. "
                    "Increase max_episode_steps or reduce episodes_per_update."
                )
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                if use_masking:
                    action_masks = get_action_masks(env)
                actions, values, log_probs = self.policy(obs_tensor, action_masks=action_masks)

            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                action_masks=action_masks,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones
            completed_episodes += int(np.asarray(dones, dtype=np.bool_).sum())

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.finalize_for_training()
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        self.last_rollout_completed_episodes = completed_episodes
        self.logger.record("train/completed_episodes", completed_episodes)
        self.logger.record("train/rollout_steps_this_update", rollout_buffer.buffer_size)
        callback.on_rollout_end()
        return True

    def learn(
        self,
        total_timesteps: int,
        callback: Any = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        total_episodes: int = 0,
    ) -> "EpisodeMaskablePPO":
        if total_episodes <= 0:
            return super().learn(
                total_timesteps=total_timesteps,
                callback=callback,
                log_interval=log_interval,
                tb_log_name=tb_log_name,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=progress_bar,
            )

        iteration = 0
        self.completed_episodes_total = 0
        self.last_rollout_completed_episodes = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        callback.on_training_start(locals(), globals())

        assert self.env is not None
        while self.completed_episodes_total < total_episodes:
            continue_training = self.collect_rollouts(
                self.env,
                callback,
                self.rollout_buffer,
                n_rollout_steps=self.n_steps,
            )
            if not continue_training:
                break

            iteration += 1
            self.completed_episodes_total += int(self.last_rollout_completed_episodes)
            self.logger.record("train/total_completed_episodes", self.completed_episodes_total)
            self._update_current_progress_remaining(
                min(self.completed_episodes_total, total_episodes),
                total_episodes,
            )
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self.dump_logs(iteration)

            original_batch_size = int(self.batch_size)
            adjusted_batch_size = _safe_train_batch_size(self.rollout_buffer.buffer_size, original_batch_size)
            if adjusted_batch_size != original_batch_size:
                if self.verbose >= 1:
                    print(
                        "adjusting_batch_size_to_avoid_singleton_minibatch "
                        f"buffer_size={self.rollout_buffer.buffer_size} "
                        f"batch_size={original_batch_size} -> {adjusted_batch_size}"
                    )
                self.logger.record("train/adjusted_batch_size", adjusted_batch_size)
                self.batch_size = adjusted_batch_size
            try:
                self.train()
            finally:
                self.batch_size = original_batch_size

        callback.on_training_end()
        return self


def main(cfg: TrainConfig) -> None:
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    require_existing_model_path(cfg.model_path)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    resolved_env_names = resolve_env_names(cfg)
    resolved_schedule_mode, resolved_rollout_steps, resolved_total_episodes = _resolve_schedule(cfg)
    is_episode_mode = resolved_schedule_mode == "episode"
    resolved_tensorboard_log = _resolve_tensorboard_log(cfg, save_dir)
    resolved_resume_vecnormalize = _resolve_resume_vecnormalize(cfg)
    wandb_run = init_wandb_run(
        enabled=cfg.use_wandb,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        group=cfg.wandb_group,
        name=cfg.wandb_run_name or save_dir.name,
        job_type=cfg.wandb_job_type,
        tags=cfg.wandb_tags,
        config={
            **asdict(cfg),
            "resolved_schedule_mode": resolved_schedule_mode,
            "resolved_tensorboard_log": resolved_tensorboard_log,
            "resolved_env_names": resolved_env_names,
            "resolved_total_episodes": resolved_total_episodes,
            "resolved_rollout_steps": resolved_rollout_steps,
            "resolved_resume_vecnormalize": resolved_resume_vecnormalize,
        },
        dir=cfg.wandb_dir,
        mode=cfg.wandb_mode,
        sync_tensorboard=bool(cfg.wandb_sync_tensorboard and resolved_tensorboard_log is not None),
    )
    vec_env: Any | None = None

    try:
        maybe_wandb_save_many(cfg.wandb_extra_paths, enabled=cfg.use_wandb)

        def make_env():
            env = build_temporal_skip_env(cfg)
            env = ActionMasker(env, lambda wrapped_env: wrapped_env.action_masks())
            return Monitor(env)

        vec_env = DummyVecEnv([make_env])
        if resolved_resume_vecnormalize:
            vec_env = VecNormalize.load(resolved_resume_vecnormalize, vec_env)
            vec_env.training = True
            vec_env.norm_reward = False
        else:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=False,
                clip_obs=10.0,
            )
        model_cls: Any = MaskablePPO
        model_extra_kwargs: dict[str, Any] = {}
        if is_episode_mode:
            if vec_env.num_envs != 1:
                raise ValueError("episodes_per_update currently only supports a single training env.")
            model_cls = EpisodeMaskablePPO
            model_extra_kwargs = {
                "episodes_per_update": int(cfg.episodes_per_update),
                "rollout_buffer_class": (
                    PartialMaskableDictRolloutBuffer
                    if isinstance(vec_env.observation_space, spaces.Dict)
                    else PartialMaskableRolloutBuffer
                ),
            }
        if cfg.resume_checkpoint:
            model = model_cls.load(
                cfg.resume_checkpoint,
                env=vec_env,
                device=cfg.ppo_device,
                n_steps=resolved_rollout_steps,
                **model_extra_kwargs,
            )
            model.verbose = cfg.verbose
            model.tensorboard_log = resolved_tensorboard_log
        else:
            model = model_cls(
                "MultiInputPolicy",
                vec_env,
                learning_rate=cfg.learning_rate,
                n_steps=resolved_rollout_steps,
                batch_size=cfg.batch_size,
                n_epochs=cfg.n_epochs,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                clip_range=cfg.clip_range,
                ent_coef=cfg.ent_coef,
                vf_coef=cfg.vf_coef,
                max_grad_norm=cfg.max_grad_norm,
                target_kl=cfg.target_kl,
                tensorboard_log=resolved_tensorboard_log,
                verbose=cfg.verbose,
                seed=cfg.seed,
                device=cfg.ppo_device,
                **model_extra_kwargs,
            )

        callbacks: list[Any] = []
        if is_episode_mode:
            callbacks.append(
                EpisodeCheckpointCallback(
                    save_freq_episodes=int(cfg.episodes_per_update),
                    save_path=str(save_dir),
                    name_prefix="ppo_schedule",
                    save_vecnormalize=True,
                    verbose=cfg.verbose,
                )
            )
        elif int(cfg.checkpoint_every) > 0:
            callbacks.append(
                CheckpointCallback(
                    save_freq=max(int(cfg.checkpoint_every), 1),
                    save_path=str(save_dir),
                    name_prefix="ppo_schedule",
                    save_vecnormalize=True,
                )
            )
        wandb_callback = _make_wandb_callback(cfg, save_dir=save_dir)
        if wandb_callback is not None:
            callbacks.append(wandb_callback)

        learn_callback: Any | None
        if not callbacks:
            learn_callback = None
        elif len(callbacks) == 1:
            learn_callback = callbacks[0]
        else:
            learn_callback = CallbackList(callbacks)

        learn_kwargs = {
            "total_timesteps": cfg.total_timesteps,
            "callback": learn_callback,
            "reset_num_timesteps": cfg.resume_checkpoint is None,
        }
        if is_episode_mode:
            learn_kwargs["total_episodes"] = resolved_total_episodes
        model.learn(**learn_kwargs)
        model_path = save_dir / "ppo_schedule"
        model.save(str(model_path))
        vecnormalize_path = save_dir / "ppo_schedule_vecnormalize.pkl"
        vec_env.save(str(vecnormalize_path))

        metadata: dict[str, Any] = {
            "config": asdict(cfg),
            "resolved_schedule_mode": resolved_schedule_mode,
            "resolved_env_names": resolved_env_names,
            "resolved_total_episodes": resolved_total_episodes,
            "resolved_rollout_steps": resolved_rollout_steps,
            "resolved_resume_vecnormalize": resolved_resume_vecnormalize,
            "saved_model": str(model_path.with_suffix(".zip")),
            "vecnormalize_path": str(vecnormalize_path),
        }
        if wandb_run is not None:
            metadata["wandb"] = {
                "run_id": wandb_run.id,
                "run_name": wandb_run.name,
                "run_url": wandb_run.url,
                "project": cfg.wandb_project,
            }
            summary_payload = {
                "final_model_path": str(model_path.with_suffix(".zip")),
                "vecnormalize_path": str(vecnormalize_path),
            }
            if resolved_tensorboard_log is not None:
                summary_payload["tensorboard_log"] = resolved_tensorboard_log
            update_wandb_summary(wandb_run, summary_payload)
        metadata_path = save_dir / "ppo_schedule_training.json"
        write_json(metadata_path, metadata)

        maybe_wandb_save(model_path.with_suffix(".zip"), enabled=cfg.use_wandb)
        maybe_wandb_save(vecnormalize_path, enabled=cfg.use_wandb)
        maybe_wandb_save(metadata_path, enabled=cfg.use_wandb)
        maybe_wandb_save_many(cfg.wandb_extra_paths, enabled=cfg.use_wandb)
        print({"saved_model": str(model_path.with_suffix(".zip"))})
    finally:
        if vec_env is not None:
            vec_env.close()
        finish_wandb_run(wandb_run)


def cli_main() -> None:
    main(tyro.cli(TrainConfig))


if __name__ == "__main__":
    cli_main()
