"""Chunked temporal-skip env with env-owned scheduler orchestration."""

from __future__ import annotations

from typing import Any
import os

import gymnasium as gym
import numpy as np

from elegantvla.gr00t.core.rule_policy import RuleSchedulePolicy
from elegantvla.gr00t.core.types import (
    ActionHeadAction,
    BackboneAction,
    ScheduleAction,
    ScheduleObservation,
    ScheduleState,
)


RL_BACKBONE_MISMATCH_PENALTY = float(
    os.environ.get("TEMPORAL_SKIP_RL_BACKBONE_MISMATCH_PENALTY", "1.0")
)
RL_ACTION_HEAD_MISMATCH_PENALTY = float(
    os.environ.get("TEMPORAL_SKIP_RL_ACTION_HEAD_MISMATCH_PENALTY", "0.5")
)
RL_SUCCESS_BONUS = float(os.environ.get("TEMPORAL_SKIP_RL_SUCCESS_BONUS", "8.0"))
RL_BACKBONE_AGGRESSIVE_BONUS = float(
    os.environ.get("TEMPORAL_SKIP_RL_BACKBONE_AGGRESSIVE_BONUS", "0.0")
)
RL_ACTION_HEAD_AGGRESSIVE_BONUS = float(
    os.environ.get("TEMPORAL_SKIP_RL_ACTION_HEAD_AGGRESSIVE_BONUS", "0.0")
)
RL_NO_IMAGE_ENTRY_COST = float(os.environ.get("TEMPORAL_SKIP_RL_NO_IMAGE_ENTRY_COST", "0.0"))


class TemporalSkipEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Chunk the policy execution into scheduler decisions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        sim_env: gym.Env[Any, Any],
        wrapper: Any,
        reward_policy: Any | None = None,
        reference_schedule_policy: Any | None = None,
        reward_mode: str = "stage1",
        stage2_efficiency_penalty_mode: str = "step",
        stage2_backbone_conservative_penalty: float = 0.05,
        stage2_action_head_conservative_penalty: float = 0.025,
        max_episode_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.sim_env = sim_env
        self.wrapper = wrapper
        resolved_reward_policy = reference_schedule_policy or reward_policy or RuleSchedulePolicy()
        self.reward_policy = resolved_reward_policy
        self.reference_schedule_policy = resolved_reward_policy
        self.modality_configs = self.wrapper.policy.modality_configs
        self.reward_mode = str(reward_mode).strip().lower()
        if self.reward_mode not in {"stage1", "stage2"}:
            raise ValueError(
                f"Unsupported reward_mode={reward_mode!r}. Expected 'stage1' or 'stage2'."
            )
        self.stage2_efficiency_penalty_mode = str(stage2_efficiency_penalty_mode).strip().lower()
        if self.stage2_efficiency_penalty_mode not in {"step", "terminal"}:
            raise ValueError(
                "Unsupported stage2_efficiency_penalty_mode="
                f"{stage2_efficiency_penalty_mode!r}. Expected 'step' or 'terminal'."
            )
        self.stage2_backbone_conservative_penalty = float(stage2_backbone_conservative_penalty)
        self.stage2_action_head_conservative_penalty = float(stage2_action_head_conservative_penalty)

        self._sim_obs: dict[str, Any] | None = None
        self._schedule_state: ScheduleState | None = None
        self._episode_success = False
        self._episode_step = 0
        self._episode_stage2_conservative_penalty = 0.0
        self._last_stage2_step_conservative_penalty = 0.0
        self.max_episode_steps = self._resolve_max_episode_steps(
            sim_env,
            max_episode_steps=max_episode_steps,
        )

        self.action_space = gym.spaces.MultiDiscrete([len(BackboneAction), len(ActionHeadAction)])
        self.observation_space = ScheduleObservation.space()

    @staticmethod
    def _resolve_max_episode_steps(
        sim_env: gym.Env[Any, Any],
        *,
        max_episode_steps: int | None,
    ) -> int:
        if max_episode_steps is not None:
            return max(int(max_episode_steps), 1)
        for attr in ("max_episode_steps", "_max_episode_steps"):
            value = getattr(sim_env, attr, None)
            if value is not None:
                return max(int(value), 1)
        spec = getattr(sim_env, "spec", None)
        value = None if spec is None else getattr(spec, "max_episode_steps", None)
        return max(int(value), 1) if value is not None else 1

    def _normalized_step_progress(self) -> float:
        return float(
            np.clip(
                float(self._episode_step) / float(max(int(self.max_episode_steps), 1)),
                0.0,
                1.0,
            )
        )

    def _attach_step_progress_to_schedule_state(self) -> dict[str, np.ndarray]:
        if self._schedule_state is None:
            raise RuntimeError("Scheduler state is missing. Call reset() before step().")
        schedule_obs = dict(self._schedule_state.schedule_obs)
        schedule_obs["step_progress"] = np.asarray(
            [self._normalized_step_progress()],
            dtype=np.float32,
        )
        self._schedule_state.schedule_obs = schedule_obs
        return schedule_obs

    def action_masks(self) -> np.ndarray:
        if self._schedule_state is None:
            backbone_mask = np.ones(len(BackboneAction), dtype=bool)
        else:
            backbone_mask = self._schedule_state.backbone_action_mask.astype(bool, copy=False)
        return np.concatenate(
            [
                backbone_mask,
                np.ones(len(ActionHeadAction), dtype=bool),
            ],
            axis=0,
        )

    def _compute_schedule_reward(
        self,
        *,
        schedule_state: ScheduleState,
        backbone_action: BackboneAction,
        resolved_backbone_action: BackboneAction,
        action_head_action: ActionHeadAction,
        success: bool,
        episode_done: bool,
    ) -> float:
        if self.reward_mode == "stage2":
            return self._compute_stage2_schedule_reward(
                schedule_state=schedule_state,
                backbone_action=resolved_backbone_action,
                action_head_action=action_head_action,
                success=success,
                episode_done=episode_done,
            )
        return self._compute_stage1_schedule_reward(
            schedule_state=schedule_state,
            backbone_action=backbone_action,
            action_head_action=action_head_action,
            success=success,
        )

    def _compute_stage1_schedule_reward(
        self,
        *,
        schedule_state: ScheduleState,
        backbone_action: BackboneAction,
        action_head_action: ActionHeadAction,
        success: bool,
    ) -> float:
        target_schedule_action = self.reference_schedule_policy.select_action(
            schedule_state.schedule_obs,
            backbone_action_mask=schedule_state.backbone_action_mask,
        )
        target_backbone_action = target_schedule_action.backbone_action
        target_action_head_action = target_schedule_action.action_head_action
        reward = 0.0
        reward += RL_BACKBONE_AGGRESSIVE_BONUS * float(int(backbone_action))
        reward += RL_ACTION_HEAD_AGGRESSIVE_BONUS * float(int(action_head_action))
        if backbone_action.budget > 0:
            reward -= RL_NO_IMAGE_ENTRY_COST * float(backbone_action.budget)

        valid_backbone_actions = int(np.sum(schedule_state.backbone_action_mask > 0.5))
        if valid_backbone_actions > 1 and backbone_action != target_backbone_action:
            reward -= RL_BACKBONE_MISMATCH_PENALTY
        if action_head_action != target_action_head_action:
            reward -= RL_ACTION_HEAD_MISMATCH_PENALTY
        if success:
            reward += RL_SUCCESS_BONUS
        return float(reward)

    @staticmethod
    def _normalized_gap(selected_index: int, valid_indices: np.ndarray) -> float:
        if valid_indices.size == 0:
            return 0.0
        min_valid = int(np.min(valid_indices))
        max_valid = int(np.max(valid_indices))
        if max_valid <= min_valid:
            return 0.0
        clipped_selected = min(max(int(selected_index), min_valid), max_valid)
        gap = max_valid - clipped_selected
        return float(gap / float(max_valid - min_valid))

    def _compute_stage2_step_conservative_penalty(
        self,
        *,
        schedule_state: ScheduleState,
        backbone_action: BackboneAction,
        action_head_action: ActionHeadAction,
    ) -> float:
        backbone_mask = (
            np.ones(len(BackboneAction), dtype=bool)
            if schedule_state.backbone_action_mask is None
            else np.asarray(schedule_state.backbone_action_mask, dtype=np.float32).reshape(-1) > 0.5
        )
        valid_backbone_indices = np.flatnonzero(backbone_mask)
        backbone_penalty = self.stage2_backbone_conservative_penalty * self._normalized_gap(
            int(backbone_action),
            valid_backbone_indices,
        )
        valid_action_head_indices = np.arange(len(ActionHeadAction), dtype=np.int64)
        action_head_penalty = self.stage2_action_head_conservative_penalty * self._normalized_gap(
            int(action_head_action),
            valid_action_head_indices,
        )
        return float(backbone_penalty + action_head_penalty)

    def _compute_stage2_schedule_reward(
        self,
        *,
        schedule_state: ScheduleState,
        backbone_action: BackboneAction,
        action_head_action: ActionHeadAction,
        success: bool,
        episode_done: bool,
    ) -> float:
        step_penalty = self._compute_stage2_step_conservative_penalty(
            schedule_state=schedule_state,
            backbone_action=backbone_action,
            action_head_action=action_head_action,
        )
        self._last_stage2_step_conservative_penalty = float(step_penalty)
        self._episode_stage2_conservative_penalty += float(step_penalty)
        if self.stage2_efficiency_penalty_mode == "step":
            reward = -float(step_penalty)
            if success:
                reward += RL_SUCCESS_BONUS
            return float(reward)

        if not episode_done:
            return 0.0
        reward = -float(self._episode_stage2_conservative_penalty)
        if success:
            reward += RL_SUCCESS_BONUS
        return float(reward)

    def _build_policy_observation(self, sim_obs: dict[str, Any]) -> dict[str, Any]:
        policy_obs = {"video": {}, "state": {}, "language": {}}
        for modality in ("video", "state", "language"):
            for key in self.modality_configs[modality].modality_keys:
                if modality == "language":
                    parsed_key = key
                    if key == "task" and "annotation.human.coarse_action" in sim_obs:
                        parsed_key = "annotation.human.coarse_action"
                    values = sim_obs[parsed_key]
                    if isinstance(values, str):
                        policy_obs["language"][key] = [[values]]
                    elif isinstance(values, np.ndarray):
                        policy_obs["language"][key] = [[str(item)] for item in values.reshape(-1)]
                    elif isinstance(values, (list, tuple)):
                        policy_obs["language"][key] = [[str(item)] for item in values]
                    else:
                        policy_obs["language"][key] = [[str(values)]]
                elif modality == "video":
                    policy_obs["video"][key] = np.expand_dims(
                        np.asarray(sim_obs[f"video.{key}"], dtype=np.uint8),
                        axis=0,
                    )
                else:
                    policy_obs["state"][key] = np.expand_dims(
                        np.asarray(sim_obs[f"state.{key}"], dtype=np.float32),
                        axis=0,
                    )
        return policy_obs

    @staticmethod
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

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        sim_obs, env_info = self.sim_env.reset(seed=seed, options=options)
        self.wrapper.reset()
        self._sim_obs = sim_obs
        self._episode_success = False
        self._episode_step = 0
        self._episode_stage2_conservative_penalty = 0.0
        self._last_stage2_step_conservative_penalty = 0.0
        self._schedule_state = None
        next_gym_obs, prep_info = self._build_schedule_state()
        return next_gym_obs, {
            "env_info": env_info,
            "reward_mode": self.reward_mode,
            **prep_info,
        }

    def _build_schedule_state(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self._sim_obs is None:
            raise RuntimeError("Current simulator observation is missing. Call reset() first.")
        self._schedule_state = self.wrapper.build_schedule_state(
            self._build_policy_observation(self._sim_obs)
        )
        schedule_obs = self._attach_step_progress_to_schedule_state()
        info = {
            "backbone_action_mask": self._schedule_state.backbone_action_mask,
        }
        scheduler_prepare_ms = getattr(self, "_current_scheduler_prepare_ms", None)
        if scheduler_prepare_ms is not None:
            info["scheduler_prepare_ms"] = float(scheduler_prepare_ms)
        return schedule_obs, info

    def _execute_schedule_action(
        self,
        action: np.ndarray | ScheduleAction,
    ) -> tuple[float, bool, bool, dict[str, Any]]:
        if self._sim_obs is None:
            raise RuntimeError("Current simulator observation is missing. Call reset() first.")
        if self._schedule_state is None:
            raise RuntimeError("Scheduler state is missing. Call reset() before step().")
        schedule_action = ScheduleAction.from_any(action)
        backbone_action = schedule_action.backbone_action
        action_head_action = schedule_action.action_head_action
        current_schedule_state = self._schedule_state
        robot_action, policy_info = self.wrapper.execute_schedule_action(
            self._build_policy_observation(self._sim_obs),
            schedule_action,
            current_schedule_state,
        )
        temporal_skip_info = dict(policy_info or {})

        next_sim_obs, env_reward, terminated, truncated, env_info = self.sim_env.step(
            self._to_sim_action(robot_action)
        )
        self._sim_obs = next_sim_obs
        self._episode_step += 1
        success = env_info.get("success")
        if success is None:
            final_info = env_info.get("final_info")
            if isinstance(final_info, dict):
                success = final_info.get("success")
        if success is not None:
            self._episode_success = bool(self._episode_success or np.any(success))

        resolved_backbone_action = BackboneAction.from_any(
            temporal_skip_info.get("backbone_action", backbone_action)
        )

        terminated = bool(terminated)
        truncated = bool(truncated)
        schedule_reward = self._compute_schedule_reward(
            schedule_state=current_schedule_state,
            backbone_action=backbone_action,
            resolved_backbone_action=resolved_backbone_action,
            action_head_action=action_head_action,
            success=bool(self._episode_success),
            episode_done=bool(terminated or truncated),
        )

        info = {
            "reward_mode": self.reward_mode,
            "backbone_action": int(backbone_action),
            "action_head_action": int(action_head_action),
            "resolved_backbone_action": int(resolved_backbone_action),
            "success": bool(self._episode_success),
            "env_reward": float(env_reward),
            "stage2_step_conservative_penalty": float(self._last_stage2_step_conservative_penalty),
            "stage2_episode_conservative_penalty": float(self._episode_stage2_conservative_penalty),
            "temporal_skip": temporal_skip_info,
        }
        self._schedule_state = None
        return float(schedule_reward), terminated, truncated, info

    def step(
        self,
        action: np.ndarray | ScheduleAction,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        reward, terminated, truncated, info = self._execute_schedule_action(action)
        if not terminated and not truncated:
            next_gym_obs, prep_info = self._build_schedule_state()
            info["backbone_action_mask"] = prep_info["backbone_action_mask"]
        else:
            next_gym_obs = {
                key: np.zeros(space.shape, dtype=space.dtype)
                for key, space in self.observation_space.spaces.items()
            }
        return next_gym_obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.sim_env.close()
