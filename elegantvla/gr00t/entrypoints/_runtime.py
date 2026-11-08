"""Shared runtime builders for entrypoints and analysis tools."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch

from gr00t.eval.rollout_policy import MultiStepConfig, VideoConfig, WrapperConfigs, create_eval_env
from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.policy.gr00t_policy import Gr00tPolicy

from elegantvla.gr00t.core.env import TemporalSkipEnv
from elegantvla.gr00t.core.rule_policy import RuleSchedulePolicy
from elegantvla.gr00t.core.wrapper import TemporalSkipPolicyWrapper


GOOGLE_ALL_ENV_NAMES = [
    "simpler_env_google/google_robot_pick_coke_can",
    "simpler_env_google/google_robot_pick_object",
    "simpler_env_google/google_robot_move_near",
    "simpler_env_google/google_robot_open_drawer",
    "simpler_env_google/google_robot_close_drawer",
    "simpler_env_google/google_robot_place_in_closed_drawer",
]

WIDOWX_ALL_ENV_NAMES = [
    "simpler_env_widowx/widowx_spoon_on_towel",
    "simpler_env_widowx/widowx_carrot_on_plate",
    "simpler_env_widowx/widowx_stack_cube",
    "simpler_env_widowx/widowx_put_eggplant_in_basket",
    "simpler_env_widowx/widowx_put_eggplant_in_sink",
    "simpler_env_widowx/widowx_open_drawer",
    "simpler_env_widowx/widowx_close_drawer",
]

TASK_GROUPS = {
    "google_all": GOOGLE_ALL_ENV_NAMES,
    "widowx_all": WIDOWX_ALL_ENV_NAMES,
}


class MultiTaskTemporalSkipEnv(TemporalSkipEnv):
    """Round-robin across same-embodiment tasks while reusing one wrapper."""

    def __init__(
        self,
        *,
        env_names: list[str],
        sim_env_builder: Any,
        wrapper: TemporalSkipPolicyWrapper,
        reward_policy: RuleSchedulePolicy,
        reward_mode: str = "stage1",
        stage2_efficiency_penalty_mode: str = "step",
        stage2_backbone_conservative_penalty: float = 0.05,
        stage2_action_head_conservative_penalty: float = 0.025,
        **env_kwargs: Any,
    ) -> None:
        if not env_names:
            raise ValueError("env_names must not be empty")
        self.env_names = list(env_names)
        self._sim_env_builder = sim_env_builder
        self._sim_envs: dict[str, Any] = {}
        self._next_env_index = 0
        super().__init__(
            sim_env=self._get_or_create_sim_env(self.env_names[0]),
            wrapper=wrapper,
            reward_policy=reward_policy,
            reward_mode=reward_mode,
            stage2_efficiency_penalty_mode=stage2_efficiency_penalty_mode,
            stage2_backbone_conservative_penalty=stage2_backbone_conservative_penalty,
            stage2_action_head_conservative_penalty=stage2_action_head_conservative_penalty,
            **env_kwargs,
        )

    def _get_or_create_sim_env(self, env_name: str) -> Any:
        sim_env = self._sim_envs.get(env_name)
        if sim_env is None:
            sim_env = self._sim_env_builder(env_name)
            self._sim_envs[env_name] = sim_env
        return sim_env

    def _next_env_name(self) -> str:
        env_name = self.env_names[self._next_env_index % len(self.env_names)]
        self._next_env_index += 1
        return env_name

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        env_name = self._next_env_name()
        self.sim_env = self._get_or_create_sim_env(env_name)
        observation, info = super().reset(seed=seed, options=options)
        info = dict(info)
        info["env_name"] = env_name
        return observation, info

    def close(self) -> None:
        for sim_env in self._sim_envs.values():
            sim_env.close()
        self._sim_envs.clear()


def require_existing_model_path(model_path: str) -> None:
    if model_path.startswith("/") and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist.")


def seed_episode_rng(seed: int | None) -> None:
    if seed is None:
        return
    resolved_seed = int(seed)
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    torch.manual_seed(resolved_seed)


def _extract_scalar_observation(
    observation: dict[str, Any],
    key: str,
    *,
    default: float = float("nan"),
) -> float:
    if key not in observation:
        return float(default)
    return float(np.asarray(observation[key]).reshape(-1)[0])


def extract_scheduler_observation(observation: dict[str, Any]) -> dict[str, float]:
    return {
        "feature_similarity": _extract_scalar_observation(observation, "feature_similarity"),
        "gripper_speed": _extract_scalar_observation(observation, "gripper_speed"),
        "eef_translation_speed": _extract_scalar_observation(observation, "eef_translation_speed"),
        "eef_rotation_speed": _extract_scalar_observation(observation, "eef_rotation_speed"),
        "step_progress": _extract_scalar_observation(observation, "step_progress"),
    }


def resolve_task_group_env_names(task_group: str | None) -> list[str]:
    resolved_group = "" if task_group is None else str(task_group).strip().lower()
    if not resolved_group:
        return []
    env_names = TASK_GROUPS.get(resolved_group)
    if env_names is None:
        raise ValueError(
            f"Unknown task_group={task_group!r}. Expected one of: {sorted(TASK_GROUPS)}"
        )
    return list(env_names)


def resolve_env_names(cfg: Any) -> list[str]:
    env_names = resolve_task_group_env_names(getattr(cfg, "task_group", ""))
    if env_names:
        return env_names
    return [str(cfg.env_name)]


def _build_sim_env(cfg: Any, *, env_name: str) -> Any:
    wrapper_configs = WrapperConfigs(
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
    )
    return create_eval_env(
        env_name=env_name,
        env_idx=0,
        total_n_envs=1,
        wrapper_configs=wrapper_configs,
    )


def build_temporal_skip_env(
    cfg: Any,
    *,
    device_attr: str = "gr00t_device",
    wrapper_cls: type[TemporalSkipPolicyWrapper] = TemporalSkipPolicyWrapper,
    env_cls: type[TemporalSkipEnv] = TemporalSkipEnv,
    multi_task_env_cls: type[MultiTaskTemporalSkipEnv] = MultiTaskTemporalSkipEnv,
) -> TemporalSkipEnv:
    env_names = resolve_env_names(cfg)
    primary_env_name = env_names[0]
    base_policy = Gr00tPolicy(
        embodiment_tag=get_embodiment_tag_from_env_name(primary_env_name),
        model_path=cfg.model_path,
        device=getattr(cfg, device_attr),
        strict=cfg.strict,
    )
    reward_policy = RuleSchedulePolicy(
        gate_threshold=cfg.gate_threshold,
        free1_similarity_threshold=cfg.free1_similarity_threshold,
        free2_similarity_threshold=cfg.free2_similarity_threshold,
        free3_similarity_threshold=cfg.free3_similarity_threshold,
        action_head_selection_mode=cfg.action_head_cache_selection,
        ah_speed_threshold=cfg.ah_speed_threshold,
        ah_secondary_threshold=cfg.ah_secondary_threshold,
        action_head_eef_rotation_speed_weight=cfg.action_head_eef_rotation_speed_weight,
        action_head_gripper_speed_weight=cfg.action_head_gripper_speed_weight,
    )
    wrapper = wrapper_cls(
        base_policy,
        speed_smoothing_window=cfg.action_head_gripper_speed_smoothing_window,
        compile_action_head=cfg.compile_action_head,
        compile_mode=cfg.compile_mode,
        static_inference_cache=getattr(cfg, "static_inference_cache", True),
        strict=cfg.strict,
    )
    env_kwargs = {
        "reward_mode": getattr(cfg, "reward_mode", "stage1"),
        "stage2_efficiency_penalty_mode": getattr(
            cfg,
            "stage2_efficiency_penalty_mode",
            "step",
        ),
        "stage2_backbone_conservative_penalty": getattr(
            cfg,
            "stage2_backbone_conservative_penalty",
            0.05,
        ),
        "stage2_action_head_conservative_penalty": getattr(
            cfg,
            "stage2_action_head_conservative_penalty",
            0.025,
        ),
    }
    if len(env_names) == 1:
        return env_cls(
            sim_env=_build_sim_env(cfg, env_name=primary_env_name),
            wrapper=wrapper,
            reward_policy=reward_policy,
            **env_kwargs,
        )
    return multi_task_env_cls(
        env_names=env_names,
        sim_env_builder=lambda env_name: _build_sim_env(cfg, env_name=env_name),
        wrapper=wrapper,
        reward_policy=reward_policy,
        **env_kwargs,
    )
