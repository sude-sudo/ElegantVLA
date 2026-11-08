"""OpenPI schedule preparation and execution, matching the GR00T wrapper API."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from elegantvla.openpi.core.adapter import OpenPIPolicyTemporalSkipAdapter
from elegantvla.openpi.core.config import TemporalSkipConfig
from elegantvla.openpi.core.rule_policy import RuleSchedulePolicy
from elegantvla.openpi.core.types import (
    ActionHeadAction,
    BackboneAction,
    ScheduleAction,
    ScheduleState,
)
from elegantvla.scheduling.controller import SkipController


class TemporalSkipPolicyWrapper:
    def __init__(self, model, *, config=None, adapter=None, schedule_policy=None):
        self.config = config or TemporalSkipConfig(enabled=True)
        self.adapter = adapter or OpenPIPolicyTemporalSkipAdapter(
            model, first_layers=self.config.first_layers, last_layers=self.config.last_layers
        )
        self.ctrl = SkipController(
            enabled=self.config.enabled,
            max_reuse_steps=self.config.max_reuse_steps,
            speed_smoothing_window=self.config.speed_smoothing_window,
        )
        self.schedule_policy = schedule_policy or RuleSchedulePolicy(
            gate_threshold=self.config.gate_threshold,
            free1_similarity_threshold=self.config.free1_similarity_threshold,
            free2_similarity_threshold=self.config.free2_similarity_threshold,
            free3_similarity_threshold=self.config.free3_similarity_threshold,
            action_head_selection_mode=self.config.action_head_selection_mode,
            ah_speed_threshold=self.config.ah_speed_threshold,
            ah_secondary_threshold=self.config.ah_secondary_threshold,
            action_head_eef_rotation_speed_weight=self.config.action_head_eef_rotation_speed_weight,
            action_head_gripper_speed_weight=self.config.action_head_gripper_speed_weight,
        )
        self._cache_key = None
        self._last_feature_similarity = 0.0
        self._step_index = 0
        self._pending_context = None

    def reset(self):
        self.ctrl.reset()
        self._cache_key = None
        self._last_feature_similarity = 0.0
        self._step_index = 0
        self._pending_context = None

    @torch.inference_mode()
    def build_schedule_state(
        self, observation, *, motion_observation=None, prompt_key=None, noise=None, num_steps=10
    ) -> ScheduleState:
        key = self.adapter.cache_key(observation, prompt_key=prompt_key)
        if key != self._cache_key:
            self.reset()
            self._cache_key = key
        prepared = self.adapter.prepare_inputs(observation, noise=noise, num_steps=num_steps)
        mask = self.ctrl.build_backbone_action_mask()
        image_free = self.ctrl._no_image_budget > 0 and bool(mask[int(BackboneAction.REUSE)])
        probe = None if image_free else self.adapter.get_first_block_gate_probe(prepared)
        motion = motion_observation or {}
        schedule_observation = self.ctrl.build_schedule_observation(
            motion,
            current_gate_feature=None if probe is None else probe.gate_feature,
            feature_similarity=self._last_feature_similarity,
        )
        if self.config.max_episode_steps is not None:
            schedule_observation = replace(
                schedule_observation, step_progress=self._step_index / self.config.max_episode_steps
            )
        self._last_feature_similarity = schedule_observation.feature_similarity
        context = {"prepared": prepared, "probe": probe, "motion": motion, "observation": observation}
        self._pending_context = context
        return ScheduleState(
            schedule_obs=schedule_observation.as_gym_observation(),
            backbone_action_mask=mask.astype(np.float32, copy=False),
            execution_ctx=context,
        )

    @torch.inference_mode()
    def execute_schedule_action(self, observation, schedule_action, schedule_state):
        context = schedule_state.execution_ctx
        if context is None or context is not self._pending_context or context["observation"] is not observation:
            raise ValueError("Schedule state is stale, already consumed, or belongs to another observation.")
        self._pending_context = None
        action = ScheduleAction.from_any(schedule_action)
        backbone_action = self.ctrl.resolve_backbone_action(action.backbone_action)
        action_head_action = action.action_head_action if self.config.enabled else ActionHeadAction.REFRESH
        reuse = backbone_action != BackboneAction.REFRESH
        image_free = reuse and self.ctrl._no_image_budget > 0
        prepared, probe = context["prepared"], context["probe"]
        if probe is None and not image_free:
            probe = self.adapter.get_first_block_gate_probe(prepared)
        used_middle_cache = False
        anchor = None
        if image_free:
            output = self.ctrl._backbone_cache
        elif reuse:
            output, anchor, used_middle_cache = self.adapter.continue_backbone_from_middle_cache(
                probe, self.ctrl._backbone_middle_cache
            )
            if anchor is not None:
                reuse = False
                backbone_action = BackboneAction.REFRESH
        else:
            output, anchor = self.adapter.continue_backbone_from_probe_with_middle_cache(probe)
        # Commit cache state only after both backbone and action head succeed.
        actions, action_info = self.adapter.run_action_head(output, prepared, action_head_action)
        if anchor is not None:
            self.ctrl._backbone_middle_cache = anchor
        if not image_free:
            self.ctrl._backbone_cache = output
        self.ctrl.record_motion(context["motion"])
        info = self.ctrl.record_step(
            use_backbone_cache=reuse,
            image_free_skip=image_free,
            gate_feature=None if probe is None else probe.gate_feature,
            backbone_action=backbone_action,
        )
        self._step_index += 1
        computed = (
            0
            if image_free
            else (self.adapter.first_layers + self.adapter.last_layers if used_middle_cache else self.adapter.depth)
        )
        path = "image_free" if image_free else ("boundary" if used_middle_cache else "refresh")
        info.update(action_info)
        info.update(
            {
                "path": path,
                "use_backbone_cache": reuse,
                "image_free_skip": image_free,
                "backbone_middle_cache_used": used_middle_cache,
                "backbone_computed_blocks": computed,
                "backbone_total_blocks": self.adapter.depth,
                "feature_similarity": self._last_feature_similarity,
                "no_image_budget": self.ctrl._no_image_budget,
                "consecutive_reuse_steps": self.ctrl._consecutive_reuse_steps,
            }
        )
        return actions, info

    def infer(self, observation, **kwargs):
        state = self.build_schedule_state(observation, **kwargs)
        action = self.schedule_policy.select_action(state.schedule_obs, state.backbone_action_mask)
        return self.execute_schedule_action(observation, action, state)
