"""Private tracing and runtime instrumentation used by eval and tools."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from elegantvla.gr00t.core.env import TemporalSkipEnv
from elegantvla.gr00t.entrypoints._runtime import MultiTaskTemporalSkipEnv
from elegantvla.gr00t.core.rl_policy import RLSchedulePolicy
from elegantvla.gr00t.core.types import ActionHeadAction, BackboneAction, ScheduleAction
from elegantvla.gr00t.core.wrapper import TemporalSkipPolicyWrapper


@dataclass
class TracingRLSchedulePolicy(RLSchedulePolicy):
    last_action: ScheduleAction | None = None
    last_backbone_action_mask: np.ndarray | None = None
    last_backbone_action_probs: np.ndarray | None = None
    last_action_head_action_probs: np.ndarray | None = None

    @staticmethod
    def _entropy(probs: np.ndarray | None) -> float | None:
        if probs is None:
            return None
        array = np.asarray(probs, dtype=np.float32).reshape(-1)
        if array.size == 0:
            return None
        clipped = np.clip(array, 1e-12, 1.0)
        return float(-np.sum(array * np.log(clipped)))

    @staticmethod
    def _named_probability_fields(
        prefix: str,
        enum_cls: Any,
        probs: np.ndarray | None,
    ) -> dict[str, float | None]:
        fields: dict[str, float | None] = {}
        array = None if probs is None else np.asarray(probs, dtype=np.float32).reshape(-1)
        for enum_value in enum_cls:
            key = f"{prefix}_{str(enum_value.name).lower()}"
            value = None
            if array is not None and int(enum_value) < array.size:
                value = float(array[int(enum_value)])
            fields[key] = value
        return fields

    @staticmethod
    def _selected_probability(
        probs: np.ndarray | None,
        action: int | None,
    ) -> float | None:
        if probs is None or action is None:
            return None
        array = np.asarray(probs, dtype=np.float32).reshape(-1)
        index = int(action)
        if index < 0 or index >= array.size:
            return None
        return float(array[index])

    def select_action(
        self,
        observation: dict[str, np.ndarray],
        backbone_action_mask: np.ndarray | None = None,
    ) -> ScheduleAction:
        action, effective_backbone_mask, backbone_probs, action_head_probs = self._predict_action(
            observation,
            backbone_action_mask=backbone_action_mask,
        )
        self.last_action = action
        self.last_backbone_action_mask = effective_backbone_mask
        self.last_backbone_action_probs = backbone_probs
        self.last_action_head_action_probs = action_head_probs
        return action

    def last_trace(self) -> dict[str, Any]:
        backbone_action = (
            BackboneAction.REFRESH
            if self.last_action is None
            else self.last_action.backbone_action
        )
        action_head_action = (
            ActionHeadAction.REFRESH
            if self.last_action is None
            else self.last_action.action_head_action
        )
        backbone_entropy = self._entropy(self.last_backbone_action_probs)
        action_head_entropy = self._entropy(self.last_action_head_action_probs)
        return {
            "backbone_action": int(backbone_action),
            "backbone_action_name": backbone_action.name,
            "action_head_action": int(action_head_action),
            "action_head_action_name": action_head_action.name,
            "backbone_action_mask": (
                None
                if self.last_backbone_action_mask is None
                else self.last_backbone_action_mask.astype(int).tolist()
            ),
            "backbone_action_probs": (
                None
                if self.last_backbone_action_probs is None
                else self.last_backbone_action_probs.tolist()
            ),
            "action_head_action_probs": (
                None
                if self.last_action_head_action_probs is None
                else self.last_action_head_action_probs.tolist()
            ),
            "selected_backbone_action_prob": self._selected_probability(
                self.last_backbone_action_probs,
                None if self.last_action is None else int(self.last_action.backbone_action),
            ),
            "selected_action_head_action_prob": self._selected_probability(
                self.last_action_head_action_probs,
                None if self.last_action is None else int(self.last_action.action_head_action),
            ),
            "backbone_entropy": backbone_entropy,
            "action_head_entropy": action_head_entropy,
            "policy_entropy": (
                None
                if backbone_entropy is None or action_head_entropy is None
                else float(backbone_entropy + action_head_entropy)
            ),
            **self._named_probability_fields(
                "backbone_prob",
                BackboneAction,
                self.last_backbone_action_probs,
            ),
            **self._named_probability_fields(
                "action_head_prob",
                ActionHeadAction,
                self.last_action_head_action_probs,
            ),
        }


class InstrumentedTemporalSkipPolicyWrapper(TemporalSkipPolicyWrapper):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.last_prepare_diagnostics: dict[str, Any] = {}
        self.last_step_diagnostics: dict[str, Any] = {}

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self.last_prepare_diagnostics = {}
        self.last_step_diagnostics = {}
        return super().reset(options)

    def _prepare_first_block_precompute(
        self,
        observation: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        started_at = time.perf_counter()
        prepare_inputs_started_at = time.perf_counter()
        prepared = self.adapter.prepare_inputs(observation)
        prepare_inputs_ms = float((time.perf_counter() - prepare_inputs_started_at) * 1000.0)
        probe_started_at = time.perf_counter()
        with torch.inference_mode():
            probe_output, action_input = self.adapter.get_first_block_gate_probe(
                prepared["collated_inputs"]
            )
        first_block_probe_ms = float((time.perf_counter() - probe_started_at) * 1000.0)
        diagnostics = {
            "prepare_inputs_ms": prepare_inputs_ms,
            "first_block_probe_ms": first_block_probe_ms,
            "first_block_precompute_ms": float((time.perf_counter() - started_at) * 1000.0),
        }
        return {
            "prepared": prepared,
            "probe_output": probe_output,
            "action_input": action_input,
            "_instrumentation": diagnostics,
        }

    def _prepare_schedule_observation(
        self,
        observation: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        started_at = time.perf_counter()
        execution_ctx = None
        prepare_diagnostics: dict[str, Any] = {}
        used_cached_similarity = self.ctrl._no_image_budget > 0
        build_schedule_started_at = time.perf_counter()
        if used_cached_similarity:
            schedule_observation = self.ctrl.build_schedule_observation(
                observation,
                feature_similarity=self._last_feature_similarity,
            )
        else:
            execution_ctx = self._prepare_first_block_precompute(observation)
            prepare_diagnostics.update(dict(execution_ctx.get("_instrumentation") or {}))
            build_schedule_started_at = time.perf_counter()
            schedule_observation = self.ctrl.build_schedule_observation(
                observation,
                current_gate_feature=execution_ctx["probe_output"]["hidden_states"],
            )
        build_schedule_observation_ms = float(
            (time.perf_counter() - build_schedule_started_at) * 1000.0
        )
        self._last_feature_similarity = float(schedule_observation.feature_similarity)
        prepare_diagnostics.update(
            {
                "schedule_build_observation_ms": build_schedule_observation_ms,
                "schedule_prepare_total_ms": float((time.perf_counter() - started_at) * 1000.0),
                "schedule_used_cached_similarity": bool(used_cached_similarity),
                "schedule_has_execution_ctx": bool(execution_ctx is not None),
            }
        )
        self.last_prepare_diagnostics = prepare_diagnostics
        return schedule_observation, execution_ctx

    def _run_skip_step(
        self,
        observation: dict[str, Any],
        *,
        backbone_action,
        action_head_action,
        precomputed: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        started_at = time.perf_counter()
        resolved_backbone_action = self.ctrl.resolve_backbone_action(backbone_action)
        use_backbone_cache = resolved_backbone_action != BackboneAction.REFRESH
        image_free_skip = use_backbone_cache and self.ctrl._no_image_budget > 0
        action_head_reuse_strategy = action_head_action.reuse_strategy
        effective_use_backbone_cache = use_backbone_cache
        used_middle_cache = False
        prepare_inputs_ms = 0.0
        first_block_probe_ms = 0.0
        first_block_precompute_ms = 0.0
        used_precomputed_ctx = precomputed is not None
        if precomputed is None:
            if image_free_skip:
                prepare_inputs_started_at = time.perf_counter()
                prepared = self.adapter.prepare_inputs(observation, action_only=True)
                prepare_inputs_ms = float((time.perf_counter() - prepare_inputs_started_at) * 1000.0)
                probe_output = None
                action_input = None
            else:
                precomputed = self._prepare_first_block_precompute(observation)
                instrumentation = dict(precomputed.get("_instrumentation") or {})
                prepare_inputs_ms = float(instrumentation.get("prepare_inputs_ms", 0.0))
                first_block_probe_ms = float(instrumentation.get("first_block_probe_ms", 0.0))
                first_block_precompute_ms = float(
                    instrumentation.get(
                        "first_block_precompute_ms",
                        prepare_inputs_ms + first_block_probe_ms,
                    )
                )
                prepared = precomputed["prepared"]
                probe_output = precomputed["probe_output"]
                action_input = precomputed["action_input"]
        else:
            prepared = precomputed["prepared"]
            probe_output = precomputed["probe_output"]
            action_input = precomputed["action_input"]

        continue_backbone_ms = 0.0
        run_action_head_ms = 0.0
        with torch.inference_mode():
            if use_backbone_cache:
                if image_free_skip or probe_output is None:
                    backbone_cache = self.ctrl._backbone_cache
                    backbone_output = None
                else:
                    continue_backbone_started_at = time.perf_counter()
                    backbone_output, refreshed_middle_cache, used_middle_cache = (
                        self.adapter.continue_backbone_from_middle_cache(
                            probe_output,
                            self.ctrl._backbone_middle_cache,
                        )
                    )
                    continue_backbone_ms = float(
                        (time.perf_counter() - continue_backbone_started_at) * 1000.0
                    )
                    if refreshed_middle_cache is not None:
                        self.ctrl._backbone_middle_cache = refreshed_middle_cache
                        effective_use_backbone_cache = False
                    backbone_cache = None
            else:
                continue_backbone_started_at = time.perf_counter()
                if self.static_inference_cache:
                    backbone_output, refreshed_middle_cache = (
                        self.adapter.continue_backbone_from_probe_with_middle_cache(probe_output)
                    )
                    self.ctrl._backbone_middle_cache = refreshed_middle_cache
                else:
                    backbone_output = self.adapter.continue_backbone_from_probe(probe_output)
                    self.ctrl._backbone_middle_cache = None
                continue_backbone_ms = float(
                    (time.perf_counter() - continue_backbone_started_at) * 1000.0
                )
                backbone_cache = None

            run_action_head_started_at = time.perf_counter()
            model_pred, cache_to_use = self.adapter.run_action_head(
                backbone_cache=backbone_cache,
                backbone_output=backbone_output,
                action_input=action_input,
                collated_inputs=prepared["collated_inputs"] if image_free_skip else None,
                reuse_strategy=action_head_reuse_strategy,
            )
            run_action_head_ms = float((time.perf_counter() - run_action_head_started_at) * 1000.0)
            if (not effective_use_backbone_cache) or (use_backbone_cache and not image_free_skip):
                self.ctrl._backbone_cache = cache_to_use

        decode_started_at = time.perf_counter()
        action = self.adapter.decode_action(model_pred, prepared["state"])
        decode_action_ms = float((time.perf_counter() - decode_started_at) * 1000.0)
        record_runtime_state_started_at = time.perf_counter()
        self.ctrl.record_motion(observation)
        gate_feature = None if probe_output is None else probe_output["hidden_states"]
        step_info = self.ctrl.record_step(
            use_backbone_cache=effective_use_backbone_cache,
            image_free_skip=image_free_skip,
            gate_feature=gate_feature,
            backbone_action=(
                resolved_backbone_action
                if effective_use_backbone_cache
                else BackboneAction.REFRESH
            ),
        )
        record_runtime_state_ms = float(
            (time.perf_counter() - record_runtime_state_started_at) * 1000.0
        )
        step_info.update(
            {
                "use_backbone_cache": bool(effective_use_backbone_cache),
                "image_free_skip": bool(image_free_skip),
                "backbone_middle_cache_used": bool(used_middle_cache),
            }
        )
        diagnostics = dict(step_info)
        diagnostics.update(
            {
                "policy_step_ms": float((time.perf_counter() - started_at) * 1000.0),
                "policy_prepare_inputs_ms": prepare_inputs_ms,
                "policy_first_block_probe_ms": first_block_probe_ms,
                "policy_first_block_precompute_ms": first_block_precompute_ms,
                "policy_continue_backbone_ms": continue_backbone_ms,
                "policy_run_action_head_ms": run_action_head_ms,
                "policy_decode_action_ms": decode_action_ms,
                "policy_record_runtime_state_ms": record_runtime_state_ms,
                "policy_used_precomputed_ctx": bool(used_precomputed_ctx),
            }
        )
        self.last_step_diagnostics = diagnostics
        return action, diagnostics


class FullComputeInstrumentedTemporalSkipPolicyWrapper(
    InstrumentedTemporalSkipPolicyWrapper
):
    """Profile full computation without paying feature-similarity scheduler overhead."""

    def _prepare_schedule_observation(
        self,
        observation: dict[str, Any],
    ) -> tuple[Any, None]:
        started_at = time.perf_counter()
        schedule_observation = self.ctrl.build_schedule_observation(
            observation,
            feature_similarity=0.0,
        )
        total_ms = float((time.perf_counter() - started_at) * 1000.0)
        self._last_feature_similarity = 0.0
        self.last_prepare_diagnostics = {
            "schedule_build_observation_ms": total_ms,
            "schedule_prepare_total_ms": total_ms,
            "schedule_used_cached_similarity": False,
            "schedule_has_execution_ctx": False,
            "full_compute_control": True,
        }
        return schedule_observation, None


class InstrumentedTemporalSkipEnv(TemporalSkipEnv):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._current_scheduler_prepare_ms = 0.0
        self._current_schedule_prepare_diagnostics: dict[str, Any] = {}

    def _build_schedule_state(self) -> tuple[dict[str, Any], dict[str, Any]]:
        started_at = time.perf_counter()
        if self._sim_obs is None:
            raise RuntimeError("Current simulator observation is missing. Call reset() first.")
        policy_observation_started_at = time.perf_counter()
        policy_observation = self._build_policy_observation(self._sim_obs)
        policy_observation_ms = float(
            (time.perf_counter() - policy_observation_started_at) * 1000.0
        )
        wrapper_started_at = time.perf_counter()
        self._schedule_state = self.wrapper.build_schedule_state(policy_observation)
        schedule_obs = self._attach_step_progress_to_schedule_state()
        wrapper_build_ms = float((time.perf_counter() - wrapper_started_at) * 1000.0)
        self._current_scheduler_prepare_ms = float((time.perf_counter() - started_at) * 1000.0)
        prepare_diagnostics = dict(getattr(self.wrapper, "last_prepare_diagnostics", {}) or {})
        prepare_diagnostics.update(
            {
                "schedule_policy_observation_ms": policy_observation_ms,
                "schedule_wrapper_build_ms": wrapper_build_ms,
                "scheduler_prepare_ms": self._current_scheduler_prepare_ms,
            }
        )
        self._current_schedule_prepare_diagnostics = prepare_diagnostics
        return schedule_obs, {
            "backbone_action_mask": self._schedule_state.backbone_action_mask,
            "full_schedule_observation": schedule_obs,
            "scheduler_prepare_ms": self._current_scheduler_prepare_ms,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        info = dict(info)
        info["temporal_skip"] = {"reset": True}
        return observation, info

    def step(
        self,
        action,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        scheduler_prepare_ms = float(self._current_scheduler_prepare_ms)
        schedule_prepare_diagnostics = dict(self._current_schedule_prepare_diagnostics)
        next_obs, reward, terminated, truncated, info = super().step(action)
        info = dict(info)
        step_diagnostics = dict(getattr(self.wrapper, "last_step_diagnostics", {}) or {})
        temporal_skip = dict(info.get("temporal_skip") or {})
        temporal_skip.update(schedule_prepare_diagnostics)
        temporal_skip.update(step_diagnostics)
        info["schedule_reward"] = float(reward)
        info["scheduler_prepare_ms"] = scheduler_prepare_ms
        info["decision_e2e_ms"] = float(
            scheduler_prepare_ms + float(step_diagnostics.get("policy_step_ms", 0.0))
        )
        info["decision_full_e2e_ms"] = float(
            scheduler_prepare_ms + float(temporal_skip.get("execute_total_ms", 0.0))
        )
        info["temporal_skip"] = temporal_skip
        return next_obs, reward, terminated, truncated, info

    def _execute_schedule_action(
        self,
        action,
    ) -> tuple[float, bool, bool, dict[str, Any]]:
        if self._sim_obs is None:
            raise RuntimeError("Current simulator observation is missing. Call reset() first.")
        if self._schedule_state is None:
            raise RuntimeError("Scheduler state is missing. Call reset() before step().")
        execute_started_at = time.perf_counter()
        schedule_action = ScheduleAction.from_any(action)
        backbone_action = schedule_action.backbone_action
        action_head_action = schedule_action.action_head_action
        current_schedule_state = self._schedule_state
        policy_observation_started_at = time.perf_counter()
        policy_observation = self._build_policy_observation(self._sim_obs)
        execute_policy_observation_ms = float(
            (time.perf_counter() - policy_observation_started_at) * 1000.0
        )
        wrapper_execute_started_at = time.perf_counter()
        robot_action, policy_info = self.wrapper.execute_schedule_action(
            policy_observation,
            schedule_action,
            current_schedule_state,
        )
        wrapper_execute_ms = float((time.perf_counter() - wrapper_execute_started_at) * 1000.0)
        temporal_skip_info = dict(policy_info or {})

        sim_env_step_started_at = time.perf_counter()
        next_sim_obs, env_reward, terminated, truncated, env_info = self.sim_env.step(
            self._to_sim_action(robot_action)
        )
        sim_env_step_ms = float((time.perf_counter() - sim_env_step_started_at) * 1000.0)
        self._sim_obs = next_sim_obs
        self._episode_step += 1
        bookkeeping_started_at = time.perf_counter()
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
        execute_post_env_bookkeeping_ms = float(
            (time.perf_counter() - bookkeeping_started_at) * 1000.0
        )
        temporal_skip_info.update(
            {
                "execute_policy_observation_ms": execute_policy_observation_ms,
                "execute_wrapper_ms": wrapper_execute_ms,
                "sim_env_step_ms": sim_env_step_ms,
                "execute_post_env_bookkeeping_ms": execute_post_env_bookkeeping_ms,
                "execute_total_ms": float((time.perf_counter() - execute_started_at) * 1000.0),
            }
        )
        info = {
            "backbone_action": int(backbone_action),
            "action_head_action": int(action_head_action),
            "resolved_backbone_action": int(resolved_backbone_action),
            "success": bool(self._episode_success),
            "env_reward": float(env_reward),
            "temporal_skip": temporal_skip_info,
        }
        self._schedule_state = None
        return float(schedule_reward), terminated, truncated, info


class InstrumentedMultiTaskTemporalSkipEnv(
    InstrumentedTemporalSkipEnv,
    MultiTaskTemporalSkipEnv,
):
    """Multi-task env with the same timing surface as the single-task evaluator."""

    pass
